# app.py
import os
import time
import subprocess
import json
from flask import Flask, render_template, request, send_from_directory, jsonify
from flask_cors import CORS
import lane_detection
import damage_detection

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def _extract_gps_with_exiftool(file_path):
    """Return {"latitude": float, "longitude": float} if found via exiftool, else None.

    Requires exiftool to be installed and available on PATH.
    """
    try:
        # Try common Windows names too
        candidates = ["exiftool", "exiftool.exe", "exiftool(-k).exe"]
        result = None
        for exe in candidates:
            try:
                # -n for numeric values; -j for JSON; request a few relevant tags
                # Large file support ensures no early exit on big MP4s
                result = subprocess.run(
                    [
                        exe,
                        "-n",
                        "-j",
                        "-api",
                        "largefilesupport=1",
                        "-GPSLatitude",
                        "-GPSLongitude",
                        "-QuickTime:Location",
                        file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                # If process executed, break regardless of return code; we'll check code next
                break
            except FileNotFoundError:
                result = None
                continue
        if result is None or result.returncode != 0:
            return None
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "[]")
        if not isinstance(data, list) or not data:
            return None
        tags = data[0] or {}

        # Primary: explicit GPSLatitude/GPSLongitude
        lat = tags.get("GPSLatitude")
        lon = tags.get("GPSLongitude")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            return {"latitude": float(lat), "longitude": float(lon)}

        # Fallback: QuickTime:Location in ISO6709 format e.g. +37.7858-122.4064+000.00/
        loc = tags.get("QuickTime:Location") or tags.get("Location")
        if isinstance(loc, str):
            s = loc.strip()
            # Parse +lat+lon or +lat-lon variants; find sign changes
            try:
                # Find boundaries by sign characters after first char
                # Example: +37.7858-122.4064... → split between first and second sign
                signs = [i for i, ch in enumerate(s[1:], start=1) if ch in "+-" ]
                if signs:
                    split_idx = signs[0]
                    lat_s = s[:split_idx]
                    lon_s = s[split_idx:]
                    # Strip trailing altitude if present
                    if lon_s.endswith('/'):
                        lon_s = lon_s[:-1]
                    # Also handle +...+alt or -...+alt by cutting at third sign if present
                    third_signs = [i for i, ch in enumerate(lon_s[1:], start=1) if ch in "+-" ]
                    if third_signs:
                        lon_s = lon_s[:third_signs[0]]
                    lat_v = float(lat_s)
                    lon_v = float(lon_s)
                    return {"latitude": lat_v, "longitude": lon_v}
            except Exception:
                pass

        return None
    except Exception:
        return None

app = Flask(__name__, static_folder="static")
CORS(app)


@app.route('/')
def index1():
    return render_template('index1.html')


@app.route('/pothole')
def pothole():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Basic validation
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    # --- DEBUG: log incoming form/file keys (temporary) ---
    try:
        print("---- /upload called ----")
        print("Form keys:", list(request.form.keys()))
        for k in request.form.keys():
            print(f" form[{k}] = {request.form.get(k)}")
        print("Files keys:", list(request.files.keys()))
    except Exception as e:
        print("Debug print error:", e)
    # --------------------------------------------------------

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file with timestamp prefix to avoid collisions
    filename = file.filename
    ts = int(time.time())
    saved_name = f"{ts}_{filename}"
    file_path = os.path.join(UPLOAD_FOLDER, saved_name)
    file.save(file_path)

    # Try to extract GPS immediately from uploaded MP4 using exiftool
    exiftool_gps = _extract_gps_with_exiftool(file_path)

    # IMPORTANT: do NOT accept GPS from the browser form.
    # We will always rely on extracting GPS from video metadata (ffprobe) in damage_detection.
    # So we intentionally ignore any latitude/longitude fields in the request.

    output_filename = f"processed_{ts}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    process_type = request.form.get("process_type", "lane")

    try:
        if process_type == "lane":
            # lane detection (existing behavior)
            lane_detection.process_video(file_path, output_path)
            # Use exiftool GPS for lane processing as well (if found).
            metadata = {"gps": None, "pothole_images": []}
            if exiftool_gps:
                metadata["gps"] = exiftool_gps
            else:
                # Fallback: read GPS from lane sidecar JSON if created
                try:
                    json_path = os.path.splitext(output_path)[0] + "_metadata.json"
                    if os.path.exists(json_path):
                        with open(json_path, "r") as jf:
                            sidecar = json.load(jf)
                        gps = sidecar.get("gps") if isinstance(sidecar, dict) else None
                        if isinstance(gps, dict):
                            lat = gps.get("latitude")
                            lon = gps.get("longitude")
                            if lat is not None and lon is not None:
                                metadata["gps"] = {"latitude": float(lat), "longitude": float(lon)}
                except Exception:
                    pass
        else:
            # DAMAGE processing: do NOT pass gps from the browser.
            # Let damage_detection.extract GPS from the video metadata itself by setting overlay_gps=True.
            # Note: damage_detection.process_video returns a dict in the improved script.
            result = damage_detection.process_video(file_path, output_path, overlay_gps=True, save_sidecar=True)
            if isinstance(result, dict):
                metadata = result
                metadata.setdefault("gps", None)
                metadata.setdefault("pothole_images", [])
            else:
                # fallback if older damage_detection returns None
                metadata = {"gps": None, "pothole_images": []}
    except Exception as e:
        print("Processing error:", e)
        return jsonify({"error": str(e)}), 500

    timestamp = int(time.time())
    response = {"video_url": f"/output/{output_filename}?t={timestamp}"}

    # prefer exiftool GPS if present; otherwise use metadata gps (if any)
    gps_val = exiftool_gps
    if metadata.get("gps"):
        mg = metadata["gps"]
        if isinstance(mg, dict) and "latitude" in mg and "longitude" in mg:
            if gps_val is None:
                gps_val = {"latitude": float(mg["latitude"]), "longitude": float(mg["longitude"])}
        elif isinstance(mg, (list, tuple)) and len(mg) >= 2:
            if gps_val is None:
                gps_val = {"latitude": float(mg[0]), "longitude": float(mg[1])}

    if gps_val:
        response["location"] = gps_val

    # include pothole images (if any) as URLs
    if metadata.get("pothole_images"):
        imgs = []
        for imgname in metadata["pothole_images"]:
            imgs.append(f"/output/{imgname}?t={timestamp}")
        response["pothole_images"] = imgs

    # DEBUG: echo what server actually received in form (temporary)
    try:
        debug_received = {k: request.form.get(k) for k in request.form.keys()}
    except Exception:
        debug_received = {}
    response["_debug_received"] = debug_received

    return jsonify(response)


@app.route('/output/<path:filename>')
def get_output_file(filename):
    # Serve any file from OUTPUT_FOLDER (videos, thumbnails, sidecar json).
    # In production, validate filename to avoid path traversal.
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
