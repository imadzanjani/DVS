# app.py
import os
import time
import subprocess
import json
import smtplib
import ssl
import mimetypes
from email.message import EmailMessage
from urllib.parse import urlparse, parse_qs
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

# Load .env if present (code-only config without shell env)
def _load_env_file(env_path: str = '.env'):
    try:
        path = os.path.join(os.path.dirname(__file__), env_path)
        if not os.path.exists(path):
            return
        with open(path, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '=' not in s:
                    continue
                k, v = s.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        pass

_load_env_file('.env')

# Email (SMTP) configuration via environment variables for security
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "")

# Optional: override from config_local.json (not committed; place next to app.py)
def _load_local_smtp_overrides():
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), 'config_local.json')
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {
                    'SMTP_HOST': data.get('SMTP_HOST'),
                    'SMTP_PORT': data.get('SMTP_PORT'),
                    'SMTP_USER': data.get('SMTP_USER'),
                    'SMTP_PASS': data.get('SMTP_PASS'),
                    'SMTP_FROM': data.get('SMTP_FROM'),
                }
    except Exception:
        pass
    return {}

_over = _load_local_smtp_overrides()
if _over:
    SMTP_HOST = _over.get('SMTP_HOST') or SMTP_HOST
    try:
        if _over.get('SMTP_PORT') is not None:
            SMTP_PORT = int(_over.get('SMTP_PORT'))
    except Exception:
        pass
    SMTP_USER = _over.get('SMTP_USER') or SMTP_USER
    SMTP_PASS = _over.get('SMTP_PASS') or SMTP_PASS
    SMTP_FROM = _over.get('SMTP_FROM') or SMTP_FROM


def _safe_output_path_from_url(url: str):
    try:
        # strip query string and get basename
        path = urlparse(url).path
        name = os.path.basename(path)
        # ensure served only from OUTPUT_FOLDER
        abs_path = os.path.join(OUTPUT_FOLDER, name)
        abs_root = os.path.abspath(OUTPUT_FOLDER)
        if os.path.commonpath([os.path.abspath(abs_path), abs_root]) != abs_root:
            return None
        if not os.path.isfile(abs_path):
            return None
        return abs_path
    except Exception:
        return None


def _send_email_with_attachments(to_email: str, subject: str, body: str, attachment_paths):
    if not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP credentials not configured. Set SMTP_USER and SMTP_PASS env vars.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM or SMTP_USER
    msg["To"] = to_email
    msg.set_content(body)

    # Attach files
    for p in attachment_paths or []:
        try:
            ctype, encoding = mimetypes.guess_type(p)
            if ctype is None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            with open(p, "rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(p))
        except Exception:
            continue

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.ehlo()
        if SMTP_PORT == 587:
            server.starttls(context=context)
            server.ehlo()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)


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


@app.route('/send_email', methods=['POST'])
def send_email_route():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    to_email = (data or {}).get("to")
    image_urls = (data or {}).get("images") or []
    location = (data or {}).get("location") or None

    if not to_email:
        return jsonify({"error": "Missing 'to' email address"}), 400

    # Map URLs to output file paths safely
    attachments = []
    for url in image_urls:
        p = _safe_output_path_from_url(url)
        if p:
            attachments.append(p)
    # Trim to max 5 attachments
    attachments = attachments[:5]

    # Build body with location
    body_lines = [
        "Pothole detection report",
    ]
    if isinstance(location, dict) and location.get("latitude") is not None and location.get("longitude") is not None:
        lat = float(location.get("latitude"))
        lon = float(location.get("longitude"))
        body_lines.append(f"Location: {lat:.6f}, {lon:.6f}")
        body_lines.append(f"OpenStreetMap: https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=16/{lat:.6f}/{lon:.6f}")
        body_lines.append(f"Google Maps: https://maps.google.com/?q={lat:.6f},{lon:.6f}")
    else:
        body_lines.append("Location: unavailable")

    body_lines.append("")
    if attachments:
        body_lines.append(f"Attached pothole snapshots: {len(attachments)}")
    else:
        body_lines.append("No snapshots available to attach.")

    try:
        _send_email_with_attachments(
            to_email=to_email,
            subject="Pothole Detection Results",
            body="\n".join(body_lines),
            attachment_paths=attachments,
        )
        return jsonify({"status": "sent", "to": to_email, "attached": [os.path.basename(a) for a in attachments]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/output/<path:filename>')
def get_output_file(filename):
    # Serve any file from OUTPUT_FOLDER (videos, thumbnails, sidecar json).
    # In production, validate filename to avoid path traversal.
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=False)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
