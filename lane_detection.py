# lane_detection.py
import cv2
import numpy as np
import subprocess
import os
import json
import re
import logging




def parse_iso6709(s):
    """
    Best-effort parsing of ISO6709-like strings and other common latitude,longitude formats.
    Returns (lat, lon) as floats or None.
    """
    if not s:
        return None
    s = s.strip().rstrip('/')

    # common comma/space separated floats: "12.345678, -98.765432"
    if ',' in s:
        parts = [p.strip() for p in s.split(',')]
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except Exception:
                pass

    # space-separated numbers (two floats)
    m = re.search(r'([+\-]?\d+\.\d+)[ ,;]+([+\-]?\d+\.\d+)', s)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            pass

    # ISO6709 compact form: +DD.DDDD+DDD.DDDD or -DD.DDDD-DDD.DDDD (no comma)
    m2 = re.search(r'([+\-]\d+(?:\.\d+)?)([+\-]\d+(?:\.\d+)?)', s)
    if m2:
        try:
            lat = float(m2.group(1))
            lon = float(m2.group(2))
            return lat, lon
        except Exception:
            pass

    # fallback: find any two floats in the string
    floats = re.findall(r'([+\-]?\d+\.\d+)', s)
    if len(floats) >= 2:
        try:
            return float(floats[0]), float(floats[1])
        except Exception:
            pass

    return None

def get_gps_from_video(video_path):
    """
    Extract GPS-like tags from video via ffprobe (best-effort).
    Returns (lat, lon) or None.
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        logging.warning("ffprobe failed: %s", e)
        return None
    except Exception as e:
        logging.warning("ffprobe error: %s", e)
        return None

    tags = {}
    fmt = info.get("format", {})
    if isinstance(fmt.get("tags"), dict):
        tags.update(fmt["tags"])

    for st in info.get("streams", []):
        if isinstance(st.get("tags"), dict):
            tags.update(st["tags"])

    # normalize to lowercase keys for easier lookup
    tags_lower = {k.lower(): v for k, v in tags.items()}

    # direct numeric tags
    for lat_key in ("gpslatitude", "gps_latitude", "latitude"):
        if lat_key in tags_lower:
            try:
                lat = float(tags_lower[lat_key])
                # find lon
                for lon_key in ("gpslongitude", "gps_longitude", "longitude"):
                    if lon_key in tags_lower:
                        lon = float(tags_lower[lon_key])
                        return lat, lon
            except Exception:
                pass

    # check common location tags (try both original-case and lower-case)
    for key in ("location", "com.apple.quicktime.location.iso6709", "com.apple.quicktime.location", "gps"):
        v = tags.get(key) or tags.get(key.lower())
        if v:
            parsed = parse_iso6709(v)
            if parsed:
                return parsed

    # scan all tag values for embedded coords
    for v in tags.values():
        if isinstance(v, str):
            parsed = parse_iso6709(v)
            if parsed:
                return parsed

    return None


def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (int(0.05 * width), height),
        (int(0.45 * width), int(0.6 * height)),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.95 * width), height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    return cv2.bitwise_and(image, image, mask=combined_mask)

def detect_lanes(frame):
    filtered = color_filter(frame)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    masked = region_of_interest(edges)
    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=150)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
    return frame

def convert_to_h264(input_path, output_path):
    temp_output = output_path.replace('.mp4', '_h264.mp4')
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-movflags', '+faststart',
        temp_output
    ]
    subprocess.run(command, check=True)
    os.replace(temp_output, output_path)

def process_video(video_path, output_path):
    """
    Process video for lane detection and overlay GPS if metadata is present.
    Also writes a simple JSON sidecar next to the output with GPS when available.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    # Try to get GPS from the video metadata via ffprobe (best-effort)
    gps = get_gps_from_video(video_path)
    if gps:
        logging.info("[lane] Extracted GPS from video: %s", gps)
    else:
        logging.info("[lane] No GPS metadata found in video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        fps = int(round(fps))
        if fps <= 0:
            fps = 30
    except Exception:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = detect_lanes(frame)

            # Overlay GPS text if available
            if gps:
                lat, lon = gps
                text = f"GPS: {lat:.6f}, {lon:.6f}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                pad = 8
                x0, y0 = 30, height - 30 - th
                cv2.rectangle(processed, (x0 - pad//2, y0 - pad//2),
                              (x0 + tw + pad//2, y0 + th + pad//2), (0, 0, 0), -1)
                cv2.putText(processed, text, (x0, y0 + th - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            out.write(processed)
    finally:
        cap.release()
        out.release()

    # Convert to H.264 for compatibility
    convert_to_h264(output_path, output_path)

    # Write sidecar metadata with GPS when available
    try:
        meta = {
            "source_video": os.path.basename(video_path),
            "output_video": os.path.basename(output_path),
            "gps": {
                "latitude": float(gps[0]) if gps else None,
                "longitude": float(gps[1]) if gps else None,
            },
        }
        json_path = os.path.splitext(output_path)[0] + "_metadata.json"
        with open(json_path, "w") as jf:
            json.dump(meta, jf, indent=2)
        logging.info("[lane] Saved metadata sidecar to %s", json_path)
    except Exception as e:
        logging.warning("[lane] Failed to save sidecar metadata: %s", e)