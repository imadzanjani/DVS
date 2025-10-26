# damage_detection.py
import cv2
import numpy as np
import os
import subprocess
import json
import re
import logging
import heapq

logging.basicConfig(level=logging.INFO)
ISO6709_RE = re.compile(
    r'([+\-]?\d+(?:\.\d+)?)[, ]+[+\-]?\d+(?:\.\d+)?|([+\-]\d+(?:\.\d+)?)([+\-]\d+(?:\.\d+)?)'
)

# Minimum pothole size filters
MIN_POTHOLE_AREA_PX = 1500           # absolute pixel area floor (ignores tiny noise)
MIN_POTHOLE_AREA_RATIO = 0.001       # relative to frame area (0.1%)
MIN_POTHOLE_MIN_DIM_PX = 20          # minimum width/height in pixels

# Bucketing to retain some midsized potholes along with large ones
LARGE_POTHOLE_AREA_RATIO = 0.004     # >= 0.4% of frame area considered large
MAX_TOTAL_SHOTS = 5
MAX_LARGE_SHOTS = 3
MAX_MID_SHOTS = 2

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

def detect_damage(frame):
    """
    Very simple damage (pothole-like) detector based on thresholding, contours, and heuristics.
    Draws rectangles on `frame` and returns (annotated_frame, candidates).
    Each candidate is a dict with keys: bbox=(x,y,w,h), score=area.
    Applies size filters to ignore very small potholes.
    """
    h_frame, w_frame = frame.shape[:2]
    min_area = max(MIN_POTHOLE_AREA_PX, int(MIN_POTHOLE_AREA_RATIO * w_frame * h_frame))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pothole_found = False
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_POTHOLE_MIN_DIM_PX or h < MIN_POTHOLE_MIN_DIM_PX:
            continue
        h_nonzero = h if h != 0 else 1
        aspect_ratio = w / float(h_nonzero)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if hull is not None else area
        solidity = area / float(hull_area + 1e-6)
        if 0.2 < aspect_ratio < 5 and solidity > 0.4:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            pothole_found = True
            candidates.append({"bbox": (int(x), int(y), int(w), int(h)), "score": float(area)})

    if pothole_found:
        cv2.putText(frame, "Pothole Detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return frame, candidates

def convert_to_h264(input_path, output_path):
    """
    Convert the given file to an H.264 MP4. This writes to a temporary file then replaces output_path.
    """
    if not output_path.lower().endswith('.mp4'):
        raise ValueError("output_path should end with .mp4 for convert_to_h264")

    temp_output = output_path.replace('.mp4', '_h264.mp4')
    command = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        '-movflags', '+faststart',
        temp_output
    ]
    try:
        subprocess.run(command, check=True)
        os.replace(temp_output, output_path)
    except Exception as e:
        logging.warning("ffmpeg conversion failed: %s", e)
        # If conversion failed, attempt to keep the original input as the output path (best-effort)
        try:
            if input_path != output_path:
                os.replace(input_path, output_path)
        except Exception as e2:
            logging.error("Could not replace file after failed conversion: %s", e2)
            raise

def process_video(video_path, output_path, overlay_gps=True, save_sidecar=True, override_gps=None):
    """
    Process video for pothole detection.
    - Always attempts to extract GPS from the uploaded video when overlay_gps is True.
    - If override_gps is provided (tuple lat,lon), that will be used instead of metadata extraction.
    - Writes annotated output video to output_path and a JSON sidecar next to it.
    - Additionally, it selects the best 5 pothole crops (by area) across the video and saves them alongside the output.
    Returns a dict with keys: source, output, gps, pothole_images (list of filenames).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Cannot open video file")

    gps = None
    if override_gps is not None:
        gps = tuple(override_gps)
        logging.info("Using override_gps provided by caller: %s", gps)
    elif overlay_gps:
        logging.info("Attempting to extract GPS from video metadata...")
        gps = get_gps_from_video(video_path)
        if gps:
            logging.info("Extracted GPS from video: %s", gps)
        else:
            logging.info("No GPS metadata found in video.")

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
    # Write a temporary raw output first (to avoid partial overwrite if convert later)
    temp_out_path = output_path + ".tmp.mp4"
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

    # maintain separate heaps to ensure mix of sizes
    # each heap holds tuples: (score, idx, crop_bgr)
    large_heap = []  # capacity ~ MAX_LARGE_SHOTS (min-heap)
    mid_heap = []    # capacity ~ MAX_MID_SHOTS (min-heap)
    any_heap = []    # overall top-N fallback heap (min-heap, capacity MAX_TOTAL_SHOTS)
    idx_counter = 0

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed, candidates = detect_damage(frame)

            # collect candidate crops
            for c in candidates:
                (x, y, w, h) = c["bbox"]
                score = float(c.get("score", w * h))
                pad = 10
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(width, x + w + pad)
                y1 = min(height, y + h + pad)
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                # overall fallback heap (keep best MAX_TOTAL_SHOTS by score)
                if len(any_heap) < MAX_TOTAL_SHOTS:
                    heapq.heappush(any_heap, (score, idx_counter, crop))
                else:
                    if score > any_heap[0][0]:
                        heapq.heapreplace(any_heap, (score, idx_counter, crop))

                # bucket by area ratio to preserve some midsized examples
                ratio = score / float(width * height + 1e-6)
                if ratio >= LARGE_POTHOLE_AREA_RATIO:
                    if len(large_heap) < MAX_LARGE_SHOTS:
                        heapq.heappush(large_heap, (score, idx_counter, crop))
                    else:
                        if score > large_heap[0][0]:
                            heapq.heapreplace(large_heap, (score, idx_counter, crop))
                else:
                    if len(mid_heap) < MAX_MID_SHOTS:
                        heapq.heappush(mid_heap, (score, idx_counter, crop))
                    else:
                        if score > mid_heap[0][0]:
                            heapq.heapreplace(mid_heap, (score, idx_counter, crop))

                idx_counter += 1

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
            frame_idx += 1
    finally:
        cap.release()
        out.release()

    # Convert to H.264 for compatibility; will replace the final output_path
    try:
        convert_to_h264(temp_out_path, output_path)
    except Exception:
        # If conversion fails, try to fallback by renaming temp file to output
        try:
            os.replace(temp_out_path, output_path)
        except Exception as e:
            logging.error("Failed to produce final output video: %s", e)
            raise

    # Save mixed set of pothole crops (prioritize large then include mids)
    out_dir = os.path.dirname(output_path)
    base = os.path.splitext(os.path.basename(output_path))[0]
    saved_images = []
    try:
        selected = []
        used_ids = set()

        # sort heaps descending by score
        large_sorted = sorted(large_heap, key=lambda t: t[0], reverse=True)
        mid_sorted = sorted(mid_heap, key=lambda t: t[0], reverse=True)
        any_sorted = sorted(any_heap, key=lambda t: t[0], reverse=True)

        # take up to MAX_LARGE_SHOTS from large
        for score, idx, crop in large_sorted[:MAX_LARGE_SHOTS]:
            selected.append((score, idx, crop))
            used_ids.add(idx)
        # take up to MAX_MID_SHOTS from mid
        for score, idx, crop in mid_sorted:
            if len([1 for _, i, _ in selected if i not in used_ids]) >= MAX_TOTAL_SHOTS:
                break
            if len([1 for _ in selected if _]) >= MAX_LARGE_SHOTS + MAX_MID_SHOTS:
                break
            if idx in used_ids:
                continue
            selected.append((score, idx, crop))
            used_ids.add(idx)

        # fill remaining up to MAX_TOTAL_SHOTS from any_heap as fallback
        for score, idx, crop in any_sorted:
            if len(selected) >= MAX_TOTAL_SHOTS:
                break
            if idx in used_ids:
                continue
            selected.append((score, idx, crop))
            used_ids.add(idx)

        # write images
        for i, (score, idx, crop) in enumerate(selected[:MAX_TOTAL_SHOTS]):
            img_name = f"{base}_pothole_{i+1}.jpg"
            img_path = os.path.join(out_dir, img_name)
            cv2.imwrite(img_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved_images.append(img_name)
        if saved_images:
            logging.info("Saved %d pothole crops (mixed sizes)", len(saved_images))
    except Exception as e:
        logging.warning("Failed to save pothole crop images: %s", e)

    # Save sidecar metadata (always save, even if gps is None)
    if save_sidecar:
        try:
            meta = {
                "source_video": os.path.basename(video_path),
                "output_video": os.path.basename(output_path),
                "gps": {
                    "latitude": float(gps[0]) if gps else None,
                    "longitude": float(gps[1]) if gps else None
                },
                "pothole_images": saved_images
            }
            json_path = os.path.splitext(output_path)[0] + "_metadata.json"
            with open(json_path, "w") as jf:
                json.dump(meta, jf, indent=2)
            logging.info("Saved metadata sidecar to %s", json_path)
        except Exception as e:
            logging.warning("failed to save sidecar metadata: %s", e)

    return {
        "source": video_path,
        "output": output_path,
        "gps": (gps if gps else None),
        "pothole_images": saved_images
    }

if __name__ == "__main__":
    # simple CLI example (for quick testing)
    import argparse
    parser = argparse.ArgumentParser(description="Process a video for pothole detection and overlay GPS.")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output", help="Output mp4 path")
    parser.add_argument("--no-gps-overlay", action="store_true", help="Do not try to overlay GPS from metadata")
    parser.add_argument("--no-sidecar", action="store_true", help="Do not write metadata sidecar")
    args = parser.parse_args()

    res = process_video(args.input, args.output, overlay_gps=not args.no_gps_overlay, save_sidecar=not args.no_sidecar)
    logging.info("Processing complete: %s", res)
