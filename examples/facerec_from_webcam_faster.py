# Real-time face recognition: best-frame-per-track, tiered saves, verification queue.
# OpenCV + face_recognition; loads encodings.pkl. Processes alternate frames for speed.
import face_recognition
import cv2
import os
import re
import time
import numpy as np
import pickle
from datetime import datetime

# ----- Confidence tiers (percent): confidence = (1 - distance) * 100 -----
HIGH_CONFIDENCE = 70.0
MEDIUM_LOW = 45.0
MEDIUM_HIGH = 60.0
MIN_CONF_SAVE = 30.0

MIN_MARGIN = 0.08
FRAME_SCALE = 0.25
BLUR_VAR_THRESHOLD = 50.0
# Same-face merge: min face_distance to existing track encoding (lower = stricter).
MERGE_DISTANCE_MAX = 0.45
REPR_EMA = 0.65
# Optional: min seconds between disk writes for same face_id (0 = off).
TRACK_SAVE_COOLDOWN_SEC = 0.0
COOLDOWN_CONF_DELTA = 3.0

# =========================
# LOAD TRAINING DATA
# =========================
print("Loading saved encodings...")

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

print("✓ Loaded from cache\n")

BASE_CAPTURE = "captured_faces"
KNOWN_DIR = os.path.join(BASE_CAPTURE, "known")
UNKNOWN_DIR = os.path.join(BASE_CAPTURE, "unknown")
VERIFY_DIR = os.path.join(BASE_CAPTURE, "verification")
for d in (KNOWN_DIR, UNKNOWN_DIR, VERIFY_DIR):
    os.makedirs(d, exist_ok=True)

# face_id -> { "repr": ndarray }
face_tracks = {}
next_face_id = 0

# face_id -> { "name", "confidence", "image" (BGR), "tier", "path", "last_save_ts" }
best_faces = {}

# face_id -> latest verification payload (for dashboard / UI)
verification_queue = {}

process_this_frame = True
face_locations = []
face_names = []
face_confidences = []
face_ids_display = []


def safe_filename_part(name):
    return re.sub(r"[^\w\-]+", "_", str(name)).strip("_") or "guest"


def laplacian_blur_variance(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def scale_location_to_full(loc_small, scale_inv):
    top, right, bottom, left = loc_small
    return (
        int(top * scale_inv),
        int(right * scale_inv),
        int(bottom * scale_inv),
        int(left * scale_inv),
    )


def clip_rect(top, right, bottom, left, width, height):
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width))
    top = max(0, min(top, height - 1))
    bottom = max(0, min(bottom, height))
    if right <= left or bottom <= top:
        return None
    return top, right, bottom, left


def assign_face_id(encoding):
    """Match encoding to an existing track or create a new face_id."""
    global next_face_id
    enc = np.asarray(encoding, dtype=np.float64)
    best_id = None
    best_d = 1e9
    for fid, track in face_tracks.items():
        d = float(face_recognition.face_distance([track["repr"]], enc)[0])
        if d < best_d:
            best_d = d
            best_id = fid
    if best_id is not None and best_d <= MERGE_DISTANCE_MAX:
        t = face_tracks[best_id]
        t["repr"] = REPR_EMA * t["repr"] + (1.0 - REPR_EMA) * enc
        return best_id
    fid = next_face_id
    next_face_id += 1
    face_tracks[fid] = {"repr": enc.copy()}
    return fid


def classify_tier(confidence, margin, candidate_name):
    """
    Returns (storage_tier, display_name, logical_name_for_file).
    storage_tier in {None skip, 'known', 'unknown', 'verification'}.
    """
    if confidence < MIN_CONF_SAVE:
        return None, "UNKNOWN", candidate_name

    if confidence < MEDIUM_LOW:
        return "unknown", "UNKNOWN", candidate_name

    if MEDIUM_LOW <= confidence <= MEDIUM_HIGH:
        return "verification", "VERIFICATION REQUIRED", candidate_name

    if confidence > HIGH_CONFIDENCE:
        return "known", candidate_name, candidate_name

    if margin > MIN_MARGIN:
        return "known", candidate_name, candidate_name

    return "verification", "VERIFICATION REQUIRED", candidate_name


def crop_face_bgr(frame_bgr, loc_small):
    scale_inv = 1.0 / FRAME_SCALE
    top, right, bottom, left = loc_small
    top, right, bottom, left = (
        int(top * scale_inv),
        int(right * scale_inv),
        int(bottom * scale_inv),
        int(left * scale_inv),
    )
    h, w = frame_bgr.shape[:2]
    rect = clip_rect(top, right, bottom, left, w, h)
    if rect is None:
        return None
    top, right, bottom, left = rect
    crop = frame_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    return crop


def maybe_update_best_face(
    face_id, tier, display_name, logical_name, confidence, crop_bgr
):
    if tier is None or crop_bgr is None:
        return

    now = time.monotonic()
    prev = best_faces.get(face_id)
    if prev is not None:
        if confidence <= prev["confidence"]:
            return
        if (
            TRACK_SAVE_COOLDOWN_SEC > 0
            and (now - prev.get("last_save_ts", 0)) < TRACK_SAVE_COOLDOWN_SEC
            and confidence < prev["confidence"] + COOLDOWN_CONF_DELTA
        ):
            return

    if BLUR_VAR_THRESHOLD > 0.0:
        if laplacian_blur_variance(crop_bgr) < BLUR_VAR_THRESHOLD:
            return

    date_tag = datetime.now().strftime("%Y%m%d")
    ts_full = datetime.now().strftime("%H%M%S")

    if tier == "known":
        sub = KNOWN_DIR
        fname = f"{safe_filename_part(logical_name)}_{int(round(confidence))}_{date_tag}_{ts_full}.jpg"
    elif tier == "unknown":
        sub = UNKNOWN_DIR
        fname = (
            f"unknown_f{face_id}_{int(round(confidence))}_{date_tag}_{ts_full}.jpg"
        )
    else:
        sub = VERIFY_DIR
        fname = f"verify_{safe_filename_part(logical_name)}_{int(round(confidence))}_{date_tag}_{ts_full}.jpg"

    path = os.path.join(sub, fname)

    if prev and prev.get("path") and os.path.isfile(prev["path"]):
        try:
            os.remove(prev["path"])
        except OSError:
            pass

    cv2.imwrite(path, crop_bgr)
    best_faces[face_id] = {
        "name": display_name,
        "confidence": float(confidence),
        "image": crop_bgr.copy(),
        "tier": tier,
        "path": path,
        "logical_name": logical_name,
        "last_save_ts": now,
    }

    if tier == "verification":
        verification_queue[face_id] = {
            "face_id": face_id,
            "predicted_name": logical_name,
            "confidence": float(confidence),
            "image_path": path,
        }
    elif face_id in verification_queue:
        del verification_queue[face_id]


video_capture = cv2.VideoCapture(0)

print("Starting webcam... Press 'esc' to exit")
print(f"Saves → {os.path.abspath(BASE_CAPTURE)}/{{known,unknown,verification}}")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
        rgb_small_frame = np.ascontiguousarray(
            cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        )

        raw_locs = face_recognition.face_locations(rgb_small_frame)
        faces_data = []

        for loc in raw_locs:
            try:
                enc = face_recognition.face_encodings(rgb_small_frame, [loc])
                if enc and len(enc) > 0:
                    faces_data.append((loc, enc[0]))
            except Exception as e:
                print(f"⚠️ Skipping bad face: {e}")

        face_locations = [fd[0] for fd in faces_data]
        face_names = []
        face_confidences = []
        face_ids_display = []

        if known_face_encodings:
            for loc, face_encoding in faces_data:
                face_id = assign_face_id(face_encoding)

                distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                sorted_idx = np.argsort(distances)
                best_idx = int(sorted_idx[0])
                second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else best_idx

                best_distance = float(distances[best_idx])
                second_distance = float(distances[second_idx])
                margin = second_distance - best_distance

                confidence = (1.0 - best_distance) * 100.0
                candidate_name = known_face_names[best_idx]

                tier, display_name, logical_name = classify_tier(
                    confidence, margin, candidate_name
                )
                crop = crop_face_bgr(frame, loc)
                maybe_update_best_face(
                    face_id, tier, display_name, logical_name, confidence, crop
                )

                face_names.append(display_name)
                face_confidences.append(confidence)
                face_ids_display.append(face_id)
        else:
            for loc, face_encoding in faces_data:
                face_id = assign_face_id(face_encoding)
                confidence = 35.0
                tier, display_name, logical_name = classify_tier(
                    confidence, MIN_MARGIN, "UNKNOWN"
                )
                crop = crop_face_bgr(frame, loc)
                maybe_update_best_face(
                    face_id, tier, display_name, logical_name, confidence, crop
                )
                face_names.append(display_name)
                face_confidences.append(confidence)
                face_ids_display.append(face_id)

    process_this_frame = not process_this_frame

    scale_inv = 1.0 / FRAME_SCALE
    for (top, right, bottom, left), name, conf, fid in zip(
        face_locations, face_names, face_confidences, face_ids_display
    ):
        top_f, right_f, bottom_f, left_f = scale_location_to_full(
            (top, right, bottom, left), scale_inv
        )
        if name == "UNKNOWN":
            color = (0, 0, 255)
        elif name == "VERIFICATION REQUIRED":
            color = (0, 165, 255)
        elif conf > HIGH_CONFIDENCE:
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)

        cv2.rectangle(frame, (left_f, top_f), (right_f, bottom_f), color, 2)
        cv2.rectangle(
            frame,
            (left_f, bottom_f - 35),
            (right_f, bottom_f),
            color,
            cv2.FILLED,
        )
        short = name if len(name) <= 22 else name[:19] + "..."
        label = f"id{fid} {short} {conf:.0f}%"
        cv2.putText(
            frame,
            label,
            (left_f + 6, bottom_f - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    cv2.imshow("Webcam Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

print("\n===== DETECTED PERSONS (best per face_id) =====")
if best_faces:
    for fid in sorted(best_faces.keys()):
        b = best_faces[fid]
        print(
            f"  face_id={fid}  {b['name']}  "
            f"confidence={b['confidence']:.2f}%  tier={b['tier']}  path={b['path']}"
        )
else:
    print("  (none above MIN_CONF_SAVE / blur / cooldown)")

print("\n===== VERIFICATION QUEUE (for UI / manager) =====")
if verification_queue:
    for item in verification_queue.values():
        print(
            f"  face_id={item['face_id']}  predicted={item['predicted_name']}  "
            f"confidence={item['confidence']:.2f}%  file={item['image_path']}"
        )
        print(
            "    → Manager: YES = confirm mapping | NO = UNKNOWN | NOT_SURE = review later"
        )
else:
    print("  (empty)")
