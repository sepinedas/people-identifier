#!/usr/bin/env python3
"""
face_id.py - A simple CLI for enrolling and recognizing people's faces.

Uses the face_recognition library (built on dlib's deep learning models)
for face detection and encoding, providing higher accuracy than classical
methods like Haar cascades. Everything runs locally/offline - no cloud
APIs, no internet connection required.

Commands:
    enroll     Add face images of a person to the dataset
    train      Train (or encode) the recognition model on enrolled faces
    recognize  Identify faces in an image or via webcam
    list       List everyone currently enrolled
    remove     Remove a person from the dataset

Typical workflow:
    python3 face_id.py enroll "Ada Lovelace" --webcam
    python3 face_id.py enroll "Alan Turing" --images ./photos_of_alan/*.jpg
    python3 face_id.py train
    python3 face_id.py recognize --webcam
    python3 face_id.py recognize --image group_photo.jpg --output result.jpg
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import cv2
import face_recognition
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths / storage layout
#
#   data/
#     dataset/<person_name>/<uuid>.jpg   -- original face samples
#     encodings.json                     -- {"name": [[encoding_values], ...], ...}
#     labels.json                        -- metadata (optional)
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("FACE_ID_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
ENCODINGS_PATH = os.path.join(DATA_DIR, "encodings.json")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

# Confidence threshold: LOWER = stricter matching. Range: 0.0 (strictest) to 1.0 (most permissive)
# Default 0.6 is good for typical use; try 0.5 for stricter, 0.65 for more lenient.
DEFAULT_THRESHOLD = 0.6

# Model complexity for encoding: "small" (fast, less accurate) or "large" (slower, more accurate)
ENCODING_MODEL = "large"


def _ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)


def _detect_faces(image):
    """
    Detect face locations in an image using face_recognition library.
    Returns list of (top, right, bottom, left) tuples.
    """
    return face_recognition.face_locations(image, model="hog")


def _encode_face(image):
    """
    Generate a 128-dimensional encoding for a face.
    Returns array or None if no face detected.
    """
    try:
        encodings = face_recognition.face_encodings(image, model=ENCODING_MODEL)
        return encodings[0] if encodings else None
    except Exception as e:
        print(f"  warning: encoding error: {e}")
        return None


def _load_image(path):
    """Load an image file and return as RGB array."""
    try:
        img = face_recognition.load_image_file(path)
        return img
    except Exception as e:
        print(f"  warning: failed to load {path}: {e}")
        return None


def _safe_name_to_dir(name):
    return name.strip().replace(os.sep, "_")


def _person_dir(name):
    return os.path.join(DATASET_DIR, _safe_name_to_dir(name))


def _load_encodings():
    """Load all stored encodings. Returns dict: {name: [list of encodings]}"""
    if not os.path.exists(ENCODINGS_PATH):
        return {}
    try:
        with open(ENCODINGS_PATH) as f:
            data = json.load(f)
            # Convert lists back to numpy arrays
            return {name: [np.array(enc) for enc in encs] for name, encs in data.items()}
    except Exception as e:
        print(f"warning: failed to load encodings: {e}")
        return {}


def _save_encodings(encodings_dict):
    """Save encodings to disk. Converts numpy arrays to lists for JSON."""
    try:
        serializable = {name: [enc.tolist() for enc in encs] for name, encs in encodings_dict.items()}
        with open(ENCODINGS_PATH, "w") as f:
            json.dump(serializable, f, indent=2)
    except Exception as e:
        print(f"error: failed to save encodings: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------

def cmd_enroll(args):
    _ensure_dirs()
    name = args.name.strip()
    if not name:
        sys.exit("error: name cannot be empty")

    out_dir = _person_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0

    def save_face(image_array, filename):
        nonlocal saved
        # Convert RGB back to BGR for cv2.imwrite
        bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(out_dir, filename)
        cv2.imwrite(out_path, bgr)
        saved += 1

    if args.webcam:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            sys.exit("error: could not open webcam (try a different --camera-index)")
        print(f"Enrolling '{name}'. Look at the camera. Press SPACE to capture, "
              f"'q' to finish early. Target: {args.count} samples.")
        try:
            while saved < args.count:
                ok, frame = cap.read()
                if not ok:
                    print("warning: failed to read frame from webcam")
                    break
                # Convert BGR to RGB for face_recognition
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locs = _detect_faces(rgb)
                
                display = frame.copy()
                for (top, right, bottom, left) in face_locs:
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
                
                cv2.putText(display, f"{name}: {saved}/{args.count} (SPACE=capture, q=quit)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("face_id enroll", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and len(face_locs) > 0:
                    # Use the largest detected face
                    face_locs_with_area = [(loc, (loc[2]-loc[0]) * (loc[1]-loc[3])) for loc in face_locs]
                    largest_loc = max(face_locs_with_area, key=lambda x: x[1])[0]
                    top, right, bottom, left = largest_loc
                    face_image = rgb[top:bottom, left:right]
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    save_face(face_image, f"{ts}.jpg")
                    print(f"  captured {saved}/{args.count}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    elif args.images:
        paths = []
        for pattern in args.images:
            matched = glob.glob(pattern)
            paths.extend(matched if matched else [pattern])
        if not paths:
            sys.exit("error: no image files matched --images pattern(s)")
        
        for path in paths:
            image = _load_image(path)
            if image is None:
                print(f"  skip (unreadable): {path}")
                continue
            face_locs = _detect_faces(image)
            if len(face_locs) == 0:
                print(f"  skip (no face found): {path}")
                continue
            
            # Use largest face
            face_locs_with_area = [(loc, (loc[2]-loc[0]) * (loc[1]-loc[3])) for loc in face_locs]
            largest_loc = max(face_locs_with_area, key=lambda x: x[1])[0]
            top, right, bottom, left = largest_loc
            face_image = image[top:bottom, left:right]
            
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            save_face(face_image, f"{ts}.jpg")
            print(f"  captured from {path}")
    else:
        sys.exit("error: specify either --webcam or --images <files...>")

    if saved == 0:
        print("No face samples were saved. Nothing enrolled.")
        return

    print(f"\nSaved {saved} face sample(s) for '{name}' to {out_dir}")
    print("Run 'python3 face_id.py train' to update the recognition model.")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args):
    _ensure_dirs()
    people = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not people:
        sys.exit("error: no enrolled people found. Run 'enroll' first.")

    encodings_dict = {}
    total_samples = 0

    for person in people:
        person_dir = os.path.join(DATASET_DIR, person)
        samples = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not samples:
            print(f"warning: '{person}' has no face samples, skipping")
            continue

        encodings_dict[person] = []
        for fname in samples:
            image = _load_image(os.path.join(person_dir, fname))
            if image is None:
                continue
            
            encoding = _encode_face(image)
            if encoding is not None:
                encodings_dict[person].append(encoding)
                total_samples += 1
            else:
                print(f"  warning: could not encode {person}/{fname}")

        if not encodings_dict[person]:
            del encodings_dict[person]
            print(f"warning: no valid encodings for '{person}'")

    if not encodings_dict:
        sys.exit("error: no valid face encodings to train on")

    _save_encodings(encodings_dict)

    num_people = len(encodings_dict)
    print(f"\nTrained encodings for {total_samples} samples across {num_people} people.")
    print(f"Encodings saved to {ENCODINGS_PATH}")


# ---------------------------------------------------------------------------
# recognize
# ---------------------------------------------------------------------------

def _recognize_faces(image_rgb, encodings_dict, threshold):
    """
    Detect and recognize faces in an image.
    Returns list of (name, confidence, (top, right, bottom, left)) tuples.
    """
    face_locs = _detect_faces(image_rgb)
    face_encodings = face_recognition.face_encodings(image_rgb)
    
    results = []
    for face_encoding, face_loc in zip(face_encodings, face_locs):
        best_match_name = "Unknown"
        best_distance = float('inf')

        for person_name, person_encodings in encodings_dict.items():
            distances = face_recognition.face_distance(person_encodings, face_encoding)
            min_distance = np.min(distances)

            if min_distance < best_distance:
                best_distance = min_distance
                best_match_name = person_name if min_distance <= threshold else "Unknown"

        confidence = 1.0 - best_distance if best_distance != float('inf') else 0.0
        results.append((best_match_name, confidence, face_loc))

    return results


def _annotate_frame(frame, results):
    """Draw boxes and labels on frame."""
    for name, confidence, (top, right, bottom, left) in results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def cmd_recognize(args):
    encodings_dict = _load_encodings()
    if not encodings_dict:
        sys.exit("error: no trained encodings found. Run 'train' first.")

    threshold = args.threshold

    if args.webcam:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            sys.exit("error: could not open webcam (try a different --camera-index)")
        print("Press 'q' to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("warning: failed to read frame from webcam")
                    break
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = _recognize_faces(rgb, encodings_dict, threshold)
                _annotate_frame(frame, results)
                
                cv2.imshow("face_id recognize", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"error: could not read image '{args.image}'")
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _recognize_faces(rgb, encodings_dict, threshold)
        _annotate_frame(frame, results)

        if not results:
            print("No faces detected.")
        else:
            for name, confidence, (top, right, bottom, left) in results:
                box = (left, top, right, bottom)
                print(f"  {name}  (confidence={confidence:.3f})  box={box}")

        if args.output:
            cv2.imwrite(args.output, frame)
            print(f"Annotated image saved to {args.output}")
    else:
        sys.exit("error: specify either --webcam or --image <file>")


# ---------------------------------------------------------------------------
# list / remove
# ---------------------------------------------------------------------------

def cmd_list(args):
    _ensure_dirs()
    people = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    if not people:
        print("No one is enrolled yet.")
        return

    print("Enrolled people:")
    for person in people:
        count = len([
            f for f in os.listdir(os.path.join(DATASET_DIR, person))
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        print(f"  - {person}  ({count} sample(s))")

    trained = os.path.exists(ENCODINGS_PATH)
    print(f"\nEncodings trained: {'yes' if trained else 'no (run: python3 face_id.py train)'}")


def cmd_remove(args):
    name = args.name.strip()
    person_dir = _person_dir(name)
    if not os.path.isdir(person_dir):
        sys.exit(f"error: no enrolled person named '{name}'")
    
    import shutil
    shutil.rmtree(person_dir)
    print(f"Removed '{name}'. Run 'train' again to update the model.")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="face_id.py",
        description="Enroll and recognize people's faces from images or a webcam (fully offline, deep learning-based).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Add face samples for a person")
    p_enroll.add_argument("name", help="Person's name")
    p_enroll.add_argument("--webcam", action="store_true", help="Capture samples from webcam")
    p_enroll.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    p_enroll.add_argument("--count", type=int, default=20, help="Number of webcam samples to capture (default: 20)")
    p_enroll.add_argument("--images", nargs="+", help="Image file(s) or glob pattern(s) to enroll from")
    p_enroll.set_defaults(func=cmd_enroll)

    p_train = sub.add_parser("train", help="Encode enrolled faces into the recognition model")
    p_train.set_defaults(func=cmd_train)

    p_rec = sub.add_parser("recognize", help="Recognize faces in an image or webcam feed")
    p_rec.add_argument("--webcam", action="store_true", help="Recognize from live webcam feed")
    p_rec.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    p_rec.add_argument("--image", help="Path to an image file to analyze")
    p_rec.add_argument("--output", help="Path to save annotated output image (with --image)")
    p_rec.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Max distance to count as a match; lower=stricter, 0.0-1.0 (default: {DEFAULT_THRESHOLD})")
    p_rec.set_defaults(func=cmd_recognize)

    p_list = sub.add_parser("list", help="List enrolled people")
    p_list.set_defaults(func=cmd_list)

    p_remove = sub.add_parser("remove", help="Remove a person from the dataset")
    p_remove.add_argument("name", help="Person's name")
    p_remove.set_defaults(func=cmd_remove)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
