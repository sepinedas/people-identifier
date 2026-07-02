#!/usr/bin/env python3
"""
face_id.py - A simple CLI for enrolling and recognizing people's faces.

Uses OpenCV's Haar cascade for face detection and its LBPH algorithm
for recognition. Everything runs locally/offline - no cloud APIs, no
internet connection required.

Commands:
    enroll     Add face images of a person to the dataset
    train      Train (or retrain) the recognition model on enrolled faces
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
import numpy as np

# ---------------------------------------------------------------------------
# Paths / storage layout
#
#   data/
#     dataset/<person_name>/<uuid>.png   -- cropped grayscale face samples
#     model.yml                          -- trained LBPH model
#     labels.json                        -- {"0": "Ada Lovelace", "1": ...}
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("FACE_ID_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODEL_PATH = os.path.join(DATA_DIR, "model.yml")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

FACE_SIZE = (200, 200)  # all stored face crops are normalized to this size
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Confidence threshold for LBPH: LOWER distance = more confident match.
# Anything above this is reported as "Unknown". Tune with --threshold.
DEFAULT_THRESHOLD = 70.0


def _ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)


def _get_detector():
    if not os.path.exists(CASCADE_PATH):
        sys.exit(f"error: Haar cascade file not found at {CASCADE_PATH}")
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        sys.exit("error: failed to load face detector")
    return detector


def _detect_faces(gray_img, detector):
    """Return list of (x, y, w, h) boxes for detected faces."""
    return detector.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )


def _load_gray(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _safe_name_to_dir(name):
    return name.strip().replace(os.sep, "_")


def _person_dir(name):
    return os.path.join(DATASET_DIR, _safe_name_to_dir(name))


# ---------------------------------------------------------------------------
# enroll
# ---------------------------------------------------------------------------

def cmd_enroll(args):
    _ensure_dirs()
    detector = _get_detector()
    name = args.name.strip()
    if not name:
        sys.exit("error: name cannot be empty")

    out_dir = _person_dir(name)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0

    def save_face(gray_frame, box):
        nonlocal saved
        x, y, w, h = box
        face = gray_frame[y:y + h, x:x + w]
        face = cv2.resize(face, FACE_SIZE)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        out_path = os.path.join(out_dir, f"{ts}.png")
        cv2.imwrite(out_path, face)
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = _detect_faces(gray, detector)
                display = frame.copy()
                for (x, y, w, h) in boxes:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display, f"{name}: {saved}/{args.count} (SPACE=capture, q=quit)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("face_id enroll", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" ") and len(boxes) > 0:
                    # use the largest detected face
                    box = max(boxes, key=lambda b: b[2] * b[3])
                    save_face(gray, box)
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
            gray = _load_gray(path)
            if gray is None:
                print(f"  skip (unreadable): {path}")
                continue
            boxes = _detect_faces(gray, detector)
            if len(boxes) == 0:
                print(f"  skip (no face found): {path}")
                continue
            box = max(boxes, key=lambda b: b[2] * b[3])
            save_face(gray, box)
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

    faces = []
    labels = []
    label_map = {}

    for idx, person in enumerate(people):
        label_map[str(idx)] = person
        person_dir = os.path.join(DATASET_DIR, person)
        samples = [f for f in os.listdir(person_dir) if f.lower().endswith(".png")]
        if not samples:
            print(f"warning: '{person}' has no face samples, skipping")
            continue
        for fname in samples:
            gray = _load_gray(os.path.join(person_dir, fname))
            if gray is None:
                continue
            faces.append(gray)
            labels.append(idx)

    if not faces:
        sys.exit("error: no valid face samples to train on")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Trained model on {len(faces)} samples across {len(people)} people.")
    print(f"Model saved to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# recognize
# ---------------------------------------------------------------------------

def _load_model():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        sys.exit("error: no trained model found. Run 'train' first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    return recognizer, label_map


def _annotate_and_report(frame, gray, detector, recognizer, label_map, threshold):
    boxes = _detect_faces(gray, detector)
    results = []
    for (x, y, w, h) in boxes:
        face = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
        label_id, distance = recognizer.predict(face)
        if distance <= threshold:
            name = label_map.get(str(label_id), "Unknown")
        else:
            name = "Unknown"
        results.append((name, distance, (x, y, w, h)))
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        caption = f"{name} ({distance:.0f})"
        cv2.putText(frame, caption, (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return results


def cmd_recognize(args):
    detector = _get_detector()
    recognizer, label_map = _load_model()
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
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _annotate_and_report(frame, gray, detector, recognizer, label_map, threshold)
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = _annotate_and_report(frame, gray, detector, recognizer, label_map, threshold)

        if not results:
            print("No faces detected.")
        for name, distance, box in results:
            print(f"  {name}  (confidence distance={distance:.1f}, lower=better)  box={box}")

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
            if f.lower().endswith(".png")
        ])
        print(f"  - {person}  ({count} sample(s))")

    trained = os.path.exists(MODEL_PATH)
    print(f"\nModel trained: {'yes' if trained else 'no (run: python3 face_id.py train)'}")


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
        description="Enroll and recognize people's faces from images or a webcam (fully offline).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_enroll = sub.add_parser("enroll", help="Add face samples for a person")
    p_enroll.add_argument("name", help="Person's name")
    p_enroll.add_argument("--webcam", action="store_true", help="Capture samples from webcam")
    p_enroll.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    p_enroll.add_argument("--count", type=int, default=20, help="Number of webcam samples to capture (default: 20)")
    p_enroll.add_argument("--images", nargs="+", help="Image file(s) or glob pattern(s) to enroll from")
    p_enroll.set_defaults(func=cmd_enroll)

    p_train = sub.add_parser("train", help="Train the recognition model on enrolled faces")
    p_train.set_defaults(func=cmd_train)

    p_rec = sub.add_parser("recognize", help="Recognize faces in an image or webcam feed")
    p_rec.add_argument("--webcam", action="store_true", help="Recognize from live webcam feed")
    p_rec.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    p_rec.add_argument("--image", help="Path to an image file to analyze")
    p_rec.add_argument("--output", help="Path to save annotated output image (with --image)")
    p_rec.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Max LBPH distance to count as a match; lower=stricter (default: {DEFAULT_THRESHOLD})")
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
