# face_id.py — offline face recognition CLI

A command-line tool to enroll people's faces and later recognize them in
photos or a live webcam feed. Runs entirely on your machine — no cloud
service, no API keys, no internet connection needed at runtime.

Under the hood it uses OpenCV's Haar cascade for face **detection** and the
LBPH (Local Binary Patterns Histograms) algorithm for face **recognition**.
This is lighter-weight than deep-learning face recognizers, works well for
small personal datasets (a handful of people, a household, a small team),
and needs nothing beyond `opencv-contrib-python`.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
If you don't have it yet: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(or see the uv docs for other platforms).

```bash
uv sync
```

That creates a local `.venv` with the right dependencies. You don't need to
activate it — just prefix commands with `uv run` as shown below.

Note: the dependency is `opencv-contrib-python` specifically (not just
`opencv-python`) — the `cv2.face` module lives in the contrib package.
This is already set correctly in `pyproject.toml`.

## Usage

Run every command through `uv run`, which transparently uses the project's
`.venv`:

### 1. Enroll people

From a webcam (press SPACE to capture a sample, `q` to stop early):

```bash
uv run face_id.py enroll "Ada Lovelace" --webcam --count 20
```

From existing photos (one clear face per photo works best; you can pass a
glob pattern or multiple files):

```bash
uv run face_id.py enroll "Alan Turing" --images photos/alan_*.jpg
```

Repeat for each person. Enroll 10–30 samples per person for good accuracy —
more angles/lighting = better recognition later.

### 2. Train the model

```bash
uv run face_id.py train
```

Re-run this any time you enroll or remove someone.

### 3. Recognize faces

On a single image, saving an annotated copy:

```bash
uv run face_id.py recognize --image group_photo.jpg --output result.jpg
```

Live from webcam:

```bash
uv run face_id.py recognize --webcam
```

Each detected face gets a box and a name (or "Unknown" if no confident
match). Tune strictness with `--threshold` (lower = stricter; default 70).

### Manage enrolled people

```bash
uv run face_id.py list
uv run face_id.py remove "Alan Turing"
```

### Optional: install as a `face-id` command

`pyproject.toml` also declares a `face-id` entry point, so after `uv sync`
you can use `uv run face-id ...` instead of `uv run face_id.py ...`, or
install it globally as a tool:

```bash
uv tool install .
face-id list
```

## How it works

- **Detection**: OpenCV's Haar cascade (`haarcascade_frontalface_default.xml`,
  bundled with OpenCV) finds face bounding boxes in each frame/image.
- **Storage**: each enrolled face sample is cropped, converted to grayscale,
  resized to 200×200, and saved under `data/dataset/<name>/`.
- **Training**: `cv2.face.LBPHFaceRecognizer` is trained on all saved samples,
  producing `data/model.yml` plus `data/labels.json` (numeric label → name).
- **Recognition**: each detected face is compared against the trained model.
  LBPH returns a *distance* (lower = more similar); anything above
  `--threshold` is reported as "Unknown" rather than a wrong guess.

## Limitations & tips

- Works best with clear, front-facing, well-lit faces — same as most
  classical (non-deep-learning) face recognition.
- Accuracy is good for small groups (a few to a few dozen people) but won't
  match commercial-grade systems on large populations or hard angles/lighting.
  If you need higher accuracy, swap in a deep-learning encoder (e.g. the
  `face_recognition` / dlib library, or an ONNX face-embedding model) — the
  CLI's enroll/train/recognize structure would stay the same.
- All data stays local in `./data/`. Delete that folder to wipe everything.
- Be mindful of consent and privacy when enrolling and recognizing people.
