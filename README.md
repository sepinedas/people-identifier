# face_id.py — offline face recognition CLI

A command-line tool to enroll people's faces and later recognize them in
photos or a live webcam feed. Runs entirely on your machine — no cloud
service, no API keys, no internet connection needed at runtime.

Under the hood it uses the `deepface` library, which provides access to
state-of-the-art deep learning models (VGGFace2, ArcFace, Facenet, etc.) for
high-accuracy face **detection** and **embedding generation**. This provides
superior accuracy compared to classical methods and offers flexibility to
choose different underlying models for different use cases.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
If you don't have it yet: `curl -LsSf https://astral.sh/uv/install.sh | sh`
(or see the uv docs for other platforms).

```bash
uv sync
```

That creates a local `.venv` with the right dependencies. You don't need to
activate it — just prefix commands with `uv run` as shown below.

Note: The dependencies include `deepface` and `tensorflow`, which may require
additional system packages on some platforms. See
[deepface's installation docs](https://github.com/serengp/deepface#installation)
if you run into issues.

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

Repeat for each person. Enroll 5–10 samples per person for good accuracy —
more diverse angles/lighting = better recognition later. (Deep learning
models need fewer samples than classical methods.)

### 2. Train the model

```bash
uv run face_id.py train
```

This encodes all enrolled face samples into a compact representation and
saves them to `data/encodings.json`. Re-run this any time you enroll or
remove someone.

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
match). Tune strictness with `--threshold` (default 0.6; lower = stricter,
range 0.0–1.0).

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

- **Detection**: `DeepFace.extract_faces()` uses advanced detector backends
  (OpenCV, MTCNN, SSD, etc.) to find face bounding boxes in each frame/image
  with high accuracy.
- **Storage**: Each enrolled face sample is stored under `data/dataset/<name>/`
  as a JPEG image.
- **Encoding**: `DeepFace.represent()` generates high-dimensional embeddings
  using state-of-the-art models like VGGFace2, ArcFace, or Facenet. These are
  more robust to variations in angle, lighting, and expression than classical methods.
- **Training**: All embeddings are computed and stored in `data/encodings.json`
  plus `data/labels.json` (metadata).
- **Recognition**: Each detected face is encoded and compared against all
  stored encodings using Euclidean distance. Lower distance = more similar;
  anything above `--threshold` is reported as "Unknown".

## Accuracy & performance

- **State-of-the-art accuracy** from modern deep learning models, especially for:
  - Faces at different angles or in variable lighting
  - Recognizing people you've enrolled with few samples (5–10)
  - Handling complex scenarios with multiple faces
- **Trade-off**: Slower inference than classical methods (~1–3 seconds per image
  with CPU), but much better accuracy. GPU acceleration is supported if TensorFlow
  is configured with CUDA.
- Works well for small to medium groups (a few to a few hundred people).
- All data stays local in `./data/`. Delete that folder to wipe everything.

## Limitations & tips

- Accuracy is highest with clear, front-facing, well-lit faces.
- For very large populations (1000s of people) or real-time processing on
  many concurrent streams, consider optimizations like model quantization,
  ONNX export, or GPU acceleration.
- Be mindful of consent and privacy when enrolling and recognizing people.
- Tune `--threshold` to your use case:
  - `0.5` or lower: very strict, few false positives, may miss some true matches
  - `0.6` (default): balanced
  - `0.7` or higher: more lenient, more false positives
- You can change the embedding model by editing the `ENCODING_MODEL` variable
  in `face_id.py` (options: "VGGFace", "VGGFace2", "OpenFace", "DeepFace",
  "ArcFace", "Facenet", etc.). Different models have different strengths and
  trade-offs.
