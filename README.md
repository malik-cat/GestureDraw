# ✋ GestureDraw — Hand-Gesture Air Canvas

**Draw, erase, and create geometric designs in real time — using nothing but your hand and a webcam.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/MediaPipe-0.10%2B-009688?logo=google&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-blue">
</p>

**Author:** [Mohammad Liaquat Ali](https://github.com/malik-cat)

GestureDraw transforms hand gestures captured by a webcam into a virtual
whiteboard. Point, draw, erase, and snap together shapes such as lines,
rectangles, circles, triangles and stars — all live, on a mirrored video feed.

---

## ✨ Features

- **Full-screen hand tracking** — gestures work anywhere in the frame, not just
  in one spot.
- **Two gesture modes:**
  - `SELECTION` — index + middle fingers raised → move the cursor / click the palette.
  - `DRAWING` — index finger raised only → free-hand draw or drag a shape.
- **Interactive top palette** (two rows):
  - Row 1: `BLUE` `GREEN` `RED` `YELLOW` `ERASER` `CLEAR`
  - Row 2: `LINE` `RECT` `CIRCLE` `TRIANGLE` `STAR` `DRAW`
- **Shape engine** — pick a shape, anchor it with a fingertip, drag to size
  (live yellow preview), release to commit.
- **Persistent dual-layer canvas** — drawings are stored on a separate matrix
  and merged over the video each frame, so strokes never flicker.
- **Clean exit** — `q` releases the camera and closes every window.

---

## 📦 Requirements

| Dependency  | Version |
|-------------|---------|
| Python      | 3.9+    |
| `opencv-python` | ≥ 4.8.0 |
| `mediapipe` | ≥ 0.10.0 |
| `numpy`     | ≥ 1.24.0 |
| Webcam      | any      |

> **Note:** MediaPipe 1.0.0 uses the new `tasks` API. This project pins
> `mediapipe>=0.10.0` and uses the `HandLandmarker` Tasks API, which works with
> the bundled `hand_landmarker.task` model.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/malik-cat/GestureDraw.git
cd GestureDraw
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run it

```bash
python air_canvas.py
```

A window titled **Air Canvas** opens showing your mirrored webcam feed with the
palette header. Raise your hand and start drawing. Press `q` to quit.

> If the model file `hand_landmarker.task` is missing, download it from
> [Google's MediaPipe models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
> and place it next to `air_canvas.py`.

---

## 🖐️ How to Use

| Gesture                    | Action                                           |
|----------------------------|--------------------------------------------------|
| Index + middle fingers up  | Move the cursor / click a palette tool           |
| Index finger up only       | Free-hand draw (or drag a selected shape)        |
| Any other hand shape       | Hover — commits the current shape / holds         |
| Hand out of frame          | Commits the partially drawn shape                |

### Drawing shapes

1. Raise index + middle and point at a shape in **Row 2** (`LINE`, `RECT`,
   `CIRCLE`, `TRIANGLE`, `STAR`).
2. Switch to the draw gesture (index only). The first fingertip point anchors
   the shape.
3. Drag your finger to size it — a yellow outline previews it live.
4. Change gesture (or remove your hand) to commit the shape to the canvas.

### Tools

- **Row 1** colours change the pen; `ERASER` paints with a wide transparent
  stroke; `CLEAR` wipes the board.
- `DRAW` in Row 2 returns to free-hand mode.

---

## 🧱 Project Structure

```
GestureDraw/
├── air_canvas.py          # Complete, runnable application (all 4 phases)
├── frame_capture.py       # Phase 1 - mirrored webcam capture pipeline
├── hand_tracker.py        # Phase 2 - MediaPipe tracking + gesture engine
├── canvas.py              # Phase 3 - dual-layer canvas + palette UI
├── main.py                # Phase 4 - modular orchestrator
├── hand_landmarker.task   # MediaPipe hand landmark model
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧠 How It Works

1. **Capture** — `cv2.VideoCapture(0)` reads frames which are mirrored with
   `cv2.flip(frame, 1)` so motion feels natural.
2. **Track** — MediaPipe `HandLandmarker` locates the 21 hand landmarks; finger
   state is derived by comparing each fingertip against its PIP joint.
3. **Classify** — the finger pattern is mapped to `SELECTION`, `DRAWING`, or
   `NONE`.
4. **Render** — strokes and shapes are drawn on a persistent canvas matrix and
   merged over the video with threshold masking (`cv2.bitwise_and/or`), keeping
   the drawing fully opaque.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE). © 2026 **Mohammad Liaquat Ali**.

---

## 🙏 Acknowledgements

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — hand landmark model
- [OpenCV](https://opencv.org) — computer vision toolkit
