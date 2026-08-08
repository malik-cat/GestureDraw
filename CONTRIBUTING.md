# Contributing to GestureDraw

Thanks for helping improve GestureDraw! This project turns hand gestures into
digital art in real time using MediaPipe, OpenCV, and NumPy.

## Development setup

Requirements: Python 3.11+ (3.14 works), a webcam, and the MediaPipe Tasks
model file `hand_landmarker.task` in the project root
(see [README → Model setup](README.md)).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (bash: source .venv/bin/activate)
pip install -r requirements.txt
pip install pytest ruff       # dev/test tooling
```

## Project layout

- `config.py` — every threshold, dimension, colour, tool id, and shortcut.
- `frame_capture.py` — webcam I/O (mirrored frames).
- `hand_tracker.py` — MediaPipe Tasks HandLandmarker + gesture debouncer.
- `canvas.py` — drawing layer, stroke logic, undo/redo, palette/sidebar.
- `main.py` — the application orchestration loop (entry point).

## Running checks

```bash
ruff check .             # lint
python -m pytest tests/  # unit tests (no camera needed)
python -m py_compile config.py hand_tracker.py canvas.py frame_capture.py main.py
```

The tests exercise only pure logic (synthetic landmarks, hit-testing,
undo/redo), so they never need a camera and run in CI.

## Code conventions

- Keep magic numbers out of source files: put thresholds in `config.py`.
- Add a short docstring to any module/function you touch, mirroring the
  existing style (phase + responsibilities).
- Use `Gesture` (an `IntEnum`), never raw integers, for gesture values.
- Prefer small, focused commits. Each phase changes one concern.
- If code changes behaviour, add or update a unit test in `tests/`.

## Requesting help

Open an issue on the repository. For accessibility gaps (e.g. keyboard-only
mode for users who can't perform the hand poses), file one issue per
capability request.