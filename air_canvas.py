"""
GestureDraw - Air Canvas (legacy entry point)
==============================================
DEPRECATED in the Phase 7 hardening pass.

The modular application now lives in `main.py`. This file is kept purely as
a thin compatibility shim so the historical `python air_canvas.py` command
keeps working --- it simply defers to the same code path as `python main.py`.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Why deprecated (Phase 7 decision):
  * The original shapes-edition monolith mixed tracking, UI, canvas and
    rendering in one 330-line file.
  * The refactored architecture splits those concerns across
    `frame_capture.py` / `hand_tracker.py` / `canvas.py` / `main.py`, which
    is testable (a camera-free pytest suite) and maintainable.
  * Keeping the monolith in sync with the fixes (stroke continuity, full-
    canvas drawing, sidebar, undo/redo, privacy) would duplicate code and
    risks the two versions drifting apart.

Use:  python main.py   (or keep using python air_canvas.py)
"""

from main import run_app

if __name__ == "__main__":
    run_app()
