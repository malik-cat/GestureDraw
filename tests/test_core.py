"""
Unit tests for GestureDraw pure-logic components (no camera required).

Run with:  python -m pytest tests/ -v
"""

import collections

import numpy as np

import config
from canvas import Canvas, Palette, Sidebar
from hand_tracker import Gesture, GestureStabilizer, HandTracker

# ------------------------------------------------------------------ #
# Synthetic landmark helper                                           #
# ------------------------------------------------------------------ #

Lm = collections.namedtuple("Lm", "x y")


def make_hand(fingers):
    """
    Build a fake 21-landmark hand.

    A finger is "raised" when its tip.y < pip.y (index/middle/ring/pinky).
    The thumb is "raised" when tip.x < pip.x (bends sideways).

    Args:
        fingers: [thumb, index, middle, ring, pinky], each 1/0.

    Returns:
        list[Lm] of 21 landmark objects (normalised .x/.y).
    """
    lms = [Lm(0.5, 0.5)] * 21
    lms[3] = Lm(0.8, 0.5)                        # thumb PIP
    lms[4] = Lm(0.4, 0.5) if fingers[0] else Lm(0.9, 0.5)
    idx_map = {8: 1, 12: 2, 16: 3, 20: 4}
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        up = fingers[idx_map[tip]]
        lms[tip] = Lm(0.5, 0.4 if up else 0.6)
        lms[pip] = Lm(0.5, 0.6 if up else 0.4)
    return lms


def drawn_pixels(canvas):
    """Number of pixels on the canvas that differ from the background."""
    bg = np.asarray(config.CANVAS_BG)
    mask = np.any(canvas.layer != bg, axis=-1)
    return int(mask.sum())


# ------------------------------------------------------------------ #
# fingers_up / classify                                              #
# ------------------------------------------------------------------ #

def test_fingers_up_neutral_hand():
    assert HandTracker.fingers_up(make_hand([0, 0, 0, 0, 0])) == \
        [0, 0, 0, 0, 0]


def test_fingers_up_all_raised():
    assert HandTracker.fingers_up(make_hand([1, 1, 1, 1, 1])) == \
        [1, 1, 1, 1, 1]


def test_fingers_up_only_index():
    assert HandTracker.fingers_up(make_hand([0, 1, 0, 0, 0])) == \
        [0, 1, 0, 0, 0]


def test_classify_draw():
    assert HandTracker.classify(HandTracker.fingers_up(
        make_hand([0, 1, 0, 0, 0]))) == Gesture.DRAW


def test_classify_select():
    assert HandTracker.classify(HandTracker.fingers_up(
        make_hand([0, 1, 1, 0, 0]))) == Gesture.SELECT


def test_classify_clear():
    assert HandTracker.classify(HandTracker.fingers_up(
        make_hand([1, 1, 1, 1, 1]))) == Gesture.CLEAR


def test_classify_fist_is_hover():
    assert HandTracker.classify(HandTracker.fingers_up(
        make_hand([0, 0, 0, 0, 0]))) == Gesture.NONE


def test_index_tip_pixels():
    lms = [Lm(0.5, 0.5)] * 21
    lms[8] = Lm(0.25, 0.75)
    assert HandTracker.index_tip_pixels(lms, 640, 200) == (160, 150)


# ------------------------------------------------------------------ #
# GestureStabilizer (Phase 1 debounce)                               #
# ------------------------------------------------------------------ #

def test_stabilizer_requires_consecutive_frames():
    s = GestureStabilizer(stable_frames=3)
    assert s.update(Gesture.DRAW) == Gesture.NONE
    assert s.update(Gesture.DRAW) == Gesture.NONE
    assert s.update(Gesture.DRAW) == Gesture.DRAW


def test_stabilizer_noise_does_not_switch():
    s = GestureStabilizer(stable_frames=2, exit_frames=5)
    s.update(Gesture.DRAW)
    s.update(Gesture.DRAW)          # DRAW becomes active
    assert s.update(Gesture.DRAW) == Gesture.DRAW
    # A noisy NONE frame must not switch away from a confirmed DRAW.
    assert s.update(Gesture.NONE) == Gesture.DRAW
    assert s.update(Gesture.NONE) == Gesture.DRAW
    assert s.update(Gesture.SELECT) == Gesture.DRAW   # even a different gesture
    assert s.update(Gesture.DRAW) == Gesture.DRAW


def test_stabilizer_releases_after_exit_frames():
    s = GestureStabilizer(stable_frames=2, exit_frames=5)
    s.update(Gesture.DRAW)
    s.update(Gesture.DRAW)
    # Active DRAW must survive a handful of NONE frames...
    for _ in range(4):
        assert s.update(Gesture.NONE) == Gesture.DRAW
    # ...but eventually a sustained NONE replaces it.
    assert s.update(Gesture.NONE) == Gesture.NONE


def test_stabilizer_reset():
    s = GestureStabilizer(stable_frames=2)
    s.update(Gesture.DRAW)
    s.update(Gesture.DRAW)
    assert s.update(Gesture.DRAW) == Gesture.DRAW
    s.reset()
    assert s.update(Gesture.DRAW) == Gesture.NONE


# ------------------------------------------------------------------ #
# Canvas.stroke (continuity + max-jump guard)                        #
# ------------------------------------------------------------------ #

def test_stroke_first_point_draws_a_dot():
    c = Canvas(320, 240)
    c.stroke(50, 50)
    assert c.last_point == (50, 50)
    assert drawn_pixels(c) > 0


def test_stroke_connects_points_with_line():
    c = Canvas(320, 240)
    c.stroke(10, 60)
    c.stroke(60, 60)
    for x in range(15, 55, 5):
        # every sampled point along the horizontal line is painted
        assert c.layer[60, x].tolist() != list(config.CANVAS_BG)


def test_max_jump_starts_new_stroke():
    c = Canvas(320, 240, max_jump=120)
    c.stroke(10, 10)
    c.stroke(280, 210)          # dx=270 > this canvas's 120px max_jump
    assert c.last_point == (280, 210)          # pointer did move
    # No long stray line: paint the new-start dot, not a connecting line.
    assert drawn_pixels(c) < 500
    mask = np.any(c.layer != config.CANVAS_BG, axis=-1)
    assert mask[210, 280]                        # post-jump point IS painted


def test_reset_pointer_begins_fresh_stroke():
    c = Canvas(320, 240)
    c.stroke(10, 10)
    c.reset_pointer()
    c.stroke(200, 200)
    # the two points are far apart and NOT joined by a connecting line:
    # the midpoint between them must stay background-coloured.
    assert c.layer[105, 105].tolist() == list(config.CANVAS_BG)


# ------------------------------------------------------------------ #
# Undo / redo                                                        #
# ------------------------------------------------------------------ #

def test_undo_restores_blank_layer():
    c = Canvas(320, 240)
    c.stroke(100, 100)
    c.push_stroke_history()
    c.stroke(200, 200)
    assert c.undo() is True
    # After undo, only the first dot remains (drawn pixel count is small).
    assert drawn_pixels(c) > 0
    assert drawn_pixels(c) < 500


def test_redo_restores_after_undo():
    c = Canvas(320, 240)
    c.stroke(10, 10)
    c.push_stroke_history()
    before = c.layer.copy()
    assert c.undo() is True
    assert c.redo() is True
    assert np.array_equal(c.layer, before)


def test_undo_empty_returns_false():
    c = Canvas(320, 240)
    assert c.undo() is False


def test_clear_accepts_bgr_tuple():
    c = Canvas(320, 240)
    c.stroke(10, 10)
    c.clear()
    assert drawn_pixels(c) == 0
    assert tuple(c.layer[0, 0]) == config.CANVAS_BG


# ------------------------------------------------------------------ #
# UI hit-testing                                                     #
# ------------------------------------------------------------------ #

def test_sidebar_hit_test():
    sb = Sidebar()
    dummy = np.zeros((480, 640, 3), np.uint8)
    sb.draw(dummy, config.TOOL_RED)
    assert sb.hit_test(5, 10) == config.TOOL_RED
    assert sb.hit_test(200, 10) is None     # outside the sidebar


def test_palette_hit_test():
    p = Palette()
    dummy = np.zeros((480, 640, 3), np.uint8)
    p.draw(dummy, config.TOOL_RED)
    assert p.hit_test(5, 5) == config.TOOL_RED
    assert p.hit_test(300, 300) is None     # below the header


# ------------------------------------------------------------------ #
# App-level undo regression (one keypress must remove a whole stroke) #
# ------------------------------------------------------------------ #

def _index_tip_at(nx, ny):
    """21 landmarks whose index tip (id 8) sits at normalised (nx, ny)."""
    lms = [Lm(0.5, 0.5)] * 21
    lms[8] = Lm(nx, ny)
    return lms


def _app_with_canvas():
    """A stripped app: full-frame canvas, no UI panels, free-hand mode."""
    from main import AirCanvasApp
    app = AirCanvasApp(model_path=config.MODEL_PATH)
    app.canvas = Canvas(640, 480)
    app.palette = None
    app.sidebar = None
    return app


def test_first_undo_removes_whole_latest_freehand_stroke():
    """Regression: 'u' once must revert the most recent stroke, not require
    two presses to see any change."""
    app = _app_with_canvas()
    # Stroke A (a connected horizontal line).
    for i in range(1, 30):
        app.handle_hand(Gesture.DRAW, _index_tip_at(0.1 + i * 0.01, 0.5),
                        640, 480, i)
    app.handle_hand(Gesture.NONE, _index_tip_at(0.5, 0.5), 640, 480, 60)
    pix_after_a = drawn_pixels(app.canvas)
    assert pix_after_a > 0

    # Stroke B at a different row.
    for i in range(1, 30):
        app.handle_hand(Gesture.DRAW, _index_tip_at(0.1 + i * 0.01, 0.7),
                        640, 480, i)
    app.handle_hand(Gesture.NONE, _index_tip_at(0.5, 0.5), 640, 480, 60)

    assert app.canvas.undo() is True
    assert drawn_pixels(app.canvas) == pix_after_a
    app.tracker.release()


def test_first_undo_removes_committed_shape():
    """Regression: one undo must take a committed shape off the canvas
    (not require two presses to see any change)."""
    app = _app_with_canvas()
    app.apply_tool(config.TOOL_RECT)
    # Anchor + drag + leave DRAW = commit a rect.
    app.handle_hand(Gesture.DRAW, _index_tip_at(0.5, 0.5), 640, 480, 1)
    app.handle_hand(Gesture.DRAW, _index_tip_at(0.75, 0.5), 640, 480, 2)
    app.handle_hand(Gesture.NONE, _index_tip_at(0.5, 0.5), 640, 480, 3)
    assert drawn_pixels(app.canvas) > 0

    assert app.canvas.undo() is True
    assert drawn_pixels(app.canvas) == 0
    app.tracker.release()


def test_camera_open_failure_releases_tracker(monkeypatch):
    """Regression: if the webcam cannot be opened, the MediaPipe handle
    must still be released (no resource leak on the error path)."""
    import main
    from main import AirCanvasApp
    app = AirCanvasApp(model_path=config.MODEL_PATH)
    released = []
    app.tracker.release = lambda: released.append(True)

    class FailingSource:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("camera in use")

    monkeypatch.setattr(main, "FrameSource", FailingSource)
    app.run()
    assert released, "tracker.release() must be called when the camera fails"
    app.tracker = None
