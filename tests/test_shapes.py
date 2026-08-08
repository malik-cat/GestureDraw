"""
Phase 6 + v2-brief tests: camera-free smoke test and shape engine.

Covers the exact failure mode that brief v2 Phase 0 describes: a cross-module
import/construction break would now fail this suite instead of only at runtime.

Run with:  python -m pytest tests/ -v
"""

import collections

import numpy as np

import config
from hand_tracker import Gesture
from shapes import ShapeEngine, ShapeTool, draw_shape

Lm = collections.namedtuple("Lm", "x y")


def _landmarks_at(nx, ny):
    """A 21-landmark hand whose index tip (id 8) sits at (nx, ny)."""
    lms = [Lm(0.5, 0.5)] * 21
    lms[8] = Lm(nx, ny)
    return lms


# ------------------------------------------------------------------ #
# Smoke tests (the Phase 0 failure mode)                              #
# ------------------------------------------------------------------ #

def test_all_modules_import():
    import canvas
    import frame_capture
    import hand_tracker
    import main
    assert callable(canvas.Canvas)
    assert callable(frame_capture.FrameSource)
    assert callable(hand_tracker.HandTracker)
    assert callable(main.AirCanvasApp)


def test_aircanvasapp_constructs_without_camera():
    """Phase 0 catastrophe check: the app must build without a camera."""
    from main import AirCanvasApp
    app = AirCanvasApp(model_path=config.MODEL_PATH)
    assert app.canvas is None          # lazily created after first frame
    assert app.tool == config.TOOL_RED
    app.tracker.release()


# ------------------------------------------------------------------ #
# Shape engine (Phase 2 of the v2 brief)                              #
# ------------------------------------------------------------------ #

def test_engine_select_begin_drag_commit():
    e = ShapeEngine()
    e.select(ShapeTool.RECT)
    assert e.tool_selected and not e.shape_active
    e.begin(50, 50)
    e.drag(120, 90)
    assert e.shape_active
    triple = e.commit()
    assert triple == (ShapeTool.RECT, (50, 50), (120, 90))
    assert e.anchor is None
    assert e.tool == ShapeTool.RECT     # still selected for next shape


def test_engine_returns_to_free_draw():
    e = ShapeEngine()
    e.select(ShapeTool.STAR)
    e.select(ShapeTool.NONE)
    assert not e.tool_selected


def test_commit_without_drag_returns_none():
    e = ShapeEngine()
    e.select(ShapeTool.LINE)
    assert e.commit() is None


def test_shape_draw_geometry_writes_pixels():
    for tool in (ShapeTool.LINE, ShapeTool.RECT, ShapeTool.CIRCLE,
                 ShapeTool.TRIANGLE, ShapeTool.STAR):
        layer = np.full((300, 400, 3), config.CANVAS_BG, np.uint8)
        draw_shape(layer, tool, (50, 50), (350, 200), 400, 300,
                   color=config.COLOR_BLUE, thickness=3)
        mask = np.any(layer != config.CANVAS_BG, axis=-1)
        assert mask.any(), f"{tool.name} wrote nothing"


def test_shape_geometry_stays_in_bounds():
    """Anchor/drag at the frame edge must clamp, not crash."""
    layer = np.full((300, 400, 3), config.CANVAS_BG, np.uint8)
    draw_shape(layer, ShapeTool.CIRCLE, (-20, 9999), (9999, -5), 400, 300)
    assert np.any(layer != config.CANVAS_BG)   # drawn inside bounds


# ------------------------------------------------------------------ #
# Phase 3: full-canvas edge coverage                                  #
# ------------------------------------------------------------------ #

def test_layer_is_full_frame():
    from canvas import Canvas
    c = Canvas(640, 480)
    assert c.layer.shape[:2] == (480, 640)
    assert c.width == 640 and c.height == 480


def test_bottom_row_and_right_column_drawable():
    """Strokes must reach the last row and last column (Phase 3)."""
    from canvas import Canvas
    c = Canvas(640, 480)
    c.stroke(638, 478)                 # within a few px of the bottom-right
    mask = np.any(c.layer != config.CANVAS_BG, axis=-1)
    assert mask[478, 638]
    assert mask[477, 638]              # 2 px from the very edge


def test_ui_hit_zones_match_config():
    from main import AirCanvasApp
    app = AirCanvasApp(model_path=config.MODEL_PATH)
    from canvas import Canvas, Sidebar
    app.canvas = Canvas(640, 480)
    app.sidebar = Sidebar()            # width from config
    # Inside sidebar -> suppressed.
    assert app.inside_ui(5, 200) is True
    # Inside header -> suppressed.
    assert app.inside_ui(400, 5) is True
    # Bottom row, right column -> drawable (outside every UI zone).
    assert app.inside_ui(639, 479) is False
    assert app.inside_ui(app.sidebar.width + 1, config.HEADER_HEIGHT + 1) \
        is False
    app.tracker.release()


def test_freehand_stroke_pushes_history_once():
    """A multi-frame free-hand stroke pushes exactly one undo snapshot."""
    from canvas import Canvas
    from main import AirCanvasApp
    app = AirCanvasApp(model_path=config.MODEL_PATH)
    app.canvas = Canvas(640, 480)
    app.palette = None
    app.sidebar = None
    app.shapes.select(ShapeTool.NONE)   # freehand mode

    # Draw several frames (simulating a connected stroke).
    for i in range(1, 20):
        app.handle_hand(Gesture.DRAW, _landmarks_at(0.2 + i * 0.02, 0.5),
                        640, 480, i)
    assert app._stroke_ongoing is True

    # Leaving DRAW (hover) commits one history snapshot.
    app.handle_hand(Gesture.NONE, _landmarks_at(0.5, 0.5), 640, 480, 20)
    assert len(app.canvas._undo_stack) == 1
    app.tracker.release()


def test_app_shape_flow_end_to_end():
    """SELECT picks a shape; DRAW anchors+drags; leaving DRAW commits."""
    from canvas import Canvas
    from main import AirCanvasApp
    app = AirCanvasApp(model_path=config.MODEL_PATH)
    app.canvas = Canvas(640, 480)
    app.palette = None
    app.sidebar = None

    # Pick the rectangle tool (as a SELECT hit would).
    app.apply_tool(config.TOOL_RECT)
    assert app.shapes.tool == ShapeTool.RECT

    # First DRAW frame: anchor.
    app.handle_hand(Gesture.DRAW, _landmarks_at(0.5, 0.5), 640, 480, 1)
    assert app.shapes.anchor == (320, 240)

    # Drag.
    app.handle_hand(Gesture.DRAW, _landmarks_at(0.75, 0.5), 640, 480, 2)
    assert app.shapes.current == (480, 240)

    # Leave DRAW (HOVER) -> commit bakes into layer + pushes history.
    app.handle_hand(Gesture.NONE, _landmarks_at(0.5, 0.5), 640, 480, 3)
    drawn = int(np.any(app.canvas.layer != config.CANVAS_BG, axis=-1).sum())
    assert drawn > 0
    assert len(app.canvas._undo_stack) == 1
    app.tracker.release()
