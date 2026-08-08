"""
Camera-free unit tests for gesture_draw_v2.py (Phases 1-3).

Covers the testable, deterministic pieces: gesture classification, pinch
ratio, cursor positioning, Canvas interpolation/resets, and Header
stable-selection (hover + pinch click).

Run:  python -m pytest tests/ -q
"""

import collections
import importlib.util
import sys
from pathlib import Path

import numpy as np

# 'v2' is a plain script; import it as a module by file path so these tests
# pass even when the repo root is not on sys.path.
_V2_PATH = Path(__file__).resolve().parent.parent / "gesture_draw_v2.py"
_spec = importlib.util.spec_from_file_location("v2mod", _V2_PATH)
v2 = importlib.util.module_from_spec(_spec)
sys.modules["v2mod"] = v2
_spec.loader.exec_module(v2)

Lm = collections.namedtuple("Lm", "x y")


def make_hand(index=0.4, middle=0.6, thumb=0.2, spread=0.1,
              index_x=None, thumb_x=None):
    """A synthetic 21-landmark right hand.

    Defaults (used by DRAW tests): index tip clearly above its PIP
    (index < 0.5) and middle folded (middle=0.6 > 0.5), so the default
    clasp is a DRAW gesture.
    index_x / thumb_x override the tips' x so a pinch test can place the
    thumb-and-index tips close together.
    """
    lms = [Lm(0.5, 0.5)] * 21
    # wrist (0) and middle MCP (9) pin the hand-size used by pinch_ratio
    lms[0] = Lm(0.5, 0.9)
    lms[9] = Lm(0.5, 0.4)
    ix = 0.46 if index_x is None else index_x
    tx = 0.40 if thumb_x is None else thumb_x
    # index finger
    lms[6] = Lm(0.44, 0.5)   # PIP below tip so tip.y < pip.y
    lms[8] = Lm(ix, index)
    # middle finger
    lms[10] = Lm(0.54, 0.5)
    lms[12] = Lm(0.53, middle)
    # ring/pinky/thumb for the PALM test
    lms[14] = Lm(0.62, 0.5)
    lms[16] = Lm(0.64, 0.6)
    lms[18] = Lm(0.70, 0.5)
    lms[20] = Lm(0.72, 0.6)
    lms[3] = Lm(0.42, 0.5)
    lms[4] = Lm(tx, thumb)
    return lms


# ------------------------------------------------------------------ #
# Phase 3 - gesture classification                                   #
# ------------------------------------------------------------------ #
def test_draw_mode_index_only():
    hand = make_hand(index=0.3, middle=0.6)
    assert v2.analyse_hand(hand).mode == v2.DRAW


def test_hover_mode_index_and_middle():
    hand = make_hand(index=0.3, middle=0.3)
    assert v2.analyse_hand(hand).mode == v2.HOVER


def test_none_when_fist():
    hand = make_hand(index=0.6, middle=0.6)
    assert v2.analyse_hand(hand).mode == v2.NONE


def test_no_hand_returns_none():
    gs = v2.analyse_hand(None)
    assert gs.mode == v2.NONE and gs.pinch == 1.0


def test_cursor_draw_uses_index_tip():
    hand = make_hand(index=0.3)
    cx, cy = v2.cursor_point(hand, v2.DRAW, 640, 480)
    assert cx == int(hand[8].x * 640)
    assert cy == int(hand[8].y * 480)


def test_cursor_hover_uses_midpoint():
    hand = make_hand(index=0.3, middle=0.3)
    cx, cy = v2.cursor_point(hand, v2.HOVER, 640, 480)
    expect_x = int((hand[8].x + hand[12].x) * 0.5 * 640)
    expect_y = int((hand[8].y + hand[12].y) * 0.5 * 480)
    assert (cx, cy) == (expect_x, expect_y)


def test_pinch_ratio_small_when_tips_close():
    # thumb tip (x=0.40, y=0.30) next to index tip (x=0.42, y=0.30)
    hand = make_hand(index=0.30, thumb=0.30, index_x=0.42, thumb_x=0.40)
    assert v2.pinch_ratio(hand) < v2.PINCH_ON_RATIO


def test_pinch_ratio_large_when_open():
    hand = make_hand(index=0.30, thumb=0.05, index_x=0.60, thumb_x=0.10)
    assert v2.pinch_ratio(hand) > v2.PINCH_ON_RATIO


# ------------------------------------------------------------------ #
# Phase 1.2 - Canvas interpolation                                    #
# ------------------------------------------------------------------ #

def test_canvas_first_point_draws_dot():
    canvas = v2.Canvas(120, 90)
    canvas.paint_next(30, 30)
    assert canvas.prev_point == (30, 30)
    assert np.any(canvas.layer != v2.CANVAS_BG)


def test_canvas_line_connects_two_points():
    canvas = v2.Canvas(120, 90)
    canvas.paint_next(0, 40)
    canvas.paint_next(100, 40)
    # some middle pixel of the segment is painted (not just 2 dots)
    mid = int(100 * 0.5)
    assert tuple(canvas.layer[40, mid]) == v2.COLOR_RED


def test_canvas_reset_breaks_stroke():
    canvas = v2.Canvas(120, 90)
    canvas.paint_next(0, 40)
    canvas.reset_stroke()
    canvas.paint_next(100, 40)
    # no connecting line across the gap: mid stays background
    assert tuple(canvas.layer[40, 50]) == v2.CANVAS_BG


def test_canvas_undo_pops_history():
    canvas = v2.Canvas(120, 90)
    canvas.paint_next(10, 10)
    canvas.push_history()
    canvas.paint_next(80, 60)
    assert canvas.undo() is True
    assert np.any(canvas.layer != v2.CANVAS_BG)  # first stroke remains


# ------------------------------------------------------------------ #
# Phase 2 - Header stable selection                                  #
# ------------------------------------------------------------------ #

def _header_ready():
    h = v2.Header()
    h.build(640)
    return h


def test_header_hit_rejects_outside():
    h = _header_ready()
    assert h.hit(5, h.height + 5) is None


def test_header_hit_finds_button():
    h = _header_ready()
    # first button spans [0, width/5) - BLUE
    tool, label, color, x1, x2 = h.hit(10, 30)
    assert tool == v2.TOOL_BLUE


def test_hover_stable_selects_after_frames():
    h = _header_ready()
    chosen = None
    for _ in range(v2.STABLE_HOVER_FRAMES + 1):
        chosen = h.update((10, 30), 0.9, v2.HOVER)
        if chosen is not None:
            break
    assert chosen == v2.TOOL_BLUE
    # it must not immediately retrigger without leaving the box
    assert h.update((10, 30), 0.9, v2.HOVER) is None


def test_move_away_resets_hover():
    h = _header_ready()
    for _ in range(v2.STABLE_HOVER_FRAMES + 1):
        if h.update((10, 30), 0.9, v2.HOVER) is not None:
            break
    # moving outside the header resets the counter
    h.update((630, 200), 0.9, v2.HOVER)
    assert h.update((10, 30), 0.9, v2.HOVER) is None
    # ...and re-hovering long enough finally selects
    chosen = None
    for _ in range(v2.STABLE_HOVER_FRAMES + 1):
        chosen = h.update((10, 30), 0.9, v2.HOVER)
        if chosen is not None:
            break
    assert chosen == v2.TOOL_BLUE


def test_pinch_click_selects_immediately():
    h = _header_ready()
    # pinch (ratio < PINCH_ON_RATIO) inside the strip => instant select
    chosen = h.update((10, 30), v2.PINCH_ON_RATIO * 0.5, v2.DRAW)
    assert chosen == v2.TOOL_BLUE
    # holding pinch doesn't repeat
    assert h.update((10, 30), v2.PINCH_ON_RATIO * 0.5, v2.DRAW) is None


def test_pinch_outside_header_does_not_select():
    h = _header_ready()
    # cursor inside strip region but x below strip (y>height) - no click
    assert h.update((630, h.height + 5), v2.PINCH_ON_RATIO * 0.5,
                    v2.DRAW) is None


# ------------------------------------------------------------------ #
# v2.1 - random colour / smoothing / bridging helpers                #
# ------------------------------------------------------------------ #

def test_random_vibrant_color_is_bgr_tuple():
    c = v2.random_vibrant_color()
    assert len(c) == 3
    assert all(0 <= v <= 255 for v in c)


def test_random_colors_distinct_across_steps():
    # golden-angle stepping should not return identical colours every time
    colors = {v2.random_vibrant_color(offset=i) for i in range(8)}
    assert len(colors) >= 3


def test_moving_average_returns_midpoint():
    pts = collections.deque([(10, 20), (20, 40)])
    assert v2.moving_average(pts) == (15, 30)


def test_moving_average_empty_is_none():
    assert v2.moving_average(collections.deque()) is None


def test_predict_bridged_point_extrapolates():
    hist = collections.deque([(0, 0), (10, 0)])
    assert v2.predict_bridged_point(hist) == (20, 0)


def test_predict_bridged_point_needs_two_samples():
    assert v2.predict_bridged_point(collections.deque([(5, 5)])) is None


# ------------------------------------------------------------------ #
# v2.1 - Canvas opacity, shapes, flood fill                          #
# ------------------------------------------------------------------ #

def test_canvas_opacity_blends_with_background():
    canvas = v2.Canvas(120, 90)
    canvas.opacity = 0.5
    canvas.paint_next(30, 30, color=v2.COLOR_RED)
    px = tuple(canvas.layer[30, 30])
    # blended halfway between white background and red
    assert abs(px[2] - (v2.COLOR_RED[2] + 255) // 2) <= 2


def test_canvas_high_opacity_is_solid():
    canvas = v2.Canvas(120, 90)
    canvas.paint_next(30, 30, color=v2.COLOR_RED)
    assert tuple(canvas.layer[30, 30]) == v2.COLOR_RED


def test_canvas_draw_shape_rect_commits():
    canvas = v2.Canvas(120, 90)
    canvas.draw_shape(v2.TOOL_SHAPE_RECT, (10, 10), (60, 40),
                      color=v2.COLOR_BLUE, brush=3)
    assert tuple(canvas.layer[25, 10]) == v2.COLOR_BLUE


def test_canvas_draw_shape_line_commits():
    canvas = v2.Canvas(120, 90)
    canvas.draw_shape(v2.TOOL_SHAPE_LINE, (0, 40), (100, 40),
                      color=v2.COLOR_BLUE, brush=3)
    assert tuple(canvas.layer[40, 50]) == v2.COLOR_BLUE


def test_canvas_flood_fill_fills_region():
    canvas = v2.Canvas(120, 90)
    # close a square with a wall, then fill its interior
    canvas.draw_shape(v2.TOOL_SHAPE_RECT, (10, 10), (40, 40),
                      color=v2.COLOR_BLACK, brush=2)
    canvas.flood_fill(20, 20, color=v2.COLOR_RED)
    assert tuple(canvas.layer[20, 20]) == v2.COLOR_RED


# ------------------------------------------------------------------ #
# v2.1 - Sidebar stable selection                                    #
# ------------------------------------------------------------------ #

def _sidebar_ready():
    s = v2.Sidebar()
    s.build(400)
    return s


def test_sidebar_hit_outside_is_none():
    s = _sidebar_ready()
    assert s.hit(v2.SIDEBAR_WIDTH + 10, 200) is None


def test_sidebar_hit_top_region_is_none():
    s = _sidebar_ready()
    assert s.hit(10, 20) is None  # above header top offset


def test_sidebar_hit_finds_shape_button():
    s = _sidebar_ready()
    action, label, color, y1, y2 = s.hit(10, s.top + 1)
    assert action == v2.TOOL_SHAPE_LINE


def test_sidebar_hover_selects_after_frames():
    s = _sidebar_ready()
    chosen = None
    for _ in range(v2.STABLE_HOVER_FRAMES + 1):
        chosen = s.update((10, s.top + 1), 0.9, v2.HOVER)
        if chosen is not None:
            break
    assert chosen == v2.TOOL_SHAPE_LINE


def test_sidebar_pinch_selects_immediately():
    s = _sidebar_ready()
    chosen = s.update((10, s.top + 1), v2.PINCH_ON_RATIO * 0.5, v2.DRAW)
    assert chosen == v2.TOOL_SHAPE_LINE


def test_header_includes_random_color_button():
    h = _header_ready()
    # the RANDOM tool should be reachable through the top palette
    found = any(tid == v2.TOOL_RANDOM for (tid, *_rest) in h.buttons)
    assert found
