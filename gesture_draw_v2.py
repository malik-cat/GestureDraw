"""
gesture_draw_v2.py — Professional GestureDraw (single-file build)
=================================================================
Precise, low-lag, hand-tracked air canvas for Python 3.11+, OpenCV 4.8+,
MediaPipe 1.0+ (Tasks API) and NumPy 1.24+.

This single, cohesive script implements the "Air Canvas v2" brief plus
the professional feature set (v2.1):

Phase 1 — Smooth, continuous lines.
    * HandLandmarker runs at min_detection/tracking/presence confidence
      0.8 in RunningMode.VIDEO (frame-to-frame tracking).
    * Interpolation: the previous tip (xp, yp) is always connected to the
      current tip with cv2.line() so skipped MediaPipe frames never crack
      a stroke.
    * Zero-drop continuity: when the hand disappears mid-stroke, up to
      BRIDGE_FRAMES (4) frames are bridged by motion-vector extrapolation
      (predict_bridged_point); the stroke only ends once it is lost for
      MAX_STROKE_LOST_FRAMES.

Phase 2 — Professional toolbar: a top palette and a left sidebar.
    * Header (y: 0..HEADER_HEIGHT): BLUE / GREEN / RED / RANDOM COLOR /
      ERASE / CLEAR ALL; the active tool is outlined in yellow.
    * Sidebar (x: 0..SIDEBAR_WIDTH, below the header): shape tools
      (LINE, RECT, CIRCLE, TRIANGLE, STAR), RANDOM SHAPE, BUCKET FILL,
      FINE eraser and parameter buttons (brush / opacity / smoothing).
    * STABLE selection fires in two independent ways:
        a) hovering a button for STABLE_HOVER_FRAMES consecutive frames;
        b) a pinch (thumb+index) release inside the panel.
    * CLEAR ALL is guarded: the first tap arms it; tapping again within
      CLEAR_CONFIRM_S really wipes the canvas.

Phase 3 — Accurate gestures from relative landmarks.
    * DRAW   : index tip (8) clearly above its PIP (6), middle folded.
    * HOVER  : index + middle raised; sticky cursor = midpoint of 8/12.
    * PINCH  : thumb-tip (4) / index-tip (8) ratio below PINCH_ON_RATIO;
      pinching on the canvas anchors a shape and release commits it.

Phase 4 — Professional drawing parameters.
    * Brush thickness : +/- keys and sidebar BRUSH+/BRUSH-.
    * Opacity         : '[' / ']' keys and sidebar OPAC+/OPAC-; the pen
      is solid at 1.0 and a translucent highlighter at lower values.
    * Smoothing       : moving-average over raw landmark coordinates,
      toggled with 't' or the sidebar SMOOTH button.
    * FINE eraser     : a precision eraser (small brush) alongside the
      large stroke eraser.

Phase 5 — Deterministic & random colour / shape tools.
    * RANDOM COLOR pulls a vivid colour from a golden-angle HSV wheel;
      every new stroke gets a fresh colour while the RANDOM tool is on.
    * RANDOM SHAPE makes the anchor-drag-release stroke commit one of
      LINE / RECT / CIRCLE / TRIANGLE / STAR chosen at random each time.
    * BUCKET FILL flood-fills a connected region on release.

Run:   python gesture_draw_v2.py
Keys:  q quit · u undo · c clear (guarded) · s save PNG ·
       f full screen · + / - brush · [ / ] opacity · t smoothing ·
       r random colour · g random shape
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===================================================================== #
# 1. Configuration                                                      #
# ===================================================================== #
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

CAMERA_SOURCE = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30.0

# MediaPipe confidence -----------------------------------------------------
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MIN_PRESENCE_CONFIDENCE = 0.8

# Tool ids ---------------------------------------------------------------
TOOL_BLUE = "blue"
TOOL_GREEN = "green"
TOOL_RED = "red"
TOOL_RANDOM = "random"
TOOL_ERASER = "eraser"
TOOL_ERASER_FINE = "eraser_fine"
TOOL_CLEAR = "clear"
TOOL_SHAPE_LINE = "line"
TOOL_SHAPE_RECT = "rect"
TOOL_SHAPE_CIRCLE = "circle"
TOOL_SHAPE_TRIANGLE = "triangle"
TOOL_SHAPE_STAR = "star"
TOOL_RANDOM_SHAPE = "random_shape"
TOOL_BUCKET = "bucket"

SHAPE_IDS = [
    TOOL_SHAPE_LINE, TOOL_SHAPE_RECT, TOOL_SHAPE_CIRCLE,
    TOOL_SHAPE_TRIANGLE, TOOL_SHAPE_STAR,
]

# Panels ---------------------------------------------------------------
PARAM_BRUSH_UP = "p_brush_up"
PARAM_BRUSH_DOWN = "p_brush_down"
PARAM_OPACITY_UP = "p_opacity_up"
PARAM_OPACITY_DOWN = "p_opacity_down"
PARAM_SMOOTH_TOGGLE = "p_smoothing"
PARAM_IDS = {PARAM_BRUSH_UP, PARAM_BRUSH_DOWN, PARAM_OPACITY_UP,
             PARAM_OPACITY_DOWN, PARAM_SMOOTH_TOGGLE}

# BGR colours -----------------------------------------------------------
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_UI_BG = (40, 40, 60)
CANVAS_BG = COLOR_WHITE
ERASER_BRUSH = 34
ERASER_FINE_BRUSH = 12
COLOR_RANDOM_FG = (200, 120, 255)   # violet used for the RANDOM button

# Header / UI ------------------------------------------------------------
HEADER_HEIGHT = 100
SIDEBAR_WIDTH = 150
STABLE_HOVER_FRAMES = 15          # ~0.5 s at 30 fps
SELECT_COOLDOWN_S = 0.4
CLEAR_CONFIRM_S = 1.0

# Stroke / brush ---------------------------------------------------------
DEFAULT_BRUSH = 8
BRUSH_STEP = 2
MIN_BRUSH = 2
MAX_BRUSH = 60

# Drawing parameters -----------------------------------------------------
DEFAULT_OPACITY = 1.0
MIN_OPACITY = 0.15
OPACITY_STEP = 0.15
SMOOTHING_WINDOW = 5              # moving-average filter length
DEFAULT_SMOOTHING = True
FINGER_TAP_DIST = 28              # px radius: a tap, not a shape drag

# Random colour -----------------------------------------------------------
HUE_GOLDEN_ANGLE = 137.50776405003785
MIN_VIBRANT_SAT = 150
MIN_VIBRANT_VAL = 180

# Gesture thresholds -----------------------------------------------------
FINGER_UP_MARGIN = 0.03
PINCH_ON_RATIO = 0.25
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
THUMB_TIP = 4

# Continuity --------------------------------------------------------------
MAX_STROKE_LOST_FRAMES = 6
BRIDGE_FRAMES = 4

# Keyboard ---------------------------------------------------------------
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_UNDO = ord("u")
KEY_CLEAR = ord("c")
KEY_BRUSH_UP = ord("+")
KEY_BRUSH_DOWN = ord("-")
KEY_FULLSCREEN = ord("f")
KEY_RANDOM = ord("r")
KEY_SMOOTH = ord("t")
KEY_OPACITY_UP = ord("]")
KEY_OPACITY_DOWN = ord("[")
KEY_RANDOM_SHAPE_CYCLE = ord("g")

WINDOW_NAME = "GestureDraw v2 - Air Canvas"

# ===================================================================== #
# 2. Gesture classification                                             #
# ===================================================================== #
DRAW, HOVER, PALM, NONE = "DRAW", "HOVER", "PALM", "NONE"


class Gesture:
    """One frame's gesture descriptors."""

    __slots__ = ("mode", "pinch")

    def __init__(self, mode: str, pinch: float = 1.0):
        self.mode = mode
        self.pinch = pinch


def pinch_ratio(hand) -> float:
    """Distance thumb-tip → index-tip normalised by hand size."""
    if hand is None:
        return 1.0
    gap = ((hand[THUMB_TIP].x - hand[INDEX_TIP].x) ** 2 +
           (hand[THUMB_TIP].y - hand[INDEX_TIP].y) ** 2) ** 0.5
    size = ((hand[0].x - hand[9].x) ** 2 +
            (hand[0].y - hand[9].y) ** 2) ** 0.5
    return gap / (size + 1e-6)


def analyse_hand(hand) -> Gesture:
    """Classify the hand: DRAW / HOVER / PALM / NONE."""
    if hand is None:
        return Gesture(NONE, 1.0)

    def extended(tip, pip):
        return hand[tip].y < hand[pip].y - FINGER_UP_MARGIN

    index_up = extended(INDEX_TIP, INDEX_PIP)
    middle_up = extended(MIDDLE_TIP, MIDDLE_PIP)

    if index_up and not middle_up:
        mode = DRAW
    elif index_up and middle_up:
        mode = HOVER
    elif all(extended(t, p) for t, p in
             ((4, 3), (8, 6), (12, 10), (16, 14), (20, 18))):
        mode = PALM
    else:
        mode = NONE

    return Gesture(mode, pinch_ratio(hand))


def cursor_point(hand, mode: str, width: int, height: int) -> tuple:
    """Pixel cursor: HOVER = midpoint of landmarks 8/12, else index tip."""
    if mode == HOVER:
        return (
            int((hand[INDEX_TIP].x + hand[MIDDLE_TIP].x) * 0.5 * width),
            int((hand[INDEX_TIP].y + hand[MIDDLE_TIP].y) * 0.5 * height),
        )
    return int(hand[INDEX_TIP].x * width), int(hand[INDEX_TIP].y * height)


# ------------------------------------------------------------------------- #
# 2.1 Random colour, smoothing and stroke-bridging helpers                  #
# ------------------------------------------------------------------------- #
def random_vibrant_color(offset: int = 0) -> tuple:
    """A vivid BGR colour stepped along a golden-angle HSV wheel.

    Each call adds ~137.5° in hue so consecutive colours are distinct.
    """
    # OpenCV HSV hue is 0..179, whereas the wheel works in 0..360
    hue = int((random.random() * 360.0 + HUE_GOLDEN_ANGLE * offset) %
              360.0)
    sat = random.randint(MIN_VIBRANT_SAT, 255)
    val = random.randint(MIN_VIBRANT_VAL, 255)
    hsv = np.uint8([[[hue // 2, sat, val]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def moving_average(points: deque) -> tuple:
    """Mean of the (x, y) landmark history, None when empty."""
    if not points:
        return None
    xs = int(round(sum(p[0] for p in points) / len(points)))
    ys = int(round(sum(p[1] for p in points) / len(points)))
    return xs, ys


def predict_bridged_point(history: deque) -> tuple | None:
    """Return the extrapolated next point using the last motion vector."""
    if history is None or len(history) < 2:
        return None
    p0, p1 = history[-2], history[-1]
    return (p1[0] + (p1[0] - p0[0]),
            p1[1] + (p1[1] - p0[1]))


# ===================================================================== #
# 3. Canvas: interpolation, opacity, shapes, flood fill                 #
# ===================================================================== #
class Canvas:
    """The persistent drawing layer.

    paint_next() connects the previous tip to the new point (dot for the
    very first sample). Opacity is a real alpha blend at paint time so
    low values behave like a highlighter. draw_shape() commits an
    outlined LINE / RECT / CIRCLE / TRIANGLE / STAR. flood_fill() fills a
    connected region.
    """

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.layer[:] = CANVAS_BG
        self.color = COLOR_RED
        self.brush = DEFAULT_BRUSH
        self.opacity = DEFAULT_OPACITY
        self.prev_point = None
        self._history = []
        self._max_history = 20
        self._recent = deque(maxlen=8)

    def reset_stroke(self):
        """Start a fresh stroke (prev point and its trace are dropped)."""
        self.prev_point = None
        self._recent.clear()

    def paint_next(self, xc: int, yc: int, color=None, brush=None):
        """Draw a segment prev→current; store the new previous point."""
        color = color if color is not None else self.color
        brush = brush if brush is not None else self.brush
        tmp = np.zeros_like(self.layer)
        if self.prev_point is None:
            cv2.circle(tmp, (xc, yc), max(1, brush // 2), color, -1)
        else:
            xp, yp = self.prev_point
            cv2.line(tmp, (xp, yp), (xc, yc), color, brush, cv2.LINE_AA)
        self._blend(tmp)
        self._recent.append((xc, yc))
        self.prev_point = (xc, yc)

    def _blend(self, tmp: np.ndarray):
        """Blend a rendered scratch into the layer subject to opacity."""
        mask = np.any(tmp != 0, axis=-1)
        if not mask.any():
            return
        if self.opacity >= 1.0 - 1e-6:
            self.layer[mask] = tmp[mask]
        else:
            top = tmp[mask].astype(np.float32)
            bot = self.layer[mask].astype(np.float32)
            self.layer[mask] = (
                top * self.opacity + bot * (1 - self.opacity)
            ).astype(np.uint8)

    # ---- shapes ------------------------------------------------------- #
    def shape_points(self, shape_id: str, p0, p1) -> np.ndarray:
        """Coordinates for the shape: used for preview and commit."""
        (x0, y0), (x1, y1) = p0, p1
        if shape_id == TOOL_SHAPE_LINE:
            return np.array([p0, p1], np.int32)
        if shape_id == TOOL_SHAPE_RECT:
            return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                            np.int32)
        if shape_id == TOOL_SHAPE_TRIANGLE:
            return np.array([(x0, y0), (x1, y1), ((x0 + x1) // 2, y0)],
                            np.int32)
        if shape_id == TOOL_SHAPE_CIRCLE:
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            r = max(2, int(round(math.hypot(x1 - x0, y1 - y0) * 0.5)))
            pts = []
            for deg in range(0, 360, 6):
                pts.append((int(cx + r * math.cos(math.radians(deg))),
                            int(cy + r * math.sin(math.radians(deg)))))
            return np.array(pts, np.int32)
        if shape_id == TOOL_SHAPE_STAR:
            return _star_verts((x0 + x1) // 2, (y0 + y1) // 2,
                               max(2, int(round(
                                   math.hypot(x1 - x0, y1 - y0) * 0.5))))
        return np.array([p0, p1], np.int32)

    def draw_shape(self, shape_id: str, p0, p1, color=None, brush=None,
                   fill: bool = False):
        """Commit an outlined (or filled) shape between p0 and p1."""
        color = color if color is not None else self.color
        brush = brush if brush is not None else self.brush
        poly = self.shape_points(shape_id, p0, p1)
        tmp = np.zeros_like(self.layer)
        if fill:
            cv2.fillPoly(tmp, [poly], color)
        else:
            cv2.polylines(tmp, [poly], True, color, brush, cv2.LINE_AA)
        self._blend(tmp)
        self.push_history()

    def preview_shape(self, frame, shape_id: str, p0, p1,
                      color=None, brush=None):
        """Overlay a translucent shape preview onto the camera frame."""
        color = color if color is not None else self._preview_color()
        brush = brush if brush is not None else max(3, self.brush // 2)
        poly = self.shape_points(shape_id, p0, p1)
        tmp = np.zeros_like(frame)
        cv2.polylines(tmp, [poly], True, color, brush, cv2.LINE_AA)
        mask = np.any(tmp != 0, axis=-1)
        alpha = 0.6
        top = tmp[mask].astype(np.float32)
        bot = frame[mask].astype(np.float32)
        frame[mask] = (top * alpha + bot * (1 - alpha)).astype(np.uint8)

    def _preview_color(self):
        return self.color

    # ---- flood fill ------------------------------------------------------ #
    def flood_fill(self, x: int, y: int, color=None, tolerance: int = 16):
        """Fill the connected region around (x, y)."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        color = color if color is not None else self.color
        seed = (int(x), int(y))
        mask = np.zeros((self.height + 2, self.width + 2), np.uint8)
        self.push_history()
        cv2.floodFill(self.layer, mask, seed,
                      tuple(int(c) for c in color),
                      (tolerance,) * 3, (tolerance,) * 3,
                      cv2.FLOODFILL_FIXED_RANGE)
        self.reset_stroke()
        return True

    # --------------------------------------------------------------------- #
    def set_color(self, color):
        self.color = color

    def undo(self):
        if not self._history:
            return False
        self.layer = self._history.pop()[0].copy()
        self.reset_stroke()
        return True

    def push_history(self):
        if self._history and np.array_equal(self._history[-1][0],
                                            self.layer):
            return
        self._history.append((self.layer.copy(),))
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def clear(self):
        self.push_history()
        self.layer[:] = CANVAS_BG
        self.reset_stroke()

    def overlay(self, frame):
        """Merge the board over the live camera frame."""
        comp = frame.copy()
        drawn = np.any(self.layer != CANVAS_BG, axis=-1)
        comp[drawn] = self.layer[drawn]
        return comp


def _star_verts(cx: int, cy: int, radius: int) -> np.ndarray:
    inner = radius * 0.5
    pts = []
    for i in range(10):
        phase = i * math.pi / 5 - math.pi / 2
        r = radius if i % 2 == 0 else inner
        pts.append((int(cx + r * math.cos(phase)),
                    int(cy + r * math.sin(phase))))
    return np.array(pts, np.int32)


# ===================================================================== #
# 4. FPS meter + pacing                                                 #
# ===================================================================== #
class FPS:
    def __init__(self, target: float = TARGET_FPS):
        self.target = target
        self.fps = 0.0
        self._start = time.perf_counter()
        self._frames = 0

    def tick(self):
        self._frames += 1
        now = time.perf_counter()
        dt = now - self._start
        if dt >= 0.5:
            self.fps = self._frames / dt
            self._frames = 0
            self._start = now

    def pace(self):
        now = time.perf_counter()
        elapsed = now - self._start
        if elapsed < 1.0 / self.target:
            time.sleep(1.0 / self.target - elapsed)
        self.tick()


# ===================================================================== #
# 5. Toolbar panels with STABLE selection                               #
# ===================================================================== #
class _StableSelect:
    """HOVER-for-N-frames OR pinch-release fire rules (shared)."""

    def __init__(self):
        self._hover_id = None
        self._hover_count = 0
        self._pinch_active = False
        self._last_fire = 0.0

    def update(self, hit_name, pinch, mode):
        now = time.perf_counter()
        pinch_down = pinch is not None and pinch < PINCH_ON_RATIO

        if pinch_down and not self._pinch_active and hit_name is not None:
            self._pinch_active = True
            if now - self._last_fire >= SELECT_COOLDOWN_S:
                self._last_fire = now
                self._reset()
                return hit_name
        elif not pinch_down:
            self._pinch_active = False

        if hit_name is not None and mode == HOVER:
            if hit_name == self._hover_id:
                self._hover_count += 1
            else:
                self._hover_id, self._hover_count = hit_name, 1
            if self._hover_count >= STABLE_HOVER_FRAMES:
                self._last_fire = now
                self._reset()
                return hit_name
        else:
            self._reset_hover()
        return None

    def _reset_hover(self):
        self._hover_id = None
        self._hover_count = 0

    def _reset(self):
        self._reset_hover()


class Header:
    """Top toolbar strip with the colour/erase/clear buttons."""

    WIDGETS = [
        (TOOL_BLUE, "BLUE", COLOR_BLUE),
        (TOOL_GREEN, "GREEN", COLOR_GREEN),
        (TOOL_RED, "RED", COLOR_RED),
        (TOOL_RANDOM, "RANDOM", COLOR_RANDOM_FG),
        (TOOL_ERASER, "ERASE", COLOR_WHITE),
        (TOOL_CLEAR, "CLEAR ALL", (60, 60, 70)),
    ]

    def __init__(self, height: int = HEADER_HEIGHT):
        self.height = height
        self.buttons = []
        self._stable = _StableSelect()

    def build(self, width: int):
        n = len(self.WIDGETS)
        bw = width / n
        self.buttons = [
            (tid, label, color, i * bw, (i + 1) * bw)
            for i, (tid, label, color) in enumerate(self.WIDGETS)
        ]

    def hit(self, x, y):
        if y is None or y < 0 or y >= self.height:
            return None
        for (tid, label, color, x1, x2) in self.buttons:
            if x1 <= x < x2:
                return (tid, label, color, x1, x2)
        return None

    def update(self, cursor, pinch, mode: str):
        btn = self.hit(*cursor) if cursor else None
        return self._stable.update(btn[0] if btn else None, pinch, mode)

    def _reset_hover(self):
        self._stable._reset_hover()

    def draw(self, frame, active_tool: str, armed: bool = False):
        width = frame.shape[1]
        self.build(width)
        for (tid, label, color, x1f, x2f) in self.buttons:
            x1, x2 = int(x1f), int(x2f)
            cv2.rectangle(frame, (x1, 0), (x2 - 3, self.height - 1),
                          color, -1)
            cv2.rectangle(frame, (x1, 0), (x2 - 3, self.height - 1),
                          COLOR_BLACK, 1)
            luma = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
            text_c = COLOR_BLACK if luma > 140 else COLOR_WHITE
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, 1)
            pos = (x1 + (x2 - 3 - x1) // 2 - tw // 2,
                   self.height // 2 + 5)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        text_c, 1, cv2.LINE_AA)
            if tid == active_tool or (tid == TOOL_CLEAR and armed):
                cv2.rectangle(frame, (x1 + 1, 1), (x2 - 4,
                                                   self.height - 2),
                              COLOR_YELLOW, 3)


class Sidebar:
    """Left strip below the header: shapes, fills, erasers, parameters."""

    WIDGETS = [
        (TOOL_SHAPE_LINE, "LINE", (90, 90, 100)),
        (TOOL_SHAPE_RECT, "RECT", (90, 90, 100)),
        (TOOL_SHAPE_CIRCLE, "CIRCLE", (90, 90, 100)),
        (TOOL_SHAPE_TRIANGLE, "TRI", (90, 90, 100)),
        (TOOL_SHAPE_STAR, "STAR", (90, 90, 100)),
        (TOOL_RANDOM_SHAPE, "SHAPE?", (130, 70, 100)),
        (TOOL_BUCKET, "FILL", (40, 130, 200)),
        (TOOL_ERASER, "ERASE", (70, 70, 70)),
        (TOOL_ERASER_FINE, "FINE", (110, 110, 110)),
        (PARAM_BRUSH_UP, "BRUSH+", (40, 110, 60)),
        (PARAM_BRUSH_DOWN, "BRUSH-", (40, 110, 60)),
        (PARAM_OPACITY_UP, "OPAC+", (120, 90, 50)),
        (PARAM_OPACITY_DOWN, "OPAC-", (120, 90, 50)),
        (PARAM_SMOOTH_TOGGLE, "SMOOTH", (40, 80, 120)),
    ]

    def __init__(self, width: int = SIDEBAR_WIDTH,
                 top: int = HEADER_HEIGHT):
        self.width = width
        self.top = top
        self.buttons = []
        self._stable = _StableSelect()

    def build(self, height: int):
        usable = height - self.top
        bh = usable / len(self.WIDGETS)
        self.buttons = [
            (action, label, color, self.top + i * bh,
             self.top + (i + 1) * bh)
            for i, (action, label, color) in enumerate(self.WIDGETS)
        ]

    def hit(self, x, y):
        if not self.buttons:
            return None
        if not (0 <= x < self.width and y >= self.top):
            return None
        for (action, label, color, y1, y2) in self.buttons:
            if y1 <= y < y2:
                return (action, label, color, y1, y2)
        return None

    def update(self, cursor, pinch, mode: str):
        btn = self.hit(*cursor) if cursor else None
        hit_name = btn[0] if btn else None
        return self._stable.update(hit_name, pinch, mode)

    def draw(self, frame, active_tool: str, highlights: dict | None = None):
        height = frame.shape[0]
        self.build(height)
        highlights = highlights or {}
        for (action, label, color, y1f, y2f) in self.buttons:
            y1, y2 = int(y1f), int(y2f)
            cv2.rectangle(frame, (0, y1), (self.width - 1, y2 - 1),
                          color, -1)
            cv2.rectangle(frame, (0, y1), (self.width - 1, y2 - 1),
                          COLOR_BLACK, 1)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.38, 1)
            pos = ((self.width - tw) // 2, (y1 + y2) // 2 + 5)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, COLOR_WHITE, 1, cv2.LINE_AA)
            lit = action == active_tool or bool(highlights.get(action))
            if lit:
                cv2.rectangle(frame, (1, y1 + 1),
                              (self.width - 2, y2 - 2), COLOR_YELLOW, 2)


# ===================================================================== #
# 6. MediaPipe hand model                                               #
# ===================================================================== #
class HandTracker:
    def __init__(self, model_path=MODEL_PATH):
        try:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
                min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            )
            self._graph = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - model is committed
            raise RuntimeError(
                f"Could not load MediaPipe model {model_path}. Download the "
                f"'hand_landmarker' task file and place it next to the "
                f"script. Details: {exc}") from exc
        self._ts = 0

    def detect(self, frame):
        """Landmarks for the first hand, or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += max(1, int(1000 / TARGET_FPS))
        result = self._graph.detect_for_video(image, self._ts)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def release(self):
        self._graph.close()


# ===================================================================== #
# 7. GestureDrawApp — orchestration                                      #
# ===================================================================== #
TOOL_COLORS = {
    TOOL_RED: COLOR_RED, TOOL_GREEN: COLOR_GREEN, TOOL_BLUE: COLOR_BLUE,
}

TOOL_LABELS = {
    TOOL_RED: "RED", TOOL_GREEN: "GREEN", TOOL_BLUE: "BLUE",
    TOOL_RANDOM: "RANDOM", TOOL_ERASER: "ERASER",
    TOOL_ERASER_FINE: "FINE", TOOL_BUCKET: "FILL",
    TOOL_SHAPE_LINE: "LINE", TOOL_SHAPE_RECT: "RECT",
    TOOL_SHAPE_CIRCLE: "CIRCLE", TOOL_SHAPE_TRIANGLE: "TRI",
    TOOL_SHAPE_STAR: "STAR", TOOL_RANDOM_SHAPE: "SHAPE?",
}


class GestureDrawApp:
    """Camera → tracker → gestures → panels → drawing."""

    def __init__(self):
        self.tracker = HandTracker()
        self.canvas = Canvas()
        self.header = Header()
        self.sidebar = Sidebar()
        self.tool = TOOL_RED
        self.smoothing = DEFAULT_SMOOTHING
        self.fps_meter = FPS()
        self._lost_frames = 0
        self._pinch = False
        self._pinch_prev = False
        self._clear_armed = False
        self._shape = None
        self._bucket = None
        self._raw = deque(maxlen=SMOOTHING_WINDOW)

    # --------------------------------------------------------------------- #
    def run(self):
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            raise RuntimeError("Could not open the webcam. Close any "
                               "other app using it and retry.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.canvas = Canvas(FRAME_WIDTH, FRAME_HEIGHT)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        try:
            while True:
                self.fps_meter.tick()
                ok, frame = cap.read()
                if not ok:
                    continue

                if frame.shape[1] != FRAME_WIDTH or \
                        frame.shape[0] != FRAME_HEIGHT:
                    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                hand = self.tracker.detect(frame)
                gc = analyse_hand(hand)

                cursor = None
                if hand is not None:
                    cursor = cursor_point(hand, gc.mode, width, height)
                    if self.smoothing:
                        self._raw.append(cursor)
                        sm = moving_average(self._raw)
                        if sm is not None:
                            cursor = sm

                self._track_pinch(gc)
                self._handle_panels(cursor, gc)
                self._canvas_interaction(cursor, gc, hand, width, height)

                comp = self.canvas.overlay(frame)
                self._draw_preview(comp)
                self.header.draw(comp, self.tool, armed=self._clear_armed)
                self.sidebar.draw(comp, self.tool,
                                  highlights={PARAM_SMOOTH_TOGGLE:
                                              self.smoothing,
                                              PARAM_OPACITY_UP: False})
                self._draw_hud(comp)

                cv2.imshow(WINDOW_NAME, comp)
                self.fps_meter.pace()

                key = cv2.waitKey(1) & 0xFF
                if key != -1 and self._keyboard(key):
                    break
        finally:
            cap.release()
            self.tracker.release()
            cv2.destroyAllWindows()

    # --------------------------------------------------------------------- #
    def _track_pinch(self, gc):
        self._pinch_prev = self._pinch
        self._pinch = gc.pinch is not None and gc.pinch < PINCH_ON_RATIO

    def _is_press(self):
        return self._pinch and not self._pinch_prev

    def _is_release(self):
        return not self._pinch and self._pinch_prev

    # --------------------------------------------------------------------- #
    def _handle_panels(self, cursor, gc):
        """Tool selection from the header and sidebar panels."""
        if cursor is None:
            return
        x, y = cursor
        if y < HEADER_HEIGHT:
            tool = self.header.update(cursor, gc.pinch, gc.mode)
        elif x < SIDEBAR_WIDTH:
            tool = self.sidebar.update(cursor, gc.pinch, gc.mode)
        else:
            self.header._reset_hover()
            self.sidebar._stable._reset_hover()
            return
        if tool:
            self._apply_tool(tool)

    # --------------------------------------------------------------------- #
    def _canvas_interaction(self, cursor, gc, hand, width, height):
        if hand is None:
            self._bridge_dropout()
            return

        self._lost_frames = 0
        if cursor is None:
            self._cancel_canvas_state()
            return

        x, y = cursor
        if not (SIDEBAR_WIDTH <= x < width and HEADER_HEIGHT <= y < height):
            self._cancel_canvas_state()
            return

        if self.tool in SHAPE_IDS or self.tool == TOOL_RANDOM_SHAPE:
            self._shape_workflow(cursor, gc)
        elif self.tool == TOOL_BUCKET:
            self._bucket_workflow(cursor)
        else:
            self._stroke_workflow(cursor, gc)

    # --------------------------------------------------------------------- #
    def _cancel_canvas_state(self):
        self.canvas.reset_stroke()
        self._shape = None
        self._bucket = None

    # --------------------------------------------------------------------- #
    def _bridge_dropout(self):
        """Predictively extend the stroke while the hand is lost."""
        self._lost_frames += 1
        if self.canvas.prev_point is None:
            return
        if self._lost_frames <= BRIDGE_FRAMES:
            nxt = predict_bridged_point(self.canvas._recent)
            if nxt is not None:
                self.canvas.paint_next(nxt[0], nxt[1],
                                       color=self._paint_color())
        if self._lost_frames >= MAX_STROKE_LOST_FRAMES:
            self.canvas.reset_stroke()

    # --------------------------------------------------------------------- #
    def _stroke_workflow(self, cursor, gc):
        if gc.mode == DRAW:
            self._maybe_new_stroke_color()
            self.canvas.paint_next(cursor[0], cursor[1],
                                   color=self._paint_color())
        else:
            if self.canvas.prev_point is not None:
                self.canvas.push_history()
            self.canvas.reset_stroke()

    def _maybe_new_stroke_color(self):
        if self.tool == TOOL_RANDOM and self.canvas.prev_point is None:
            self.canvas.set_color(random_vibrant_color())

    # --------------------------------------------------------------------- #
    def _shape_workflow(self, cursor, gc):
        if self._is_press():
            self._maybe_new_stroke_color()
            self.canvas.push_history()
            self._shape = (cursor, cursor)
            return
        if self._shape is None:
            return
        anchor, _ = self._shape
        if self._pinch:
            self._shape = (anchor, cursor)
            return
        # lifted the pinch: commit the shape we dragged out
        p0, p1 = anchor, cursor
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < FINGER_TAP_DIST:
            self._shape = None
            self.canvas.reset_stroke()
            return
        shape = self._pick_shape()
        self.canvas.draw_shape(shape, p0, p1,
                               color=self._paint_color(),
                               brush=self.canvas.brush)
        self._shape = None
        self.canvas.reset_stroke()

    def _pick_shape(self):
        if self.tool == TOOL_RANDOM_SHAPE:
            return random.choice(SHAPE_IDS)
        return self.tool

    def _draw_preview(self, comp):
        if self._shape is None:
            return
        p0, p1 = self._shape
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < FINGER_TAP_DIST:
            return
        shape = self._pick_shape()
        self.canvas.preview_shape(comp, shape, p0, p1,
                                  color=self._paint_color(),
                                  brush=self.canvas.brush)

    # --------------------------------------------------------------------- #
    def _bucket_workflow(self, cursor):
        if self._is_press():
            self._bucket = cursor
            return
        if self._is_release() and self._bucket is not None:
            self.canvas.flood_fill(cursor[0], cursor[1],
                                   color=self._paint_color())
            self._bucket = None

    # --------------------------------------------------------------------- #
    def _paint_color(self):
        """Colour stamped by the current tool (erasers erase → white)."""
        if self.tool in (TOOL_ERASER, TOOL_ERASER_FINE):
            return CANVAS_BG
        if self.tool in TOOL_COLORS:
            return TOOL_COLORS[self.tool]
        return self.canvas.color

    # --------------------------------------------------------------------- #
    def _apply_tool(self, tool_id):
        if tool_id == PARAM_BRUSH_UP:
            self.canvas.brush = min(MAX_BRUSH,
                                    self.canvas.brush + BRUSH_STEP)
        elif tool_id == PARAM_BRUSH_DOWN:
            self.canvas.brush = max(MIN_BRUSH,
                                    self.canvas.brush - BRUSH_STEP)
        elif tool_id == PARAM_OPACITY_UP:
            self.canvas.opacity = min(1.0,
                                      self.canvas.opacity + OPACITY_STEP)
        elif tool_id == PARAM_OPACITY_DOWN:
            self.canvas.opacity = max(MIN_OPACITY,
                                      self.canvas.opacity - OPACITY_STEP)
        elif tool_id == PARAM_SMOOTH_TOGGLE:
            self.smoothing = not self.smoothing
        elif tool_id == TOOL_RANDOM:
            self.tool = TOOL_RANDOM
            self.canvas.set_color(random_vibrant_color())
        elif tool_id == TOOL_CLEAR:
            self._request_clear()
        elif tool_id in TOOL_COLORS:
            self.tool = tool_id
            self.canvas.set_color(TOOL_COLORS[tool_id])
        elif tool_id in (TOOL_ERASER, TOOL_ERASER_FINE, TOOL_BUCKET):
            self.tool = tool_id
        elif tool_id in SHAPE_IDS or tool_id == TOOL_RANDOM_SHAPE:
            self.tool = tool_id

    # --------------------------------------------------------------------- #
    def _request_clear(self):
        now = time.perf_counter()
        if self._clear_armed and now - self._clear_armed <= CLEAR_CONFIRM_S:
            self.canvas.clear()
            self._clear_armed = False
        else:
            self._clear_armed = now

    # --------------------------------------------------------------------- #
    def _draw_hud(self, comp):
        y = self.header.height + 22
        static = (f"FPS {self.fps_meter.fps:5.1f} | "
                  f"{TOOL_LABELS.get(self.tool, self.tool.upper())} | "
                  f"brush {self.canvas.brush} | "
                  f"opacity {self.canvas.opacity:.2f} | "
                  f"{'smooth' if self.smoothing else 'raw'}")
        if self._clear_armed:
            static += " | CLEAR?"
        cv2.putText(comp, static, (SIDEBAR_WIDTH + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1,
                    cv2.LINE_AA)

    # --------------------------------------------------------------------- #
    def _keyboard(self, key):
        if key == KEY_QUIT:
            return True
        if key == KEY_UNDO:
            self.canvas.undo()
        elif key == KEY_CLEAR:
            self._request_clear()
        elif key == KEY_SAVE:
            self._save_canvas()
        elif key in (KEY_BRUSH_UP, ord("=")):
            self.canvas.brush = min(MAX_BRUSH,
                                    self.canvas.brush + BRUSH_STEP)
        elif key == KEY_BRUSH_DOWN or key == ord("_"):
            self.canvas.brush = max(MIN_BRUSH,
                                    self.canvas.brush - BRUSH_STEP)
        elif key == KEY_OPACITY_UP:
            self.canvas.opacity = min(1.0,
                                      self.canvas.opacity + OPACITY_STEP)
        elif key == KEY_OPACITY_DOWN:
            self.canvas.opacity = max(MIN_OPACITY,
                                      self.canvas.opacity - OPACITY_STEP)
        elif key == KEY_SMOOTH:
            self.smoothing = not self.smoothing
        elif key == KEY_RANDOM:
            self._apply_tool(TOOL_RANDOM)
        elif key == KEY_RANDOM_SHAPE_CYCLE:
            self._apply_tool(TOOL_RANDOM_SHAPE)
        elif key == KEY_FULLSCREEN:
            self._toggle_fullscreen()
        return False

    def _toggle_fullscreen(self):
        cur = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL if cur else cv2.WINDOW_FULLSCREEN)

    def _save_canvas(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = BASE_DIR / "exports"
        folder.mkdir(exist_ok=True)
        cv2.imwrite(str(folder / f"gesturedraw_v2_{stamp}.png"),
                    self.canvas.layer)
        print("[v2] canvas saved to export")


def main():
    """Start the application."""
    try:
        app = GestureDrawApp()
        app.run()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
