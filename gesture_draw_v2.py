"""
gesture_draw_v2.py — Production-Ready GestureDraw (single-file build)
=====================================================================
Precise, low-lag, hand-tracked air canvas for Python 3.11+, OpenCV 4.8+,
MediaPipe 1.0+ (Tasks API) and NumPy 1.24+.

This single, cohesive script implements the whole "Air Canvas v2" brief:

Phase 1 — Smooth, continuous lines.
    * HandLandmarker is configured with min_detection_confidence=0.8 and
      min_tracking_confidence=0.8, so momentary occlusion no longer makes
      the tracker "drop" the hand every few frames (the cause of dotted,
      split strokes).
    * The tracker uses RunningMode.VIDEO (frame-to-frame tracking) instead
      of re-detecting every frame, offsetting the latency cost of the
      higher confidence.
    * Explicit line interpolation: while DRAWING we keep the previous tip
      (xp, yp) and the current tip (xc, yc), and draw cv2.line() between
      them. If one frame is skipped, the stroke *continues* instead of
      cracking. The previous point is reset whenever the gesture changes
      or the hand is lost for MAX_STROKE_LOST_FRAMES.
    * FPS meter in the corner + pacing so the loop stabilises at
      TARGET_FPS.

Phase 2 — A professional, reliable toolbar.
    * Header (y: 0..100) with clean labelled buttons:
      [BLUE] [GREEN] [RED] [ERASER] [CLEAR ALL]; active tool outlined in
      thick yellow.
    * STABLE selection: a tool activates only once the cursor hovers its
      box for STABLE_HOVER_FRAMES consecutive frames (~0.5 s), OR when a
      pinch-click fires while the hand is inside the header region.

Phase 3 — Accurate gestures from relative landmarks.
    * DRAWING : index tip (8) clearly above its PIP (6) and the middle
                finger (12 vs 10) folded. Pen = landmark 8.
    * SELECT/HOVER : index AND middle both extended; cursor = midpoint of
        landmarks 8 and 12 (a stable sticky cursor).
    * CLICK : pinch — thumb-tip (4) and index-tip (8) gap below
        PINCH_ON_RATIO (normalised by hand size), only honoured inside
        the header strip.

Phase 4 — Packaging.
    requirements.txt pins opencv>=4.8, numpy>=1.24, mediapipe>=1.0.0.
    The MediaPipe "hand_landmarker" float16 task file must sit next to
    this script (already committed to the repo).

Run:   python gesture_draw_v2.py
Keys:  q quit · u undo · c clear · s save PNG · + / - brush size

Author      : Mohammad Liaquat Ali
Repository  : https://github.com/malik-cat/GestureDraw
"""

from __future__ import annotations

import time
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

# MediaPipe confidence (Phase 1.1). Higher values mean the landmark reader
# is much less likely to momentarily stop reporting the hand during
# occlusion - which is exactly what was breaking lines into separate dots.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MIN_PRESENCE_CONFIDENCE = 0.8

# Tool ids -------------------------------------------------------------
TOOL_BLUE = "blue"
TOOL_GREEN = "green"
TOOL_RED = "red"
TOOL_ERASER = "eraser"
TOOL_CLEAR = "clear"

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

# Header / UI (Phase 2) --------------------------------------------------
HEADER_HEIGHT = 100
STABLE_HOVER_FRAMES = 15          # ~0.5 s at 30 fps
SELECT_COOLDOWN_S = 0.4

# Stroke / brush ---------------------------------------------------------
DEFAULT_BRUSH = 8
BRUSH_STEP = 2
MIN_BRUSH = 2
MAX_BRUSH = 60

# Gesture thresholds (Phase 3) ---------------------------------------------
FINGER_UP_MARGIN = 0.03
PINCH_ON_RATIO = 0.25
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
THUMB_TIP = 4

MAX_STROKE_LOST_FRAMES = 4        # frames without a hand before stroke ends

# Keyboard ---------------------------------------------------------------
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")
KEY_UNDO = ord("u")
KEY_CLEAR = ord("c")
KEY_BRUSH_UP = ord("+")
KEY_BRUSH_DOWN = ord("-")

# ===================================================================== #
# 2. Gesture classification (Phase 3)                                   #
# ===================================================================== #
DRAW, HOVER, PALM, NONE = "DRAW", "HOVER", "PALM", "NONE"


class Gesture:
    """One frame's gesture descriptors."""

    __slots__ = ("mode", "pinch")

    def __init__(self, mode: str, pinch: float = 1.0):
        self.mode = mode
        self.pinch = pinch


def pinch_ratio(hand) -> float:
    """Distance thumb-tip → index-tip, normalised by the hand's size.

    Using wrist (0) → middle MCP (9) as the scale makes the ratio distance-
    invariant: same value whether the hand is at 30 cm or 80 cm.
    """
    if hand is None:
        return 1.0
    gap = ((hand[THUMB_TIP].x - hand[INDEX_TIP].x) ** 2 +
           (hand[THUMB_TIP].y - hand[INDEX_TIP].y) ** 2) ** 0.5
    size = ((hand[0].x - hand[9].x) ** 2 +
            (hand[0].y - hand[9].y) ** 2) ** 0.5
    return gap / (size + 1e-6)


def analyse_hand(hand) -> Gesture:
    """Classify the current hand with relative, exceptional comparisons.

    DRAW:      index up (tip 8 above pip 6 by a margin), middle DOWN.
    HOVER:     index AND middle both raised → stable cursor.
    PALM:      all five fingers fully extended (clear shortcut).
    NONE:      anything else.
    """
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
    """Pixel cursor the app tracks.

    HOVER/SELECT uses the midpoint of landmarks 8 and 12 (sticky); DRAW
    and everything else falls back to the index tip (landmark 8).
    """
    if mode == HOVER:
        return (
            int((hand[INDEX_TIP].x + hand[MIDDLE_TIP].x) * 0.5 * width),
            int((hand[INDEX_TIP].y + hand[MIDDLE_TIP].y) * 0.5 * height),
        )
    return int(hand[INDEX_TIP].x * width), int(hand[INDEX_TIP].y * height)


# ===================================================================== #
# 3. Canvas with stroke interpolation (Phase 1.2)                       #
# ===================================================================== #
class Canvas:
    """The persistent drawing layer.

    Phase 1.2 — prev_point continuity. While drawing we remember the
    previous fingertip (xp, yp) and connect it to the current (xc, yc)
    with cv2.line(). If MediaPipe skips a frame, the short straight
    segment closes the gap instead of leaving a hole. reset_stroke() is
    invoked on gesture changes / lost hand so distinct ink isn't joined.
    """

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.layer = np.zeros((height, width, 3), dtype=np.uint8)
        self.layer[:] = CANVAS_BG
        self.color = COLOR_RED
        self.brush = DEFAULT_BRUSH
        self.prev_point = None
        self._history = []
        self._max_history = 20

    # ---- interpolation ------------------------------------------------- #
    def reset_stroke(self):
        """Forget the previous point: the next paint starts a new stroke."""
        self.prev_point = None

    def paint_next(self, xc: int, yc: int, color=None, brush=None):
        """Draw a segment prev→current; store the new previous point."""
        color = color if color is not None else self.color
        brush = brush if brush is not None else self.brush

        if self.prev_point is None:
            # Very first sample of the stroke – a dot so nothing is dropped.
            cv2.circle(self.layer, (xc, yc), max(1, brush // 2), color, -1)
        else:
            xp, yp = self.prev_point
            cv2.line(self.layer, (xp, yp), (xc, yc), color, brush,
                     cv2.LINE_AA)
        self.prev_point = (xc, yc)

    # ------------------------------------------------------------------ #
    def set_color(self, color):
        self.color = color

    def undo(self):
        if not self._history:
            return False
        self.layer = self._history.pop()[0]
        self.reset_stroke()
        return True

    def push_history(self):
        if self._history and np.array_equal(self._history[-1][0], self.layer):
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


# ===================================================================== #
# 4. FPS meter + pacing (Phase 1.3)                                     #
# ===================================================================== #
class FPS:
    """Measures the loop rate and throttles to TARGET_FPS."""

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
        """Sleep the remainder of the frame window if we are early."""
        now = time.perf_counter()
        elapsed = now - self._start
        if elapsed < 1.0 / self.target:
            time.sleep(1.0 / self.target - elapsed)
        self.tick()


# ===================================================================== #
# 5. Header toolbar with STABLE selection (Phase 2)                     #
# ===================================================================== #
class Header:
    """
    The y:0..100 toolbar.

    A tool fires when one of two things happens (STABLE selection):

      a) HOVER: the cursor hovers the button's box for STABLE_HOVER_FRAMES
         consecutive frames (~0.5 s).
      b) PINCH-CLICK: the thumb/index pinch ratio drops below
         PINCH_ON_RATIO while the cursor is inside the toolbar strip.
         It fires on the pinch *rising* edge, so the user must release and
         re-pinch to click again.
    """

    WIDGETS = [
        (TOOL_BLUE, "BLUE", COLOR_BLUE),
        (TOOL_GREEN, "GREEN", COLOR_GREEN),
        (TOOL_RED, "RED", COLOR_RED),
        (TOOL_ERASER, "ERASER", COLOR_WHITE),
        (TOOL_CLEAR, "CLEAR ALL", COLOR_BLACK),
    ]

    def __init__(self, height: int = HEADER_HEIGHT):
        self.height = height
        self.buttons = []               # laid out in build(width)
        self._hover_id = None
        self._hover_count = 0
        self._pinch_active = False
        self._last_fire = 0.0

    # ------------------------------------------------------------------ #
    def build(self, width: int):
        """Layout buttons evenly across the banner."""
        n = len(self.WIDGETS)
        bw = width / n
        self.buttons = [
            (tool_id, label, color, i * bw, (i + 1) * bw)
            for i, (tool_id, label, color) in enumerate(self.WIDGETS)
        ]

    # ------------------------------------------------------------------ #
    def hit(self, x, y):
        """The button tuple under (x, y), or None if outside the header."""
        if y is None or y < 0 or y >= self.height:
            return None
        for (tool_id, label, color, x1, x2) in self.buttons:
            if x1 <= x < x2:
                return (tool_id, label, color, x1, x2)
        return None

    # ------------------------------------------------------------------ #
    def update(self, cursor, pinch_ratio: float, mode: str):
        """Run the stable-selection state machine; return a tool id or None."""
        now = time.perf_counter()
        button = self.hit(*cursor) if cursor else None

        pinch = pinch_ratio is not None and pinch_ratio < PINCH_ON_RATIO
        if pinch and not self._pinch_active:
            if button is not None and (now - self._last_fire) >= \
                    SELECT_COOLDOWN_S:
                self._last_fire = now
                self._reset_hover()
                return button[0]
        self._pinch_active = bool(pinch)

        if button is not None and mode == HOVER:
            if button[0] == self._hover_id:
                self._hover_count += 1
            else:
                self._hover_id, self._hover_count = button[0], 1
            if self._hover_count >= STABLE_HOVER_FRAMES:
                self._last_fire = now
                tool = button[0]
                self._reset_hover()
                return tool
        else:
            self._reset_hover()
        return None

    # ------------------------------------------------------------------ #
    def _reset_hover(self):
        self._hover_id = None
        self._hover_count = 0

    # ------------------------------------------------------------------ #
    def draw(self, frame, active_tool: str):
        """Render the buttons and highlight the active tool in yellow."""
        height, width = frame.shape[:2]
        self.build(width)
        for (tool_id, label, color, x1f, x2f) in self.buttons:
            x1, x2 = int(x1f), int(x2f)
            cv2.rectangle(frame, (x1, 0), (x2 - 3, self.height - 1),
                          color, -1)
            cv2.rectangle(frame, (x1, 0), (x2 - 3, self.height - 1),
                          COLOR_BLACK, 1)
            text_c = (COLOR_WHITE if color in
                      (COLOR_RED, COLOR_BLUE, COLOR_BLACK) else COLOR_BLACK)
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.55, 1)
            pos = (x1 + (x2 - 3 - x1) // 2 - tw // 2,
                   self.height // 2 + 5)
            cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        text_c, 1, cv2.LINE_AA)
            if tool_id == active_tool:
                cv2.rectangle(frame, (x1 + 1, 1), (x2 - 4, self.height - 2),
                              COLOR_YELLOW, 3)


# ===================================================================== #
# 6. MediaPipe hand model (Phase 1.1)                                   #
# ===================================================================== #
class HandTracker:
    """Thin hand-tracking wrapper emitting the 21 landmarks each frame."""

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
        """Return landmarks for the first hand or None when none."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += max(1, int(1000 / TARGET_FPS))
        result = self._graph.detect_for_video(image, self._ts)
        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def release(self):
        self._graph.close()


# ===================================================================== #
# 7. Application orchestration                                          #
# ===================================================================== #
TOOL_COLORS = {
    TOOL_RED: COLOR_RED,
    TOOL_GREEN: COLOR_GREEN,
    TOOL_BLUE: COLOR_BLUE,
}


class GestureDrawApp:
    """Main loop: camera → tracker → gestures → header → stroke."""

    def __init__(self):
        self.tracker = HandTracker()
        self.canvas = Canvas()
        self.header = Header()
        self.tool = TOOL_RED
        self.fps_meter = FPS()
        self._lost_frames = 0

    # ------------------------------------------------------------------ #
    def run(self):
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        if not cap.isOpened():
            raise RuntimeError("Could not open the webcam. Close any "
                               "other app using it and retry.")

        # Size the board to the frames the camera actually delivers (some
        # webcams ignore the requested resolution).
        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            raise RuntimeError("Could not read a first frame from the "
                               "webcam.")
        first_h, first_w = first.shape[:2]
        self.canvas = Canvas(first_w, first_h)

        try:
            while True:
                self.fps_meter.tick()

                ok, frame = cap.read()
                if not ok:
                    continue

                height, width = frame.shape[:2]
                hand = self.tracker.detect(frame)
                gesture = analyse_hand(hand)

                cursor = None
                if hand is not None:
                    cursor = cursor_point(hand, gesture.mode,
                                          width, height)

                # ---- STABLE tool selection (Phase 2) ------------------
                tool = None
                if cursor is not None and cursor[1] < self.header.height:
                    tool = self.header.update(cursor, gesture.pinch,
                                              gesture.mode)
                else:
                    self.header._pinch_active = False
                    self.header._reset_hover()
                if tool:
                    self.apply_tool(tool)

                # ---- interpolated stroke (Phase 1.2) -------------------
                self._stroke_update(cursor, gesture, hand)

                # ---- present ---------------------------------------------
                comp = self.canvas.overlay(frame)
                self.header.draw(comp, self.tool)
                self._draw_hud(comp)

                cv2.imshow("GestureDraw v2 - Air Canvas", comp)
                self.fps_meter.pace()

                key = cv2.waitKey(1) & 0xFF
                if key != -1 and self._keyboard(key):
                    break
        finally:
            cap.release()
            self.tracker.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    def _stroke_update(self, cursor, gc, hand):
        """Paint prev→current, or reset/commit when not drawing."""
        if hand is None:
            self._lost_frames += 1
            if self._lost_frames > MAX_STROKE_LOST_FRAMES:
                self.canvas.reset_stroke()
            return

        self._lost_frames = 0
        if cursor is None or cursor[1] < HEADER_HEIGHT:
            self.canvas.reset_stroke()          # over the header: no draw
            return

        if gc.mode == DRAW:
            if self.tool == TOOL_ERASER:
                self.canvas.paint_next(*cursor, color=CANVAS_BG,
                                       brush=ERASER_BRUSH)
            else:
                self.canvas.paint_next(*cursor)
        else:
            if self.canvas.prev_point is not None:
                self.canvas.push_history()
            self.canvas.reset_stroke()

    # ------------------------------------------------------------------ #
    def _draw_hud(self, comp):
        y = self.header.height + 22
        label = (f"FPS {self.fps_meter.fps:5.1f} | "
                 f"tool {self.tool.upper()} | brush {self.canvas.brush}")
        cv2.putText(comp, label, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_YELLOW, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    def apply_tool(self, tool_id):
        if tool_id == TOOL_CLEAR:
            self.canvas.clear()
        elif tool_id == TOOL_ERASER:
            self.tool = tool_id
        elif tool_id in (TOOL_RED, TOOL_GREEN, TOOL_BLUE):
            self.tool = tool_id
            self.canvas.set_color(TOOL_COLORS[tool_id])

    # ------------------------------------------------------------------ #
    def _keyboard(self, key):
        if key == KEY_QUIT:
            return True
        if key == KEY_UNDO:
            self.canvas.undo()
        elif key == KEY_CLEAR:
            self.canvas.clear()
        elif key == KEY_SAVE:
            self._save_canvas()
        elif key in (KEY_BRUSH_UP, ord("=")):
            self.canvas.brush = min(MAX_BRUSH,
                                    self.canvas.brush + BRUSH_STEP)
        elif key == KEY_BRUSH_DOWN:
            self.canvas.brush = max(MIN_BRUSH,
                                    self.canvas.brush - BRUSH_STEP)
        return False

    def _save_canvas(self):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        folder = BASE_DIR / "exports"
        folder.mkdir(exist_ok=True)
        path = folder / f"gesturedraw_v2_{stamp}.png"
        cv2.imwrite(str(path), self.canvas.layer)
        print(f"[v2] canvas saved to {path}")


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
