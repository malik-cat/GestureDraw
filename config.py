"""
Central configuration for GestureDraw.
======================================
All thresholds, dimensions, colours, tool ids and keyboard shortcuts live
here so the rest of the project has no magic numbers in code.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw
"""

import os
from pathlib import Path

# ------------------------------------------------------------------- #
# Paths                                                               #
# ------------------------------------------------------------------- #
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = PROJECT_DIR / "hand_landmarker.task"
EXPORTS_DIR = PROJECT_DIR / "exports"

# ------------------------------------------------------------------- #
# Camera                                                              #
# ------------------------------------------------------------------- #
CAMERA_SOURCE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ------------------------------------------------------------------- #
# MediaPipe                                                           #
# ------------------------------------------------------------------- #
MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ------------------------------------------------------------------- #
# Gesture / drawing behaviour (Phase 1)                               #
# ------------------------------------------------------------------- #
# A gesture must appear for this many consecutive frames before it becomes
# the active gesture (entry debounce). Kept small so drawing starts fast
# (no perceptible lag when you raise your finger).
GESTURE_STABLE_FRAMES = 2
# A gesture other than the currently-active one must persist for this many
# consecutive frames before it replaces the active gesture. This is the
# "release" threshold and is deliberately bigger than STABLE_FRAMES (hysteresis):
# transient jewellery (a 1-3 frame flicker of NONE/SELECT while drawing) no
# longer cuts an in-progress stroke mid-way.
GESTURE_EXIT_FRAMES = 5
# Maximum fingertip jump (px) between two frames. Beyond it the move is
# treated as a tracking glitch -> start a fresh stroke. Set well above any
# normal fast stroke (a 1280x720 frame at ~25fps) so quick strokes stay
# connected; only absurd teleports are re-anchored.
MAX_STROKE_JUMP = 450
# Normalised margin a finger tip must clear its PIP joint by before the
# finger counts as "raised". Prevents the tip/PIP jitters from flipping a
# gesture every frame (which used to cut strokes in the middle).
FINGER_UP_MARGIN = 0.03

# ------------------------------------------------------------------- #
# Layout                                                              #
# ------------------------------------------------------------------- #
HEADER_HEIGHT = 56              # top palette bar height (px)
SIDEBAR_WIDTH = 150             # left sidebar width (px)
SWATCH_TEXT_THRESHOLD = 80      # below this, label uses complementary colour

# ------------------------------------------------------------------- #
# Stroke / brush                                                      #
# ------------------------------------------------------------------- #
DEFAULT_BRUSH_SIZE = 8
MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 60
BRUSH_STEP = 2
ERASER_SIZE = 32

# ------------------------------------------------------------------- #
# Colours (BGR)                                                       #
# ------------------------------------------------------------------- #
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (128, 128, 128)
COLOR_BG = (40, 40, 60)         # sidebar / header panel background

# Canvas background colour (BGR). Non-white here means "drawn".
CANVAS_BG = COLOR_WHITE

# ------------------------------------------------------------------- #
# Tools                                                               #
# ------------------------------------------------------------------- #
TOOL_RED = "red"
TOOL_GREEN = "green"
TOOL_BLUE = "blue"
TOOL_ERASER = "eraser"
TOOL_CLEAR = "clear"

# Shape tools (Phase 2 of the v2 brief).
TOOL_LINE = "line"
TOOL_RECT = "rect"
TOOL_CIRCLE = "circle"
TOOL_TRIANGLE = "triangle"
TOOL_STAR = "star"
TOOL_DRAW = "draw"          # returns to free-hand drawing

# ------------------------------------------------------------------- #
# Sidebar tools/options (label, tool id or action, BGR colour)        #
# ------------------------------------------------------------------- #
SIDEBAR_ITEMS = [
    ("RED",    TOOL_RED,    COLOR_RED),
    ("GREEN",  TOOL_GREEN,  COLOR_GREEN),
    ("BLUE",   TOOL_BLUE,   COLOR_BLUE),
    ("ERASER", TOOL_ERASER, COLOR_WHITE),
    ("CLEAR",  TOOL_CLEAR,  COLOR_BLACK),
    ("LINE",   TOOL_LINE,   COLOR_GRAY),
    ("RECT",   TOOL_RECT,   COLOR_GRAY),
    ("CIRCLE", TOOL_CIRCLE, COLOR_GRAY),
    ("TRIANGLE", TOOL_TRIANGLE, COLOR_GRAY),
    ("STAR",   TOOL_STAR,   COLOR_GRAY),
    ("DRAW",   TOOL_DRAW,   COLOR_WHITE),
    ("UNDO",   "undo",      COLOR_GRAY),
    ("SAVE",   "save",      COLOR_GRAY),
    ("BRUSH+", "brush+",    COLOR_GRAY),
    ("BRUSH-", "brush-",    COLOR_GRAY),
]

# Top palette (Phase 2 decides (a): suppress strokes over UI, allow drawing
# everywhere else). Kept as a configurable list for future extension.
PALETTE_ITEMS = [
    ("RED",    TOOL_RED,    COLOR_RED),
    ("GREEN",  TOOL_GREEN,  COLOR_GREEN),
    ("BLUE",   TOOL_BLUE,   COLOR_BLUE),
    ("ERASER", TOOL_ERASER, COLOR_WHITE),
    ("CLEAR",  TOOL_CLEAR,  COLOR_BLACK),
    ("LINE",   TOOL_LINE,   COLOR_GRAY),
    ("RECT",   TOOL_RECT,   COLOR_GRAY),
    ("CIRCLE", TOOL_CIRCLE, COLOR_GRAY),
    ("TRIANGLE", TOOL_TRIANGLE, COLOR_GRAY),
    ("STAR",   TOOL_STAR,   COLOR_GRAY),
    ("DRAW",   TOOL_DRAW,   COLOR_WHITE),
]

# ------------------------------------------------------------------- #
# Keyboard shortcuts                                                  #
# ------------------------------------------------------------------- #
KEY_QUIT = "q"
KEY_CLEAR = "c"
KEY_UNDO = "u"
KEY_REDO = "y"
KEY_SAVE = "s"
KEY_BRUSH_UP = "+"
KEY_BRUSH_DOWN = "-"
KEY_ERASER = "e"

# ------------------------------------------------------------------- #
# On-screen feedback                                                  #
# ------------------------------------------------------------------- #
GESTURE_TEXT_COLOR = COLOR_YELLOW
GESTURE_TEXT_POS = (SIDEBAR_WIDTH + 12, HEADER_HEIGHT + 24)

RECORDING_INDICATOR = True      # shows "● REC" dot while camera is live
