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
# A gesture must stay stable for this many consecutive frames before the
# app acts on it (debouncing).
GESTURE_STABLE_FRAMES = 3
# Maximum fingertip jump (px) between two frames. Beyond it the move is
# treated as a tracking glitch -> start a fresh stroke.
MAX_STROKE_JUMP = 120

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

# ------------------------------------------------------------------- #
# Sidebar tools/options (label, tool id or action, BGR colour)        #
# ------------------------------------------------------------------- #
SIDEBAR_ITEMS = [
    ("RED",    TOOL_RED,    COLOR_RED),
    ("GREEN",  TOOL_GREEN,  COLOR_GREEN),
    ("BLUE",   TOOL_BLUE,   COLOR_BLUE),
    ("ERASER", TOOL_ERASER, COLOR_WHITE),
    ("CLEAR",  TOOL_CLEAR,  COLOR_BLACK),
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