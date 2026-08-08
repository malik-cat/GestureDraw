"""
GestureDraw - Air Canvas (shapes edition)
=========================================
A full-screen webcam whiteboard.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Gestures (work across the whole frame, not just one spot):
  SELECTION : index + middle raised      -> move cursor / click the palette
  DRAWING   : index raised only          -> draw freehand OR drag a shape
  NONE      : any other hand shape       -> hold / commit the current shape

Palette (two rows along the top):
  Row 1 : BLUE  GREEN  RED  YELLOW  ERASER  CLEAR
  Row 2 : LINE  RECT  CIRCLE  TRIANGLE  STAR  DRAW

Shapes:
  1. Point at a shape tool in Row 2 with a SELECT gesture.
  2. Do a DRAW gesture - the first fingertip point anchors the shape.
  3. Drag to size the preview (drawn in yellow on the feed).
  4. Release the finger (other gesture / hand out of frame) to commit
     the shape to the canvas.

Quit:  press 'q'
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------- #
# Model setup                                                           #
# --------------------------------------------------------------------- #
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    # Permissive confidence => the hand is tracked even when it is far
    # from the centre or partially close to the frame edges.
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = vision.HandLandmarker.create_from_options(options)

# --------------------------------------------------------------------- #
# Palette definitions (label, BGR colour, tool type, shape kind)        #
# --------------------------------------------------------------------- #
ROW_1 = [
    ("BLUE",     (255, 0, 0),     "colour"),
    ("GREEN",    (0, 255, 0),     "colour"),
    ("RED",      (0, 0, 255),     "colour"),
    ("YELLOW",   (0, 255, 255),   "colour"),
    ("ERASER",   (255, 255, 255), "eraser"),
    ("CLEAR",    (0, 0, 0),       "clear"),
]
ROW_2 = [
    ("LINE",     (255, 255, 255), "shape",   "line"),
    ("RECT",     (200, 200, 0),   "shape",   "rect"),
    ("CIRCLE",   (0, 200, 200),   "shape",   "circle"),
    ("TRI",      (0, 170, 255),   "shape",   "triangle"),
    ("STAR",     (255, 0, 255),   "shape",   "star"),
    ("DRAW",     (0, 255, 255),   "shape",   "free"),
]

ROW_HEIGHT = 80            # px per palette row
HEADER_HEIGHT = ROW_HEIGHT * 2

BRUSH_THICKNESS = 4
ERASER_THICKNESS = 50

# --------------------------------------------------------------------- #
# Tool state (imported from the palette)                                #
# --------------------------------------------------------------------- #
tool_mode = "colour"        # "colour" | "eraser" | "shape"
shape_kind = "free"         # line/rect/circle/triangle/star/free
pen_colour = (255, 0, 0)    # current drawing colour (BGR)

# --------------------------------------------------------------------- #
# Webcam + drawing canvas                                               #
# --------------------------------------------------------------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Could not open the webcam. Close other camera apps.")

canvas = None               # persistent drawing layer
prev_pt = (0, 0)            # previous fingertip (line continuity)

# Shape preview state.
shape_start = None          # anchor point while dragging a shape
shape_end = None            # current fingertip while dragging
drawing_shape = False


# --------------------------------------------------------------------- #
# Shape rendering helpers                                               #
# --------------------------------------------------------------------- #
def _dist(a, b):
    return int(np.hypot(b[0] - a[0], b[1] - a[1]))


def _draw_line(on, a, b, colour, thick):
    cv2.line(on, a, b, colour, thick, cv2.LINE_AA)


def _draw_rect(on, a, b, colour, thick):
    cv2.rectangle(on, a, b, colour, thick, cv2.LINE_AA)


def _draw_circle(on, a, b, colour, thick):
    r = max(4, _dist(a, b) // 2)
    cv2.circle(on, a, r, colour, thick, cv2.LINE_AA)


def _draw_triangle(on, a, b, colour, thick):
    """Triangle over the base segment (a, b)."""
    apex = ((a[0] + b[0]) // 2, a[1])
    pts = np.array([a, b, apex], np.int32).reshape(-1, 1, 2)
    cv2.polylines(on, [pts], True, colour, thick, cv2.LINE_AA)


def _draw_star(on, a, b, colour, thick):
    """5-pointed star centred on a, radius = distance(a, b)."""
    outer = max(8, _dist(a, b))
    inner = outer * 0.5
    pts = []
    for i in range(10):
        ang = np.deg2rad(-90 + i * 36)
        r = outer if i % 2 == 0 else inner
        pts.append((int(a[0] + r * np.cos(ang)), int(a[1] + r * np.sin(ang))))
    poly = np.array(pts, np.int32).reshape(-1, 1, 2)
    cv2.polylines(on, [poly], True, colour, thick, cv2.LINE_AA)


SHAPE_DRAWERS = {
    "line": _draw_line,
    "rect": _draw_rect,
    "circle": _draw_circle,
    "triangle": _draw_triangle,
    "star": _draw_star,
}


# --------------------------------------------------------------------- #
# Shape state helpers                                                   #
# --------------------------------------------------------------------- #
def _stroke_colour():
    """Return the colour used to stamp the current tool on the canvas."""
    return (0, 0, 0) if tool_mode == "eraser" else pen_colour


def _stroke_thickness():
    return ERASER_THICKNESS if tool_mode == "eraser" else BRUSH_THICKNESS


def commit_shape():
    """Stamp the dragged shape onto the persistent canvas."""
    global drawing_shape, shape_start, shape_end
    if drawing_shape and shape_start is not None and shape_end is not None:
        drawer = SHAPE_DRAWERS.get(shape_kind)
        if drawer is not None:
            drawer(canvas, shape_start, shape_end,
                   _stroke_colour(), _stroke_thickness())
    drawing_shape = False
    shape_start = None
    shape_end = None


def reset_shape():
    global drawing_shape, shape_start, shape_end
    drawing_shape = False
    shape_start = None
    shape_end = None


# --------------------------------------------------------------------- #
# Palette paint & hit testing                                           #
# --------------------------------------------------------------------- #
def paint_header(frame, w):
    box_w = w // 6
    for row, buttons in enumerate((ROW_1, ROW_2)):
        y0 = row * ROW_HEIGHT
        for i, entry in enumerate(buttons):
            label, colour = entry[0], entry[1]
            x1 = i * box_w
            x2 = x1 + box_w
            cv2.rectangle(frame, (x1, y0), (x2 - 2, y0 + ROW_HEIGHT - 2),
                          colour, -1)
            text_colour = (0, 0, 0) if colour == (255, 255, 255) else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 10, y0 + ROW_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_colour, 2,
                        cv2.LINE_AA)
    # Hint bar under the header.
    cv2.putText(frame, "index+middle=select | index=draw | release finger to commit",
                (10, HEADER_HEIGHT + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)


def pick_from_header(x, y, w):
    """Handle a palette click. Returns True when the click fell inside."""
    global tool_mode, shape_kind, pen_colour
    if y >= HEADER_HEIGHT:
        return False

    box_w = w // 6
    row = y // ROW_HEIGHT
    col = min(x // box_w, 5)

    if row == 0:
        label, colour, role = ROW_1[col]
        if role == "colour":
            pen_colour = colour
            tool_mode = "colour"
        elif role == "eraser":
            tool_mode = "eraser"
        else:                       # clear
            if canvas is not None:
                canvas.fill(0)
            reset_shape()
    else:
        label, colour, role, shape = ROW_2[col]
        if shape == "free":
            tool_mode = "colour"
        else:
            tool_mode = "shape"
        shape_kind = shape
        reset_shape()
    return True


# --------------------------------------------------------------------- #
# Main loop                                                             #
# --------------------------------------------------------------------- #
try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # mirror feed (selfie view)
        h, w = frame.shape[:2]

        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # ---------------- hand detection ------------------------------ #
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hands.detect(mp_image)

        # ---------------- palette (always visible) --------------------- #
        paint_header(frame, w)

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                lm = hand
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)
                mx, my = int(lm[12].x * w), int(lm[12].y * h)

                index_up = iy < int(lm[6].y * h)
                middle_up = my < int(lm[10].y * h)

                # ----- SELECTION: move cursor + click palette --------- #
                if index_up and middle_up:
                    prev_pt = (ix, iy)
                    pick_from_header(ix, iy, w)
                    reset_shape()

                # ----- DRAWING: index up only -------------------------- #
                elif index_up and not middle_up:
                    if tool_mode == "shape" and shape_kind != "free":
                        # Shape drag: anchor once, then update the tail.
                        if not drawing_shape:
                            drawing_shape = True
                            shape_start = (ix, iy)
                        shape_end = (ix, iy)
                        prev_pt = (ix, iy)

                    else:
                        # Free-hand stroke (pen or eraser).
                        if prev_pt == (0, 0):
                            prev_pt = (ix, iy)
                        colour = _stroke_colour()
                        thick = _stroke_thickness()
                        cv2.line(canvas, prev_pt, (ix, iy),
                                 colour, thick, cv2.LINE_AA)
                        prev_pt = (ix, iy)

                # ----- NONE: hold / commit ----------------------------- #
                else:
                    prev_pt = (0, 0)
                    commit_shape()

        else:
            # Hand out of view - commit any partially-drawn shape.
            prev_pt = (0, 0)
            commit_shape()

        # ---------------- live shape preview -------------------------- #
        if drawing_shape and shape_start is not None and shape_end is not None:
            drawer = SHAPE_DRAWERS.get(shape_kind)
            if drawer is not None:
                drawer(frame, shape_start, shape_end, (0, 255, 255), 2)

        # ---------------- overlay canvas on video ---------------------- #
        grey = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(grey, 50, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inv)
        frame = cv2.bitwise_or(frame, canvas)

        cv2.imshow("Air Canvas", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

finally:
    cap.release()
    hands.close()
    cv2.destroyAllWindows()