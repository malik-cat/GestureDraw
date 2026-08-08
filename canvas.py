"""
PHASE 3: Dual-Layer Drawing Canvas & Header Palette
===================================================
This module owns the drawing state of the application.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Design
------
* The whiteboard is a **standalone canvas matrix** whose size matches the
  video frame. Strokes are stored there and never auto-reset.
* Each frame the whiteboard matrix is merged with the video feed so the
  lines persist (dual-layer rendering).
* At the top of the video frame we render the interactive **header palette**
  (Red / Green / Blue / Eraser / Clear) so the user can point at a swatch
  with a SELECT gesture to change the active tool.
"""

import cv2
import numpy as np

# --- Colours (BGR order as OpenCV expects them) ------------------------
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)

# --- Tool identifiers (used by Phase 4 to compare) ---------------------
TOOL_RED = "red"
TOOL_GREEN = "green"
TOOL_BLUE = "blue"
TOOL_ERASER = "eraser"
TOOL_CLEAR = "clear"

# --- Header metrics for the palette bar along the top of the feed ------
HEADER_HEIGHT = 40          # total height of the header in pixels


class Canvas:
    """
    Stores the persistent drawing matrix plus the stroke settings.

    The layer is a white (BGR 255) matrix that keeps whatever is painted
    on it. The drawing and the video stream are merged in `overlay()`.
    """

    def __init__(self, width, height):
        """
        Args:
            width:  Drawing width  (px) - same as the video frame.
            height: Drawing height (px) - same as the video frame.
        """
        self.width = width
        self.height = height

        # The persistent white layer; strokes accumulate here.
        self.layer = np.full((height, width, 3), COLOR_WHITE, dtype=np.uint8)

        # Current pen configuration.
        self.current_color = COLOR_GREEN
        self.brush_size = 8
        self.eraser_size = 28

        # Fingertip position from the previous frame so strokes are
        # continuous lines instead of disconnected dots.
        self.last_point = None

    # ------------------------------------------------------------------
    # Core drawing primitives (only touch `self.layer`)
    # ------------------------------------------------------------------

    def set_color(self, color):
        """Set the drawing colour to `color` (a BGR tuple)."""
        self.current_color = color

    def reset_pointer(self):
        """Forget the previous fingertip (begin a new stroke)."""
        self.last_point = None

    def stroke(self, x, y, is_eraser=False):
        """
        Draw a line toward `(x, y)` on the canvas layer.

        When `is_eraser` is True the "ink" is white so the stroke removes
        previously drawn paint from the whiteboard.
        """
        color = COLOR_WHITE if is_eraser else self.current_color
        thickness = self.eraser_size if is_eraser else self.brush_size

        if self.last_point is not None:
            cv2.line(self.layer, self.last_point, (x, y), color, thickness,
                     lineType=cv2.LINE_AA)
        self.last_point = (x, y)

    def clear(self):
        """Erase the entire whiteboard."""
        self.layer.fill(COLOR_WHITE)
        self.last_point = None

    # ------------------------------------------------------------------
    # Rendering / merging with the video feed (dual-layer composite)
    # ------------------------------------------------------------------

    def overlay(self, frame):
        """
        Composite the drawing layer over the video frame.

        Only "painted" pixels (anything that is not pure white) overwrite
        the video; the white parts stay transparent so the webcam feed
        shows through behind the strokes.

        Args:
            frame: Mirrored BGR video frame.

        Returns:
            ndarray: frame + drawing merged into one image.
        """
        overlay = frame.copy()
        painted = np.any(self.layer != COLOR_WHITE, axis=-1)
        overlay[painted] = self.layer[painted]
        return overlay


class Palette:
    """
    Draws the interactive header bar and hit-tests fingertip positions.

    The palette is rendered into the top `HEADER_HEIGHT` rows of the video
    frame, split into equal-width swatches left -> right.
    """

    # Ordered tool list -> defines swatch layout and labels.
    TOOLS = [
        (TOOL_RED, COLOR_RED, "RED"),
        (TOOL_GREEN, COLOR_GREEN, "GREEN"),
        (TOOL_BLUE, COLOR_BLUE, "BLUE"),
        (TOOL_ERASER, COLOR_WHITE, "ERASER"),
        (TOOL_CLEAR, COLOR_BLACK, "CLEAR"),
    ]

    def __init__(self, header_height=HEADER_HEIGHT):
        self.header_height = header_height

    def swatch_width(self, frame_width):
        """Width (px) of every swatch for a frame of `frame_width`."""
        return frame_width // len(self.TOOLS)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, frame, active_tool):
        """
        Render the colour swatches into the top rows of `frame`.

        Args:
            frame:       BGR frame being edited in place.
            active_tool: tool id (TOOL_RED, ...) to highlight in yellow.
        """
        height, width = frame.shape[:2]
        box_w = self.swatch_width(width)

        for i, (tool_id, color, label) in enumerate(self.TOOLS):
            x1 = i * box_w
            x2 = (i + 1) * box_w

            # Filled swatch block + subtle black border.
            cv2.rectangle(frame, (x1, 0), (x2 - 2, self.header_height - 2),
                          color, thickness=-1)
            cv2.rectangle(frame, (x1, 0), (x2 - 2, self.header_height - 2),
                          (0, 0, 0), thickness=1)

            # Draw the label text centered-ish in the swatch.
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, 2)[0]
            label_x = x1 + (box_w - label_size[0]) // 2
            label_y = self.header_height // 2 + label_size[1] // 2
            cv2.putText(frame, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0) if color == COLOR_WHITE else (255, 255, 255),
                        2, cv2.LINE_AA)

            # Yellow outline highlights the tool currently in use.
            if tool_id == active_tool:
                cv2.rectangle(frame, (x1, 0), (x2 - 2, self.header_height - 2),
                              COLOR_YELLOW, thickness=2)

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def hit_test(self, x, y, frame_width):
        """
        Identify which tool swatch sits under pixel `(x, y)`.

        Args:
            x:           horizontal pixel coordinate (0 is the left edge).
            y:           vertical pixel coordinate (0 is the header top).
            frame_width: total frame width used to compute swatch width.

        Returns:
            str:  tool identifier (TOOL_RED, TOOL_GREEN, ...) or None when
                  the point is outside the header bar.
        """
        if y < 0 or y >= self.header_height:
            return None

        box_w = self.swatch_width(frame_width)
        index = x // box_w

        if index < 0 or index >= len(self.TOOLS):
            return None

        return self.TOOLS[index][0]     # tool identifier