"""
PHASE 3 (updated): Dual-Layer Drawing Canvas, Header Palette & Sidebar
======================================================================
This module owns all drawing state plus the two interactive UI areas.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Design
------
* The whiteboard is a **standalone canvas matrix**, same size as the
  video frame, so every pixel on screen is drawable (Phase 2 fix).
* UI panels (top palette + left sidebar) are drawn *on top of* the feed.
  Strokes are suppressed only while the fingertip is inside a UI hit-zone
  (brief Phase 2 decision (a)); everywhere else is drawable edge-to-edge.
* Stroke continuity fixes (Phase 1):
    - a lone point (no previous) draws a small dot, not nothing;
    - a fingertip jump larger than MAX_STROKE_JUMP starts a new stroke.
* Undo/redo history for Phase 4.
"""

import cv2
import numpy as np

import config


class Canvas:
    """
    Stores the persistent drawing matrix, brush settings and the undo/redo
    history.
    """

    def __init__(self, width, height, max_jump=None):
        """
        Args:
            width:   Drawing width  (px) - same as the video frame.
            height:  Drawing height (px) - same as the video frame.
            max_jump: px threshold for a "jump" -> new stroke. None uses
                the config default.
        """
        self.width = width
        self.height = height
        self.max_jump = max_jump or config.MAX_STROKE_JUMP

        # Persistent white drawing layer (same full-frame size).
        self.layer = np.full((height, width, 3), config.CANVAS_BG,
                             dtype=np.uint8)

        # Stroke settings.
        self.current_color = config.COLOR_RED
        self.brush_size = config.DEFAULT_BRUSH_SIZE
        self.eraser_size = config.ERASER_SIZE

        # Stroke endpoints continuity (Phase 1/2).
        self.last_point = None

        # Undo / redo history.
        self._undo_stack: list[np.ndarray] = []
        self._redo_stack: list[np.ndarray] = []
        self._max_history = 25

    # ------------------------------------------------------------------ #
    # Colour / brush                                                     #
    # ------------------------------------------------------------------ #

    def set_color(self, color):
        """Set the drawing colour to `color` (BGR tuple)."""
        self.current_color = color

    def set_brush(self, step):
        """
        Change brush size by `step` (px), clamped to the config bounds.

        Args:
            step: positive to grow, negative to shrink.
        """
        self.brush_size = min(
            config.MAX_BRUSH_SIZE,
            max(config.MIN_BRUSH_SIZE, self.brush_size + step))

    # ------------------------------------------------------------------ #
    # Stroke primitives (Phase 1 fixes)                                  #
    # ------------------------------------------------------------------ #

    def reset_pointer(self):
        """Forget the previous fingertip - start a new stroke."""
        self.last_point = None

    def stroke(self, x, y, is_eraser=False):
        """
        Draw toward `(x, y)` on the layer, keeping strokes continuous.

        Phase 1 fixes:
          1. If there is no previous point, draw a small dot first instead
             of silently skipping the frame (no dropped first point).
          2. If `(x, y)` moved farther than `self.max_jump` px from the
             last point, it's a tracking glitch - start a new stroke.
          3. Otherwise, connect the previous point to the current one.

        Args:
            x:        horizontal pixel coordinate.
            y:        vertical pixel coordinate.
            is_eraser: True erases (paints background colour) instead of
                drawing with the current colour.
        """
        color = (config.CANVAS_BG if is_eraser else self.current_color)
        thickness = (self.eraser_size if is_eraser else self.brush_size)

        point = (x, y)

        # Case 1: fresh stroke start == dot only.
        if self.last_point is None:
            cv2.circle(self.layer, point, max(1, thickness // 2),
                       color, thickness=-1)
            self.last_point = point
            return

        # Case 2: max-jump guard - treat an enormous move as a glitch.
        prev = self.last_point
        if abs(point[0] - prev[0]) > self.max_jump or \
           abs(point[1] - prev[1]) > self.max_jump:
            self.last_point = point       # restart stroke at the new spot
            return

        # Case 3: normal connect.
        cv2.line(self.layer, prev, point, color, thickness, cv2.LINE_AA)
        self.last_point = point

    # ------------------------------------------------------------------ #
    # Canvas-wide operations                                             #
    # ------------------------------------------------------------------ #

    def clear(self):
        """Save history, then empty the whiteboard."""
        self._push_history()
        self.layer.fill(config.CANVAS_BG)
        self.last_point = None

    # ------------------------------------------------------------------ #
    # Undo / redo (Phase 4)                                               #
    # ------------------------------------------------------------------ #

    def _push_history(self):
        """Save a snapshot of the current layer to the undo stack."""
        self._undo_stack.append(self.layer.copy())
        self._redo_stack.clear()
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)

    def push_stroke_history(self):
        """Push the current layer to the history (called after a stroke)."""
        self._push_history()

    def undo(self):
        """Undo the last stroke/clear. Returns True if anything changed."""
        if not self._undo_stack:
            return False
        self._redo_stack.append(self.layer.copy())
        self.layer = self._undo_stack.pop().copy()
        self.last_point = None
        return True

    def redo(self):
        """Redo a previously undone operation."""
        if not self._redo_stack:
            return False
        self._undo_stack.append(self.layer.copy())
        self.layer = self._redo_stack.pop().copy()
        self.last_point = None
        return True

    # ------------------------------------------------------------------ #
    # Rendering / merging with the video feed                            #
    # ------------------------------------------------------------------ #

    def overlay(self, frame):
        """
        Composite the drawing layer over the video frame.

        Pixels other than background (CANVAS_BG) fully replace the video
        so strokes stay opaque; since the layer is full-frame, the whole
        screen is drawable (Phase 2).
        """
        composite = frame.copy()
        mask = np.any(self.layer != config.CANVAS_BG, axis=-1)
        composite[mask] = self.layer[mask]
        return composite


class UIButton:
    """Immutable interactive rectangle used by both header and sidebar."""

    __slots__ = ("tool", "label", "rect")

    def __init__(self, tool, label, rect):
        self.tool = tool          # tool id / action (str)
        self.label = label        # short display text
        self.rect = rect          # (x1, y1, x2, y2)

    def contains(self, x, y):
        """True when the point (x, y) is inside this button."""
        x1, y1, x2, y2 = self.rect
        return x1 <= x < x2 and y1 <= y < y2


class Palette:
    """
    Top header palette rendered over the video.

    Rendered on top of the composite; hit-tested like the sidebar. has no
    draw-suppression logic - strokes are only suppressed while the
    fingertip is inside a button (handled by the app using hit_test).
    """

    def __init__(self, items=None, header_height=None):
        self.items = items or config.PALETTE_ITEMS
        self.header_height = header_height or config.HEADER_HEIGHT
        self.buttons = []          # (re)built in draw()

    def _build_buttons(self, width):
        """Compute button rects for the current frame width."""
        self.buttons = []
        box_w = width // len(self.items)
        for i, (label, tool, color) in enumerate(self.items):
            rect = (i * box_w, 0, (i + 1) * box_w, self.header_height)
            self.buttons.append(UIButton(tool, label, rect))

    def draw(self, frame, active_tool):
        """Render the palette into the top rows of `frame`."""
        height, width = frame.shape[:2]
        self._build_buttons(width)

        for i, button in enumerate(self.buttons):
            x1, y1, x2, y2 = button.rect
            color = self.items[i][2]

            cv2.rectangle(frame, (x1, y1), (x2 - 2, y2 - 2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2 - 2, y2 - 2), (0, 0, 0), 1)

            if color == config.COLOR_WHITE:
                text_clr = (0, 0, 0)
            else:
                text_clr = (255, 255, 255)

            cv2.putText(frame, button.label,
                        (x1 + 8, y1 + (y2 - y1) // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_clr, 1,
                        cv2.LINE_AA)

            if button.tool == active_tool:
                cv2.rectangle(frame, (x1, y1), (x2 - 2, y2 - 2),
                              config.COLOR_YELLOW, 2)

    def hit_test(self, x, y):
        """Return the tool under (x, y) or None (requires a built palette)."""
        if y < 0 or y >= self.header_height:
            return None
        for button in self.buttons:
            if button.contains(x, y):
                return button.tool
        return None


class Sidebar:
    """
    Vertical tool/options panel on the left edge (Phase 3).

    Rendered on top of the composite; everything to its right remains
    drawable. Hit-testing mirrors :class:`Palette.hit_test`.
    """

    def __init__(self, items=None, width=None):
        self.items = items or config.SIDEBAR_ITEMS
        self.width = width or config.SIDEBAR_WIDTH
        self.buttons = []

    def _build_buttons(self, height):
        self.buttons = []
        item_h = max(28, height // len(self.items))
        for i, (label, tool, color) in enumerate(self.items):
            y1 = i * item_h
            y2 = y1 + item_h
            self.buttons.append(UIButton(tool, label,
                                         (0, y1, self.width, y2)))

    def draw(self, frame, active_tool):
        """Render the sidebar into the left column of `frame."""
        height, width = frame.shape[:2]
        self._build_buttons(height)

        # Panel background so text is readable against the video.
        cv2.rectangle(frame, (0, 0), (self.width - 2, height - 1),
                      config.COLOR_BG, -1)

        for i, button in enumerate(self.buttons):
            x1, y1, x2, y2 = button.rect
            color = self.items[i][2]

            cv2.rectangle(frame, (x1 + 2, y1 + 2),
                          (x2 - 2, y2 - 2), color, -1)

            # Text colour for dark-on-light labels.
            if color == config.COLOR_WHITE:
                text_clr = (255, 255, 255)
            else:
                text_clr = (0, 0, 0)

            cv2.putText(frame, button.label,
                        (x1 + 8, y1 + (y2 - y1) // 2 + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_clr, 1,
                        cv2.LINE_AA)

            # Highlight the active tool in yellow (SELECT feedback).
            if button.tool == active_tool:
                cv2.rectangle(frame, (x1 + 2, y1 + 2),
                              (x2 - 2, y2 - 2),
                              config.COLOR_YELLOW, 3)

    def hit_test(self, x, y):
        """Return the tool under (x, y) or None."""
        if x < 0 or x >= self.width:
            return None
        for button in self.buttons:
            if button.contains(x, y):
                return button.tool
        return None
