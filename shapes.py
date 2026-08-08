"""
PHASE 2 (v2 brief): Shape Engine
===================================
Anchor → drag → commit geometry for LINE / RECT / CIRCLE / TRIANGLE / STAR.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Design
------
* `ShapeTool` enumerates every selectable shape (plus NONE = free-draw).
* `ShapeEngine` owns the drag state: the anchor point is fixed on the first
  DRAW frame, the current point follows the fingertip, and commit() returns
  the triple so the app can bake it into the canvas layer.
* Geometry is a pure function of (tool, anchor, current) — it only draws into
  an existing BGR plane, so commits can target `canvas.layer` and previews
  can target the composite frame with different colors. This keeps the whole
  drag lifecycle unit-testable without a camera.
"""

import math
from enum import IntEnum

import cv2
import numpy as np


class ShapeTool(IntEnum):
    """Every selectable shape; NONE means free-hand drawing."""
    NONE = 0
    LINE = 1
    RECT = 2
    CIRCLE = 3
    TRIANGLE = 4
    STAR = 5


# ------------------------------------------------------------------ #
# Geometry                                                            #
# ------------------------------------------------------------------ #


def _clip(p, w, h):
    """Clamp a point into the frame so shapes stay on the canvas."""
    return min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1)


def _points_of_star(cx, cy, radius):
    """10 points of a 5-point star centred on (cx, cy)."""
    inner = max(radius * 0.45, 4)
    pts = []
    for i in range(10):
        r = radius if i % 2 == 0 else inner
        ang = math.pi / 2 + i * math.pi / 5.0
        pts.append((int(cx + r * math.cos(ang)),
                    int(cy + r * math.sin(ang))))
    return np.array(pts, dtype=np.int32)


def draw_shape(plane, tool, anchor, current, width, height,
               color=(0, 255, 255), thickness=2):
    """
    Draw the shape onto `plane`.

    Args:
        plane:      BGR image (canvas layer, or the composite for preview).
        tool:       ShapeTool member.
        anchor:     (x, y) fixed on the first drag frame.
        current:    (x, y) current drag point.
        width,height: bounding size for clamping.
        color:      BGR tuple.
        thickness:  outline thickness in px.
    """
    ax, ay = _clip(anchor, width, height)
    cx, cy = _clip(current, width, height)
    t = max(thickness, 1)

    if tool == ShapeTool.LINE:
        cv2.line(plane, (ax, ay), (cx, cy), color, t, cv2.LINE_AA)

    elif tool == ShapeTool.RECT:
        cv2.rectangle(plane, (min(ax, cx), min(ay, cy)),
                      (max(ax, cx), max(ay, cy)), color, t, cv2.LINE_AA)

    elif tool == ShapeTool.CIRCLE:
        radius = int(math.hypot(cx - ax, cy - ay))
        cv2.circle(plane, (ax, ay), radius, color, t, cv2.LINE_AA)

    elif tool == ShapeTool.TRIANGLE:
        x1, x2 = min(ax, cx), max(ax, cx)
        y1, y2 = min(ay, cy), max(ay, cy)
        pts = np.array([[(x1 + x2) // 2, y1], [x1, y2], [x2, y2]],
                       dtype=np.int32)
        cv2.polylines(plane, [pts], True, color, t, cv2.LINE_AA)

    elif tool == ShapeTool.STAR:
        radius = max(int(math.hypot(cx - ax, cy - ay)), 10)
        star = _points_of_star(ax, ay, radius)
        cv2.polylines(plane, [star], True, color, t, cv2.LINE_AA)


# ------------------------------------------------------------------ #
# ShapeEngine state machine                                          #
# ------------------------------------------------------------------ #


class ShapeEngine:
    """Anchor → drag → commit bookkeeping for one shape at a time."""

    def __init__(self):
        self.tool = ShapeTool.NONE   # active shape tool
        self.anchor = None           # (x, y): fixed on first drag frame
        self.current = None          # (x, y): live drag point

    # ------------------------------------------------------------------
    # State setters

    def select(self, tool):
        """Choose the shape tool (or ShapeTool.NONE for free drawing)."""
        self.tool = tool
        self.anchor = None
        self.current = None

    def begin(self, x, y):
        """First DRAW frame after selecting a shape -> anchor it."""
        self.anchor = (x, y)
        self.current = (x, y)

    def drag(self, x, y):
        """Subsequent DRAW frames -> move the preview point."""
        if self.anchor is None:
            self.anchor = (x, y)
        self.current = (x, y)

    # ------------------------------------------------------------------
    # Queries

    @property
    def shape_active(self):
        """True when a shape tool is chosen AND the drag has started."""
        return self.tool != ShapeTool.NONE and self.anchor is not None

    @property
    def tool_selected(self):
        """True when any shape tool is chosen (drag may not have started)."""
        return self.tool != ShapeTool.NONE

    # ------------------------------------------------------------------
    # Lifecycle

    def commit(self):
        """
        Finish the drag.

        Returns:
            (tool, anchor, current) triple, or None when there is no
            in-progress shape to commit. Resets drag state afterwards but
            keeps the selected shape tool so another shape can be drawn.
        """
        if self.anchor is None:
            return None
        result = (self.tool, self.anchor, self.current)
        self.anchor = None
        self.current = None
        return result

    def cancel(self):
        """Abort the drag without committing the shape."""
        self.anchor = None
        self.current = None

    def reset_drag(self):
        """Alias of cancel(); used when the hand leaves the frame."""
        self.cancel()
