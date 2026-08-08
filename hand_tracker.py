"""
PHASE 2 (updated): Hand Tracking & Gesture Recognition Engine
==============================================================
Finger classification + gesture mapping. Rebuilt on MediaPipe **Tasks**
HandLandmarker (the legacy `mp.solutions` API was removed in MediaPipe
1.0.0) and extended with gesture debouncing to fix dropped strokes.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Responsibilities
----------------
* Run the MediaPipe HandLandmarker model over each BGR frame (Tasks API).
* Determine which fingers are raised and map them to gestures.
* Debounce / stabilise gesture output across consecutive frames so one
  noisy frame doesn't interrupt an in-progress stroke (Phase 1).

Landmark indices used (21 per hand):
  4 thumb tip | 8 index tip | 12 middle tip | 16 ring tip | 20 pinky tip
  6 index pip | 10 middle pip   (finger raised test)
"""

from enum import IntEnum

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


class Gesture(IntEnum):
    """Every gesture the application understands."""
    NONE = 0    # no relevant action, just hover
    DRAW = 1    # index raised -> start/continue drawing
    SELECT = 2  # index + middle raised -> select / move cursor
    CLEAR = 3   # all fingers raised -> wipe the canvas


class HandTracker:
    """Wraps the MediaPipe Tasks HandLandmarker and classifies gestures."""

    def __init__(self, model_path=config.MODEL_PATH,
                 max_hands=config.MAX_HANDS,
                 detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                 tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                 presence_confidence=config.MIN_PRESENCE_CONFIDENCE):
        """
        Initialise the MediaPipe Tasks model.

        Args:
            model_path: Path to the hand_landmarker.task model file.
            max_hands:  Max simultaneous tracked hands.
            detection_confidence: Min confidence to detect a hand.
            tracking_confidence:  Min confidence to keep tracking.
            presence_confidence:  Min confidence a detected hand exists.
        """
        try:
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=presence_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self._hands = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MediaPipe model from {model_path}. "
                "Download hand_landmarker.task and keep it next to "
                f"air_canvas.py. Details: {exc}") from exc

        # Gesture debouncer used by :meth:`classify_stable`.
        self._stabilizer = GestureStabilizer()

        # Monotonically-increasing video timestamp (ms) for RunningMode.VIDEO.
        self._video_ts_ms = 0

    # ------------------------------------------------------------------ #
    # Tracking API                                                       #
    # ------------------------------------------------------------------ #

    def get_hand_landmarks(self, frame):
        """
        Feed one BGR frame through the model (Tasks VIDEO mode).

        VIDEO mode keeps tracking from the previous frame, so it is much
        faster than re-detecting landmarks every frame and produces
        smoother, less jittery coordinates (fewer false jumps).

        Args:
            frame: The (mirrored) BGR frame.

        Returns:
            list[list[lm]]: per-hand lists of 21 landmark objects with
            `.x/.y` (normalised), or an empty list when no hand.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._video_ts_ms += 33            # ~30fps stride keeps timestamps monotonic
        result = self._hands.detect_for_video(image, self._video_ts_ms)
        return result.hand_landmarks or []

    @staticmethod
    def fingers_up(hand_landmarks):
        """
        Classify each of the 5 fingers as raised (1) or folded (0).

        A finger is 'up' when its tip is clearly above its PIP joint (its
        vertical coordinate), i.e. by at least ``FINGER_UP_MARGIN`` so a
        small landmark jitter doesn't flip the reading. Works on *any*
        object exposing .x/.y, which makes this trivially unit-testable.

        Args:
            hand_landmarks: list of 21 objects with .x/.y (0..1).

        Returns:
            list[int]: [thumb, index, middle, ring, pinky] of 1/0 values.
        """
        fingers = []

        # Thumb (landmark 4) bends sideways, not up/down. For a mirrored
        # image the thumb reads "out" when tip.x is clearly left of its PIP.
        if hand_landmarks[4].x < hand_landmarks[3].x - config.FINGER_UP_MARGIN:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers 2..5: tip y below PIP y (with margin) means extended.
        tips_pips = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip, pip in tips_pips:
            raised = (hand_landmarks[tip].y
                      < hand_landmarks[pip].y - config.FINGER_UP_MARGIN)
            fingers.append(1 if raised else 0)

        return fingers

    @staticmethod
    def index_tip_pixels(hand_landmarks, frame_width, frame_height):
        """Pixel (x, y) of the index finger tip (landmark 8)."""
        lm = hand_landmarks[8]
        return int(lm.x * frame_width), int(lm.y * frame_height)

    # ------------------------------------------------------------------ #
    # Gesture mapping                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify(fingers):
        """
        Turn a raw [thumb, index, middle, ring, pinky] list into a Gesture.

        Args:
            fingers: Output of :meth:`fingers_up` (5 x 1/0 entries).

        Returns:
            Gesture value.
        """
        index, middle = fingers[1], fingers[2]

        # All five raised -> CLEAR (utility gesture, fastest draw).
        if all(f == 1 for f in fingers):
            return Gesture.CLEAR

        # Index + middle up -> SELECT.
        if index == 1 and middle == 1:
            return Gesture.SELECT

        # Index only up -> DRAW.
        if index == 1 and middle == 0:
            return Gesture.DRAW

        # Anything else -> NONE (hover).
        return Gesture.NONE

    # ------------------------------------------------------------------ #
    # Gesture stabiliser (Phase 1)                                       #
    # ------------------------------------------------------------------ #

    def classify_stable(self, hand_landmarks):
        """
        Debounce gesture detection to make strokes robust to noise.

        The raw gesture must remain the same for `GESTURE_STABLE_FRAMES`
        consecutive frames before the app acts on it. Until then the
        *previous* stable gesture is returned, so a single spurious frame
        no longer interrupts an active stroke.

        Args:
            hand_landmarks: list of 21 landmarks.

        Returns:
            Gesture: the currently *stable* gesture.
        """
        raw = self.classify(self.fingers_up(hand_landmarks))
        return self._stabilizer.update(raw)

    def release(self):
        """Free the model resources."""
        self._hands.close()


class GestureStabilizer:
    """
    Debounces gesture input with asymmetric hysteresis.

    Two separate thresholds make strokes feel both *responsive* and
    *continuous*:

    * Enter: a new gesture needs only ``GESTURE_STABLE_FRAMES`` consecutive
      frames to become active, so raising your finger starts drawing almost
      immediately (no start-of-stroke lag).
    * Exit: once a gesture is active it is kept for at least
      ``GESTURE_EXIT_FRAMES`` before a *different* gesture can replace it.
      A noisy 1-3 frame flicker therefore no longer cuts an in-progress
      stroke in the middle.
    """

    def __init__(self, stable_frames=None, exit_frames=None):
        """
        Args:
            stable_frames: frames required to *enter* a gesture. None uses
                the config default.
            exit_frames: frames a different gesture must persist to *leave*
                the active gesture. None uses the config default.
        """
        self.stable_frames = (stable_frames
                              if stable_frames is not None
                              else config.GESTURE_STABLE_FRAMES)
        self.exit_frames = (exit_frames
                            if exit_frames is not None
                            else config.GESTURE_EXIT_FRAMES)

        self._candidate = None          # raw gesture currently being observed
        self._count = 0                 # consecutive frames for candidate
        self._active = Gesture.NONE     # last confirmed stable gesture

    def reset(self):
        """Force the stabiliser back to NONE (e.g. hand left the frame)."""
        self._candidate = None
        self._count = 0
        self._active = Gesture.NONE

    def update(self, raw):
        """
        Feed one raw gesture, return the stable gesture.

        While a gesture is active, the *same* raw gesture keeps it alive.
        A different raw gesture must persist for ``exit_frames`` before it
        replaces the active one, so one-off noise is ignored.

        Args:
            raw: Gesture value from this frame.

        Returns:
            Gesture: the currently-confirmed gesture.
        """
        if raw == self._active:
            # Still the confirmed gesture: reset the alternate counter.
            self._candidate = None
            self._count = 0
            return self._active

        if raw == self._candidate:
            self._count += 1
        else:
            self._candidate = raw
            self._count = 1

        # Enter a gesture fast, but hold onto an active one much longer.
        threshold = (self.stable_frames
                     if self._active == Gesture.NONE
                     else self.exit_frames)
        if self._count >= threshold:
            self._active = raw
            self._candidate = None
            self._count = 0
        return self._active


def classify_passive(fingers):
    """Small wrapper so external callers can reuse the classifier."""
    return HandTracker.classify(fingers)
