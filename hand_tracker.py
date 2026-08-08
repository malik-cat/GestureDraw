"""
PHASE 2: Hand Tracking & Gesture Recognition Engine
====================================================
This module contains everything related to MediaPipe hand tracking and the
mapping of raw hand landmarks to high-level gestures.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Responsibilities
----------------
* Run the MediaPipe Hands model over each BGR frame.
* Return the pixel coordinates of the hand landmarks.
* Determine which fingers are raised/extended.
* Translate the finger state into one of our app specific gestures:
      Gesture.DRAW       -> index finger up only          (drawing mode)
      Gesture.SELECT     -> index + middle up             (menu / cursor mode)
      Gesture.CLEAR      -> all five fingers up         (clear the canvas)
      Gesture.NONE       -> any other combination       (do nothing)

MediaPipe Hand landmark indices (21 points per hand):
    0: wrist   4: thumb tip            8: index tip
    12: middle tip  16: ring tip       20: pinky tip
    6: index pip   10: middle pip      (used to measure finger raised)
"""

import cv2
import mediapipe as mp
from enum import IntEnum


class Gesture(IntEnum):
    """Every meaningful gesture the application understands."""
    NONE = 0    # no relevant action, just hover
    DRAW = 1    # index raised -> start/continue drawing
    SELECT = 2  # index + middle raised -> select colours / move cursor
    CLEAR = 3   # all fingers raised -> wipe the whiteboard


class HandTracker:
    """Wraps MediaPipe Hands and translates landmarks into gestures."""

    def __init__(self, max_hands=1,
                 detection_confidence=0.7,
                 tracking_confidence=0.5):
        """
        Initialise the MediaPipe Hands model.

        Args:
            max_hands: Cap on simultaneous tracked hands (we only need one).
            detection_confidence: Minimum confidence to start tracking.
            tracking_confidence:  Minimum confidence to keep tracking.
        """
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,          # video stream, not images
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ------------------------------------------------------------------ #
    # Tracking API
    # ------------------------------------------------------------------ #

    def get_hand_landmarks(self, frame):
        """
        Feed one BGR frame through the model.

        Args:
            frame: The (mirrored) BGR frame from Phase 1.

        Returns:
            A MediaPipe `results` object. Access .multi_hand_landmarks
            for a list of hands (None when no hand is present).
        """
        # MediaPipe expects RGB input, our camera gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._hands.process(rgb)

    @staticmethod
    def fingers_up(hand_landmarks):
        """
        Classify each of the 5 fingers as raised (1) or folded (0).

        Heuristic: a finger is 'up' when its tip is higher than the joint
        two landmarks below it. Both coordinates are normalised [0..1].

        Args:
            hand_landmarks: A list of 21 MediaPipe landmarks.

        Returns:
            list[int]: [thumb, index, middle, ring, pinky] of 1/0 values.
        """
        fingers = []

        # Thumb (landmark 4) is special: it bends sideways, not up/down.
        # For a mirrored right hand the thumb is "out" when tip.x < pip.x.
        if hand_landmarks[4].x < hand_landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # The other four fingers: tip (y) above the pip (y) means raised.
        tips_ips = [(8, 6), (12, 10), (16, 14), (20, 18)]
        for tip, pip in tips_ips:
            if hand_landmarks[tip].y < hand_landmarks[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    @staticmethod
    def index_tip_pixels(hand_landmarks, frame_width, frame_height):
        """
        Convert the normalized index-finger tip (landmark 8) into pixels.

        Returns:
            tuple[int, int] scaled into the frame coordinate space.
        """
        lm = hand_landmarks[8]
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        return x, y

    # ------------------------------------------------------------------ #
    # Gesture mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def classify(fingers):
        """
        Turn a raw [thumb, index, middle, ring, pinky] list into an
        app-level :class:`Gesture`.

        Args:
            fingers: Output of :meth:`fingers_up` (5 x 1/0 entries).

        Returns:
            Gesture value (NONE when the pattern is unrecognised).
        """
        index, middle = fingers[1], fingers[2]

        # Index + middle raised together -> SELECT / cursor mode.
        if index == 1 and middle == 1:
            return Gesture.SELECT

        # Only the index finger raised -> DRAW.
        if index == 1 and middle == 0:
            return Gesture.DRAW

        # All five fingers raised -> CLEAR (utility gesture).
        if all(f == 1 for f in fingers):
            return Gesture.CLEAR

        # Everything else is treated as "no gesture".
        return Gesture.NONE

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Overlay the visible hand skeleton on a frame for debugging.
        """
        self._mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self._mp_hands.HAND_CONNECTIONS,
        )

    def release(self):
        """Free the model resources."""
        self._hands.close()


if __name__ == "__main__":
    print("Phase 2 - Gesture Recognition Engine self-test")
    tracker = HandTracker()
    print("MediaPipe hands model initialised OK.")
    tracker.release()