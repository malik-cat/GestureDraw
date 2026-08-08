"""
PHASE 4: Real-Time Application Loop (Orchestrator)
==================================================
Ties Phases 1-3 together into a single live application:

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

     Phase 1  FrameSource  -> delivers mirrored webcam frames
    Phase 2  HandTracker  -> search hand, compute gesture
    Phase 3  Canvas       -> persist strokes, merge with video
            Palette      -> interactive header colours

Control scheme
--------------
Gesture (indices in thumb, index, middle, ring, pinky):
  * [Index raised]                  -> DRAW    -> draw at the fingertip
  * [Index + Middle raised]        -> SELECT  -> move & pick a tool (header)
  * [All five fingers raised]      -> CLEAR   -> wipe the whiteboard
  * anything else                  -> NONE    -> hold still (no action)

Keyboard replacements (convenience):
  q  - quit
  c  - clear the whiteboard echo `c`
"""

import cv2

from frame_capture import FrameSource
from hand_tracker import HandTracker, Gesture
from canvas import (Canvas, Palette, HEADER_HEIGHT,
                    TOOL_RED, TOOL_GREEN, TOOL_BLUE,
                    TOOL_ERASER, TOOL_CLEAR,
                    COLOR_RED, COLOR_GREEN, COLOR_BLUE)


class AirCanvasApp:
    """
    Combines camera + tracker + canvas into the interactive loop.
    """

    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.tracker = HandTracker()
        # Canvas / palette are created once the frame size is known.
        self.canvas = None
        self.palette = None
        self.active_tool = TOOL_GREEN

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def apply_tool(self, tool):
        """
        Apply a palette tool selection.

        Returns True when `tool == CLEAR` so the caller can also
        immediately clear the canvas (single clear click).
        """
        if tool == TOOL_RED:
            self.canvas.set_color(COLOR_RED)
            self.active_tool = TOOL_RED
        elif tool == TOOL_GREEN:
            self.canvas.set_color(COLOR_GREEN)
            self.active_tool = TOOL_GREEN
        elif tool == TOOL_BLUE:
            self.canvas.set_color(COLOR_BLUE)
            self.active_tool = TOOL_BLUE
        elif tool == TOOL_ERASER:
            # Eraser doesn't change the stroke colour; it clears pixels.
            self.active_tool = TOOL_ERASER
        elif tool == TOOL_CLEAR:
            # Clear button: wipe the whiteboard immediately.
            self.canvas.clear()
            self.active_tool = TOOL_CLEAR

    # ------------------------------------------------------------------
    # Frame handling
    # ------------------------------------------------------------------

    def process_hand(self, frame, hand_landmarks, frame_width, frame_height):
        """
        Turn a hand into an app action on the canvas.

        Args:
            frame: Current (mirrored) BGR frame, for reference only.
            hand_landmarks: MediaPipe hand landmarks for one hand.
            frame_width:    Live frame width in pixels.
            frame_height:   Live frame height in pixels.

        Returns:
            The fingertip position (x, y) or None.
        """
        fingers = HandTracker.fingers_up(hand_landmarks)
        gesture = HandTracker.classify(fingers)

        x, y = HandTracker.index_tip_pixels(
            hand_landmarks, frame_width, frame_height)

        if gesture == Gesture.DRAW:
            # Draw mode: draw a stroke from the previous fingertip to here.
            if y > HEADER_HEIGHT:          # never draw over the palette
                self.canvas.stroke(x, y, is_eraser=False)
            else:
                self.canvas.reset_pointer()
            return (x, y)

        if gesture == Gesture.SELECT:
            # Selection mode: highlight the swatch the tip is over and,
            # if inside the header, choose that tool.
            tool = self.palette.hit_test(x, y, frame_width)
            if tool:
                self.apply_tool(tool)
            # Move the stroke pointer so drawing does not jump later.
            self.canvas.reset_pointer()
            return (x, y)

        if gesture == Gesture.CLEAR:
            # Utility gesture: wipe the whiteboard instantly.
            self.canvas.clear()
            self.canvas.reset_pointer()
            return (x, y)

        # Gesture.NONE (uniform folded fingers): hover, do nothing.
        self.canvas.reset_pointer()
        return (x, y)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Open the camera and run until the user presses 'q'."""
        camera = FrameSource(self.camera_source)
        try:
            # ----- first frame: learn the real frame size --------------
            frame = camera.read()
            if frame is None:
                print("ERROR: could not capture a frame from the camera.")
                return
            frame_height, frame_width = frame.shape[:2]

            # Build the canvas + palette for that size.
            self.canvas = Canvas(frame_width, frame_height)
            self.palette = Palette()

            while True:
                frame = camera.read()
                if frame is None:
                    break

                # ----- Phase 2: hand tracking -------------------------
                results = self.tracker.get_hand_landmarks(frame)
                fingertip = None

                if results.multi_hand_landmarks:
                    for hand in results.multi_hand_landmarks:
                        # Visual skeleton overlays the feed.
                        self.tracker.draw_landmarks(frame, hand)
                        fingertip = self.process_hand(
                            frame, hand, frame_width, frame_height)
                    # If a selection hit a toolbar button, colour the
                    # header light yellow reply after processing.
                else:
                    # No hand in view - break any partially drawn stroke.
                    self.canvas.reset_pointer()

                # ----- Phase 3: render palette + drawing ---------------
                # Make the palette row visible again (it is overdrawn).
                self.palette.draw(frame, self.active_tool)

                # Draw a little cursor dot where the fingertip is.
                if fingertip is not None:
                    cv2.circle(frame, fingertip, 8, (0, 255, 255), 2)

                # ----- Dual-layer composite ---------------------------
                composite = self.canvas.overlay(frame)
                cv2.namedWindow("GestureDraw - Air Canvas", cv2.WINDOW_NORMAL)
                cv2.imshow("GestureDraw - Air Canvas", composite)

                # ----- Keyboard handling -------------------------------
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("e"):
                    # Keyboard eraser
                    self.active_tool = TOOL_ERASER
                    self.canvas.reset_pointer()
                elif key == ord("r"):
                    self.canvas.set_color(COLOR_RED)
                    self.active_tool = TOOL_RED
                elif key == ord("g"):
                    self.canvas.set_color(COLOR_GREEN)
                    self.active_tool = TOOL_GREEN
                elif key == ord("b"):
                    self.canvas.set_color(COLOR_BLUE)
                    self.active_tool = TOOL_BLUE

        finally:
            camera.release()
            self.tracker.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirCanvasApp()
    app.run()