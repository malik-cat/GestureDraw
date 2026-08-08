"""
PHASE 4 (updated): Real-Time Application Loop (Orchestrator)
============================================================
Ties all the modules together into the live app.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Responsibilities (this phase)
-----------------------------
* Phase 1 (continuity): gesture debouncing, stroke dots, max-jump guard.
* Phase 2 (full canvas): strokes suppressed only inside UI hit-zones; the
  whole screen is otherwise drawable.
* Phase 3 (sidebar):     left tool/options panel with hit-testing.
* Phase 4 (UX):          undo/redo, save/export, brush size, gesture
                         feedback text, first-run privacy notice.
* Phase 5 (privacy):     on-screen "REC" indicator while the camera is
                         open; nothing is written to disk unless the user
                         explicitly saves/export.
"""

import os
from datetime import datetime

import cv2

import config
from canvas import Canvas, Palette, Sidebar
from frame_capture import FrameSource
from hand_tracker import HandTracker, Gesture


class AirCanvasApp:
    """Combines camera + tracker + canvas + UI into the live loop."""

    #: file written after the first launch so the consent notice appears once
    FIRST_RUN_FLAG = config.PROJECT_DIR / ".first_run_done"

    def __init__(self, camera_source=config.CAMERA_SOURCE,
                 model_path=config.MODEL_PATH):
        self.camera_source = camera_source
        self.tracker = HandTracker(model_path=model_path)
        self.canvas = None          # created once frame size is known
        self.palette = None
        self.sidebar = None

        self.tool = config.TOOL_RED        # active tool id
        self.gesture_name = "NONE"

        # Bookkeeping for drawing / composition.
        self._cursor = None

    # ------------------------------------------------------------------ #
    # Tool actions                                                        #
    # ------------------------------------------------------------------ #

    def apply_tool(self, tool):
        """Apply any tool/action id (palette, sidebar, keyboard)."""
        if tool == config.TOOL_RED:
            self.canvas.set_color(config.COLOR_RED);   self.tool = tool
        elif tool == config.TOOL_GREEN:
            self.canvas.set_color(config.COLOR_GREEN); self.tool = tool
        elif tool == config.TOOL_BLUE:
            self.canvas.set_color(config.COLOR_BLUE);  self.tool = tool
        elif tool == config.TOOL_ERASER:
            self.tool = tool
        elif tool == config.TOOL_CLEAR:
            self.canvas.clear(); self.tool = tool
        elif tool == "undo":
            self.canvas.undo()
        elif tool == "redo":
            self.canvas.redo()
        elif tool == "save":
            self.export_canvas()
        elif tool == "brush+":
            self.canvas.set_brush(config.BRUSH_STEP)
        elif tool == "brush-":
            self.canvas.set_brush(-config.BRUSH_STEP)
        self.canvas.reset_pointer()

    def export_canvas(self):
        """Save the whiteboard to a timestamped PNG in exports/."""
        config.EXPORTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = config.EXPORTS_DIR / f"gesturedraw_{stamp}.png"
        cv2.imwrite(str(path), self.canvas.layer)
        print(f"[EXPORT] saved {path}")

    # ------------------------------------------------------------------ #
    # Gesture / hand handling                                             #
    # ------------------------------------------------------------------ #

    def inside_ui(self, x, y):
        """True when (x, y) sits inside the palette or the sidebar."""
        if self.sidebar and x < self.sidebar.width:
            return True
        if y < config.HEADER_HEIGHT:
            return True
        return False

    def handle_hand(self, frame, landmarks, width, height, frame_no):
        """
        Interpret one hand and act on the canvas.

        Args:
            frame: BGR frame (for drawing cursor feedback).
            landmarks: list of 21 landmarks.
            width/height: live frame size.
            frame_no: frame counter (unused but kept for future smoothing)

        Returns:
            None (mutates app state in place).
        """
        fingers = HandTracker.fingers_up(landmarks)
        raw = HandTracker.classify(fingers)

        x, y = HandTracker.index_tip_pixels(landmarks, width, height)
        self._cursor = (x, y)

        if raw == Gesture.SELECT:
            # SELECT click: hit the palette/sidebar first.
            self.select_from_uis(x, y)
            self.canvas.reset_pointer()
            self.gesture_name = "SELECT"
            return

        if raw == Gesture.DRAW:
            # Draw everywhere except UI hit-zones (Phase 2 decision (a)).
            if self.inside_ui(x, y):
                self.canvas.reset_pointer()
            else:
                eraser = (self.tool == config.TOOL_ERASER)
                self.canvas.stroke(x, y, is_eraser=eraser)
            self.gesture_name = "DRAW"
            return

        if raw == Gesture.CLEAR:
            self.canvas.clear(); self.canvas.reset_pointer()
            self.gesture_name = "CLEAR (palm)"
            return

        self.gesture_name = "HOVER"
        self.canvas.reset_pointer()

    def select_from_uis(self, x, y):
        """Hit-test both UI panels and apply any selected tool."""
        tool = None
        if self.sidebar:
            tool = self.sidebar.hit_test(x, y)
        if tool is None and self.palette:
            tool = self.palette.hit_test(x, y)
        if tool:
            self.apply_tool(tool)

    def draw_feedback(self, frame, width, height):
        """Draw the REC indicator + current gesture/tool text."""
        # Live camera indicator (always shown while the app reads the cam).
        cv2.circle(frame, (width - 22, 18), 6, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (width - 52, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                    cv2.LINE_AA)

        label = f"{self.gesture_name} | tool: {self.tool}"
        cv2.putText(frame, label, config.GESTURE_TEXT_POS,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    config.GESTURE_TEXT_COLOR, 1, cv2.LINE_AA)

        # Small brush size readout.
        cv2.putText(frame, f"brush {self.canvas.brush_size}",
                    (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    config.GESTURE_TEXT_COLOR, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    # Main loop                                                           #
    # ------------------------------------------------------------------ #

    def run(self):
        """Open the camera and process frames until 'q'."""
        camera = None
        try:
            camera = FrameSource(self.camera_source)
        except RuntimeError as exc:
            print(f"\nERROR: {exc}\nClose other apps using the camera and try "
                  "again.")
            return

        first_frame = camera.read()
        if first_frame is None:
            print("ERROR: could not capture a frame from the camera.")
            camera.release()
            return

        height, width = first_frame.shape[:2]
        self.canvas = Canvas(width, height)
        self.palette = Palette()
        self.sidebar = Sidebar()

        # First-run consent overlay (Phase 5).
        show_consent, consent_frames = self._first_run()
        frame_no = 0

        try:
            while True:
                frame = camera.read()
                if frame is None:
                    break
                frame_no += 1
                height, width = frame.shape[:2]

                # ---- detect + debounce gestures -------------------------
                hands = self.tracker.get_hand_landmarks(frame)
                if not hands:
                    # Hand left the frame: reset stabiliser + stroke pointer.
                    self.tracker._stabilizer.reset()
                    self.canvas.reset_pointer()
                    self.gesture_name = "NO HAND"
                else:
                    # Debounced gesture feed (Phase 1 continuity fix).
                    landmarks = hands[0]
                    stable = self.tracker.classify_stable(landmarks)
                    self.gesture_name = Gesture(stable).name
                    self.handle_hand(frame, landmarks, width, height,
                                     frame_no)

                # ---- compose + draw UI ----------------------------------
                composite = self.canvas.overlay(frame)
                self.palette.draw(composite, self.tool)
                self.sidebar.draw(composite, self.tool)
                self.draw_feedback(composite, width, height)

                # Consent overlay (Phase 5).
                if show_consent and consent_frames > 0:
                    self._overlay_consent(composite)
                    consent_frames -= 1
                    if consent_frames == 0:
                        show_consent = False

                cv2.imshow("GestureDraw - Air Canvas", composite)

                key = cv2.waitKey(1) & 0xFF
                if self._keyboard(key):
                    break

        finally:
            camera.release()
            self.tracker.release()
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    # Keyboard / first-run helpers                                       #
    # ------------------------------------------------------------------ #

    def _keyboard(self, key):
        """Handle a keypress; returns True to quit."""
        if key == ord(config.KEY_QUIT):
            return True
        if key == ord(config.KEY_UNDO):
            self.canvas.undo()
        elif key == ord(config.KEY_REDO):
            self.canvas.redo()
        elif key == ord(config.KEY_CLEAR):
            self.canvas.clear()
        elif key == ord(config.KEY_SAVE):
            self.export_canvas()
        elif key == ord(config.KEY_ERASER):
            self.tool = config.TOOL_ERASER
        elif key == ord('r'):
            self.tool = config.TOOL_RED
        elif key == ord('g'):
            self.tool = config.TOOL_GREEN
        elif key == ord('b'):
            self.tool = config.TOOL_BLUE
        elif key == ord('+') or key == ord('='):
            self.canvas.set_brush(config.BRUSH_STEP)
        elif key == ord('-'):
            self.canvas.set_brush(-config.BRUSH_STEP)
        return False

    def _first_run(self):
        """Return (show_consent, frames) if this is the first launch."""
        if self.FIRST_RUN_FLAG.exists():
            return False, 0
        self.FIRST_RUN_FLAG.write_text("done\n", encoding="utf-8")
        return True, 180     # show the notice for ~3 seconds

    def _overlay_consent(self, frame):
        """Draw the local-only privacy notice over the feed."""
        msg = ("Hand tracking runs locally with MediaPipe. "
               "No image leaves this device.")
        cv2.rectangle(frame, (0, config.HEADER_HEIGHT),
                      (frame.shape[1] - 1, config.HEADER_HEIGHT + 26),
                      (0, 0, 0), -1)
        cv2.putText(frame, msg, (10, config.HEADER_HEIGHT + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)


def run_app():
    app = AirCanvasApp()
    app.run()


if __name__ == "__main__":
    run_app()