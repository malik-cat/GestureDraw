"""
PHASE 1: Project Setup & Frame Capture Pipeline
================================================
This module is responsible for the low-level camera I/O that every other
phase depends on.

Author : Mohammad Liaquat Ali
Repository : https://github.com/malik-cat/GestureDraw

Responsibilities
----------------
* Open the default webcam (`cv2.VideoCapture(0)`).
* Horizontal-flip every captured frame so the feed behaves like a selfie
  camera (mirrored) which makes hand gestures feel natural.
* Expose a small "read current frame" API that the main loop can call.

Design notes
------------
All camera handling is isolated here. Phases 2-4 never touch `cv2.VideoCapture`
directly; they only ask this object for a (mirrored) BGR frame, which keeps the
code modular and easy to test.
"""

import cv2


class FrameSource:
    """Wraps the webcam and hands out mirrored BGR frames."""

    def __init__(self, source=0, width=1280, height=720):
        """
        Open the camera and request a fixed resolution.

        Args:
            source: Index of the camera (0 = system default webcam).
            width:  Desired capture width in pixels.
            height: Desired capture height in pixels.
        """
        # Open the default camera device.
        self._capture = cv2.VideoCapture(source)

        # Requesting a resolution is a hint - not every webcam honours it,
        # so the real size is re-read after opening (see `frame_width` below).
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Fail fast if the camera could not be opened (e.g. already in use).
        if not self._capture.isOpened():
            raise RuntimeError("Could not open the webcam. "
                               "Check that no other app is using it.")

        # Ask MediaPipe-style processing to run in real time if possible.
        cv2.setUseOptimized(True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def read(self):
        """
        Grab the newest frame from the camera.

        Returns:
            ndarray | None -> Mirrored BGR frame, or None when the stream
            has ended / failed so the caller can terminate cleanly.
        """
        success, frame = self._capture.read()
        if not success or frame is None:
            return None

        # Horizontal flip so movements appear like looking into a mirror.
        return cv2.flip(frame, 1)

    def release(self):
        """Free the camera resource (call once at shutdown)."""
        self._capture.release()

    # ------------------------------------------------------------------
    # Convenience helpers
    #
    @property
    def frame_width(self) -> int:
        """Width of the frames actually delivered by the camera."""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        """Height of the frames actually delivered by the camera."""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        """Safety-net cleanup when the object is garbage collected."""
        try:
            self.release()
        except Exception:
            pass


if __name__ == "__main__":
    # Quick self-test: read one frame and show how it looks.
    print("Phase 1 - Frame Capture Pipeline self-test")
    cam = FrameSource()
    frame = cam.read()
    if frame is None:
        print("No frame captured. Check your webcam.")
    else:
        print(f"Captured frame with size: {frame.shape[1]}x{frame.shape[0]}")
    cam.release()