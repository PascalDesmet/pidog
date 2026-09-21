"""Frame-difference motion detection.

Compares each camera frame against the previous one and reports how much
of the image changed. Used by the perimeter guard to spot movement while
the dog holds its head still. Call :meth:`MotionDetector.reset` whenever
the camera view changes (e.g. after a head move) — the frames captured
while the head is swinging would otherwise look like constant motion.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:
    import cv2
    import numpy as np
except ImportError:  # dev machines without the camera stack
    cv2 = None
    np = None


class MotionDetector:
    """Detects movement between successive camera frames.

    Each call to :meth:`detected` compares the frame to the *previous*
    frame (not a fixed baseline), so an object that moved and then parked
    stops counting as movement — that is what lets the guard notice when
    the scene has gone calm again.
    """

    def __init__(self, threshold: float = 0.02,
                 blur: int = 21, pixel_threshold: int = 25):
        if cv2 is None or np is None:
            raise ImportError("MotionDetector needs opencv-python and numpy")
        # ``threshold``: fraction of pixels that must change to count as
        # movement. ``pixel_threshold``: per-pixel grey change (0-255)
        # needed before a pixel counts as changed at all.
        self.threshold = threshold
        self.blur = blur
        self.pixel_threshold = pixel_threshold
        self._prev = None
        self.last_change = 0.0

    def reset(self) -> None:
        """Forget the previous frame (call after the camera view moves)."""
        self._prev = None
        self.last_change = 0.0

    def detected(self, frame) -> bool:
        """True when ``frame`` differs enough from the previous frame.

        The first call after :meth:`reset` only stores the reference
        frame and returns False.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        if self._prev is None:
            self._prev = gray
            return False
        diff = cv2.absdiff(self._prev, gray)
        self._prev = gray
        _, mask = cv2.threshold(diff, self.pixel_threshold, 255,
                                cv2.THRESH_BINARY)
        self.last_change = float(np.count_nonzero(mask)) / mask.size
        return self.last_change >= self.threshold
