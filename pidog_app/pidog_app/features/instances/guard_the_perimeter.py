"""Guard the perimeter.

The dog lies down and very slowly sweeps its gaze across the scene,
watching for movement with the camera. Movement triggers a timestamped
photo in ``surveillance_photos/``; while movement continues, a photo is
taken every ``guard.photo_interval`` seconds. Shooting stops after
``guard.calm_timeout`` seconds without movement and the dog resumes its
slow scan.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from ...config import PROJECT_ROOT
from ...vision.motion import MotionDetector
from ..base import Feature, FeatureResult

log = logging.getLogger(__name__)

# Head positions ([yaw, roll, pitch]) swept while guarding.
GUARD_POSITIONS = ((-75, 0, 0), (-25, 0, 0), (25, 0, 0), (75, 0, 0))

HEAD_SPEED = 25   # very slow sweep
DWELL_S = 4.0     # watch time per heading
SETTLE_S = 0.4    # let a fresh frame arrive after the head stops
SAMPLE_S = 0.2    # frame sampling period while watching


class GuardThePerimeter(Feature):
    name = "guard_the_perimeter"
    description = (
        "Guard the perimeter. The dog lies down and very slowly looks "
        "around, watching for movement with the camera. When it sees "
        "movement it takes photos into the surveillance_photos folder, "
        "named with date and time. Use this when the user says 'guard "
        "the perimeter', 'keep watch', or 'watch the house'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "duration": {
                "type": "number",
                "description": (
                    "How many seconds to keep watch. Defaults to the "
                    "guard.duration config value."
                ),
            },
        },
        "required": [],
    }

    def run(self, duration=None, **kwargs) -> FeatureResult:
        photo_interval = float(self.cfg.get("guard.photo_interval", 10.0))
        calm_timeout = float(self.cfg.get("guard.calm_timeout", 5.0))
        duration = float(duration if duration is not None
                         else self.cfg.get("guard.duration", 300.0))
        threshold = float(self.cfg.get("guard.motion_threshold", 0.02))
        photos_dir = Path(self.cfg.get("guard.photos_dir",
                                       "surveillance_photos"))
        if not photos_dir.is_absolute():
            photos_dir = PROJECT_ROOT / photos_dir

        try:
            detector = MotionDetector(threshold=threshold)
        except ImportError as e:
            return FeatureResult(text=f"I can't guard: {e}", success=False)

        log.info(
            "guard_the_perimeter: %.0fs, photo_interval=%.1fs, "
            "calm_timeout=%.1fs", duration, photo_interval, calm_timeout)

        taken = 0
        deadline = time.time() + duration
        with self.body.thinking():
            self.body.lie()
            self.camera.start()
            while time.time() < deadline:
                for yrp in GUARD_POSITIONS:
                    if time.time() >= deadline:
                        break
                    # Very slow move to the next heading, then wait for a
                    # settled frame before sampling — frames captured mid-
                    # swing would look like constant motion.
                    self.body.head_move([list(yrp)], immediately=True,
                                        speed=HEAD_SPEED)
                    self.body.wait_head_done()
                    time.sleep(SETTLE_S)
                    detector.reset()
                    taken += self._watch(
                        detector, photos_dir, photo_interval, calm_timeout,
                        dwell_until=time.time() + DWELL_S,
                        deadline=deadline)
            self.body.sit()

        text = (f"I guarded the perimeter for {duration:.0f} seconds and "
                f"took {taken} photo{'s' if taken != 1 else ''}"
                f"{' in ' + str(photos_dir) if taken else ''}.")
        return FeatureResult(
            text=text,
            success=True,
            extra={"photos": taken, "photos_dir": str(photos_dir)},
        )

    def _watch(self, detector: MotionDetector, photos_dir: Path,
               photo_interval: float, calm_timeout: float,
               dwell_until: float, deadline: float) -> int:
        """Watch the current heading; return the number of photos taken.

        Watches until ``dwell_until`` when nothing moves. Once movement is
        seen it keeps watching (past ``dwell_until``) and snaps a photo
        every ``photo_interval`` seconds until ``calm_timeout`` seconds
        pass with no movement. Always ends by ``deadline``.
        """
        photos = 0
        last_motion = None
        last_photo = None
        while time.time() < deadline:
            frame = self.camera.frame()
            now = time.time()
            if frame is not None and detector.detected(frame):
                last_motion = now
                log.info("movement seen (%.1f%% of frame changed)",
                         detector.last_change * 100)
                if last_photo is None or now - last_photo >= photo_interval:
                    self._snap(photos_dir)
                    photos += 1
                    last_photo = now
            elif last_motion is None and now >= dwell_until:
                break  # quiet dwell — move on to the next heading
            elif last_motion is not None and now - last_motion >= calm_timeout:
                log.info("scene calm for %.1fs — resuming scan",
                         calm_timeout)
                break
            time.sleep(SAMPLE_S)
        return photos

    def _snap(self, photos_dir: Path) -> Path:
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.camera.capture(name, path=photos_dir)
        log.info("surveillance photo saved: %s", path)
        return path
