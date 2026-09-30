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
# yaw = left/right
# roll = rotate left/right
# pitch = up/down
#GUARD_POSITIONS = ((-75, 0, 0), (-25, 0, 0), (25, 0, 0), (75, 0, 0))
GUARD_POSITIONS = (
    (-40, 0,   0),   # far left
    (-40, 0,  15),   # up-left
    (  0, 0,  15),   # top
    ( 40, 0,  15),   # up-right
    ( 40, 0,   0),   # far right
    ( 40, 0, -15),   # down-right
    (  0, 0, -15),   # bottom
    (-40, 0, -15),   # down-left
)

class GuardThePerimeter(Feature):
    name = "guard_the_perimeter"
    description = (
        "Guard the perimeter. The dog stands up and very slowly looks "
        "around, watching for movement with the camera. When it sees "
        "movement it takes photos into the surveillance_photos folder, "
        "named with date and time. Use this when the user says 'guard "
        "the perimeter', 'keep watch', or 'watch the house'."
    )
    @property
    def parameters(self) -> dict:
        # The LLM sees this schema and tends to fill in optional numeric
        # arguments with plausible values (e.g. 60) even when the user
        # never asked for a duration. Naming the real configured default
        # and telling the model to omit the argument keeps the config
        # value authoritative.
        default = (self.cfg.get("guard.duration", 300.0)
                   if self.cfg is not None else 300.0)
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": (
                        "How many seconds to keep watch. OMIT this "
                        "argument unless the user explicitly asks for a "
                        f"specific watch duration; when omitted the "
                        f"configured default of {float(default):g} "
                        "seconds is used."
                    ),
                },
            },
            "required": [],
        }

    def run(self, duration=None, **kwargs) -> FeatureResult:
        head_speed = float(self.cfg.get("guard.head_speed", 15))
        dwell_seconds = float(self.cfg.get("guard.dwell_seconds", 4.0))
        settle_seconds = float(self.cfg.get("guard.settle_seconds", 0.4))
        sample_seconds = float(self.cfg.get("guard.sample_seconds", 0.2))
        settle_timeout = float(self.cfg.get("guard.settle_timeout", 6.0))
        photo_interval = float(self.cfg.get("guard.photo_interval", 10.0))
        calm_timeout = float(self.cfg.get("guard.calm_timeout", 5.0))
        duration = float(duration if duration is not None
                         else self.cfg.get("guard.duration", 300.0))
        threshold = float(self.cfg.get("guard.motion_threshold", 0.02))
        photos_dir = Path(self.cfg.get("guard.photos_dir",
                                       "surveillance_photos"))

        log.debug(f"using these settings in guard_the_perimeter.run: Head_speed: {head_speed}, dwell_seconds: {dwell_seconds}, settle_seconds: {settle_seconds}, settle_timeout: {settle_timeout}, sample_seconds: {sample_seconds}, photo_interval: {photo_interval}, calm_timeout: {calm_timeout}, duration: {duration}, motion_treshold: {threshold}")
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
        guard_position_counter = 1
        with self.body.thinking():
            self.body.stand()
            self.camera.start()
            while time.time() < deadline:
                for yrp in GUARD_POSITIONS:
                    if time.time() >= deadline:
                        break
                    # Very slow move to the next heading, then wait for a
                    # settled frame before sampling — frames captured mid-
                    # swing would look like constant motion.
                    self.body.head_move([list(yrp)], immediately=True, speed=head_speed)
                    log.debug(f"Head move step {guard_position_counter} of {len(GUARD_POSITIONS)} to {yrp}")
                    guard_position_counter += 1
                    self.body.wait_head_done()
                    time.sleep(settle_seconds)
                    detector.reset()
                    # Verify the view is actually still before arming the
                    # detector — absorbs servo overshoot and camera frame
                    # lag that survive past wait_head_done + settle.
                    self._wait_for_still(detector, sample_seconds, settle_timeout, deadline)
                    detector.reset()
                    taken += self._watch(
                        detector, photos_dir, photo_interval, calm_timeout,
                        dwell_until=time.time() + dwell_seconds,
                        deadline=deadline,
                        sample_seconds=sample_seconds)
            self.body.sit()

        text = (f"I guarded the perimeter for {duration:.0f} seconds and "
                f"took {taken} photo{'s' if taken != 1 else ''}"
                f"{' in ' + str(photos_dir) if taken else ''}.")
        return FeatureResult(
            text=text,
            success=True,
            extra={"photos": taken, "photos_dir": str(photos_dir)},
        )

    def _wait_for_still(self, detector: MotionDetector,
                        sample_seconds: float, timeout: float,
                        deadline: float, stable_samples: int = 2) -> None:
        """Sample frames until the view stops changing or ``timeout``.

        Returns after ``stable_samples`` consecutive frames below the
        motion threshold. A scene that is genuinely busy simply times out
        and the normal watch proceeds.
        """
        calm = 0
        end = min(deadline, time.time() + timeout)
        while time.time() < end:
            frame = self.camera.frame()
            if frame is not None:
                if detector.detected(frame):
                    calm = 0
                else:
                    calm += 1
                    if calm >= stable_samples:
                        return
            time.sleep(sample_seconds)

    def _watch(self, detector: MotionDetector, photos_dir: Path,
               photo_interval: float, calm_timeout: float,
               dwell_until: float, deadline: float, sample_seconds: float) -> int:
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
            time.sleep(sample_seconds)
        return photos

    def _snap(self, photos_dir: Path) -> Path:
        name = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.camera.capture(name, path=photos_dir)
        log.info("surveillance photo saved: %s", path)
        return path
