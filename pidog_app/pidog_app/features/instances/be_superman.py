"""Be superman.

Port of the SunFounder ``6_be_picked_up.py`` example. The dog watches
its IMU for the pick-up gesture: tilted nose-down, then held level with
its belly facing down. On pickup it strikes the superman pose — front
legs stretched forward, rear legs stretched back, red 'boom' chest
light, wagging tail and a "woohoo". Tilted nose-down again (being put
down) it stands, and after ``superman.restore_delay`` seconds at rest
it returns to the default waiting state (sit + breath-yellow light).

The same state machine has two drivers:

- The app's pickup watcher calls :meth:`poll` continuously, but only
  while the dog is awake and idle — never while another feature or
  queued task owns the body.
- The LLM-invoked :meth:`run` actively waits for a pickup for
  ``superman.duration`` seconds. It refuses to start while the dog is
  still performing queued actions or servo motion.
"""
from __future__ import annotations

import logging
import threading
import time

from pidog.action_flow import ActionStatus

from ..base import Feature, FeatureResult

log = logging.getLogger(__name__)


class BeSuperman(Feature):
    name = "be_superman"
    description = (
        "Be superman. The dog stands and waits to be picked up. When "
        "you lift it and hold it level (belly facing down) it stretches "
        "its legs like a flying superhero, lights its chest red, wags "
        "its tail and shouts woohoo; tilt it nose-down to land it back "
        "on its feet. Use this when the user says 'be superman', 'fly "
        "like superman', or wants to pick the dog up and make it fly."
    )

    # Accelerometer thresholds (ax). Gravity is -1G = -16384.
    # ax < DOWN_THRESHOLD: dog is tilted nose-down (being put down).
    # ax > UP_THRESHOLD:   dog is held level, belly down (flying).
    DOWN_THRESHOLD = -18000
    UP_THRESHOLD = -13000

    # Superman leg pose (8 leg servos: front forward, rear back).
    FLY_POSE = [45, -45, 90, -80, 90, 90, -90, -90]

    POLL_INTERVAL = 0.02

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._upflag = False
        self._downflag = False
        self._is_up = False
        # Set when the dog lands; while it stays at rest, the default
        # waiting pose is restored after ``superman.restore_delay`` secs.
        self._landed_at = None

    def _restore_delay(self) -> float:
        return float(self.cfg.get("superman.restore_delay", 3.0)
                     if self.cfg is not None else 3.0)

    @property
    def parameters(self) -> dict:
        # Same pattern as guard_the_perimeter: name the configured
        # default and tell the model to omit the argument so the LLM
        # doesn't invent a duration the user never asked for.
        default = (self.cfg.get("superman.duration", 60.0)
                   if self.cfg is not None else 60.0)
        return {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": (
                        "How many seconds to wait to be picked up. OMIT "
                        "this argument unless the user explicitly asks "
                        "for a specific duration; when omitted the "
                        f"configured default of {float(default):g} "
                        "seconds is used."
                    ),
                },
            },
            "required": [],
        }

    # ── busy guard ────────────────────────────────────────────────────
    def is_busy(self) -> bool:
        """True while another task owns the body: the action flow is
        executing queued actions, or any servo group still has buffered
        motion (e.g. a standby fidget in flight)."""
        flow = self.body.action_flow
        dog = self.body.dog
        return (
            flow.thread_action_state == ActionStatus.ACTIONS
            or not flow.action_queue.empty()
            or not (dog.is_legs_done()
                    and dog.is_head_done()
                    and dog.is_tail_done())
        )

    @property
    def is_up(self) -> bool:
        """True while the dog is being held up in the superman pose."""
        return self._is_up

    def reset(self) -> None:
        """Clear the gesture state machine.

        Called by drivers while the dog is busy so a pickup that started
        mid-task can't fire late when the dog goes idle again. The state
        machine re-arms itself on the next clean readings: a level
        reading sets ``_downflag``, a nose-down reading sets ``_upflag``.
        A pending landing timer (``_landed_at``) is left alone — the
        restore only fires while the dog is at rest and idle anyway.
        """
        with self._lock:
            self._upflag = False
            self._downflag = False
            self._is_up = False

    # ── shared state machine (port of the example loop) ───────────────
    def poll(self) -> str | None:
        """Run one pickup-detection step.

        Returns ``'fly'``, ``'stand'`` or ``'rest'`` when a transition
        fired (the transition itself is executed here and takes ~1s),
        else ``None``.
        """
        ax = self.senses.imu()[0][0]
        with self._lock:
            # Landed and still at rest (ax in the dead zone between the
            # thresholds): return to the default waiting pose once the
            # restore delay has passed. A level reading cancels the
            # pending restore — the dog is airborne again, not resting
            # on the ground.
            if (self._landed_at is not None
                    and self.DOWN_THRESHOLD <= ax <= self.UP_THRESHOLD
                    and time.time() - self._landed_at
                        >= self._restore_delay()):
                self._landed_at = None
                self._default_pose()
                return "rest"

            if ax < self.DOWN_THRESHOLD:
                # Nose-down is part of the landing gesture itself —
                # keep any pending restore timer running.
                self.body.dog.body_stop()
                if not self._upflag:
                    self._upflag = True
                if self._downflag:
                    self._is_up = False
                    self._downflag = False
                    self._stand()
                    self._landed_at = time.time()
                    return "stand"
            elif ax > self.UP_THRESHOLD:
                self._landed_at = None
                self.body.dog.body_stop()
                if self._upflag:
                    self._is_up = True
                    self._upflag = False
                    self._fly()
                    return "fly"
                if not self._downflag:
                    self._downflag = True
        return None

    # ── LLM entry point ───────────────────────────────────────────────
    def run(self, duration=None, **kwargs) -> FeatureResult:
        duration = float(
            duration if duration is not None
            else (self.cfg.get("superman.duration", 60.0)
                  if self.cfg is not None else 60.0))

        # Only allowed when no other feature or task owns the body.
        if self.is_busy():
            return FeatureResult(
                text="I'm busy with something else right now — I can't "
                     "be superman at the same time.",
                success=False,
            )

        log.info("be_superman: waiting %.0fs for pickup", duration)
        flights = 0
        deadline = time.time() + duration
        with self.body.thinking():
            # Serialize with any in-flight watcher poll so an accepted
            # fly() can't overlap this stand-up.
            with self._lock:
                self._upflag = False
                self._downflag = False
                self._is_up = False
                self._landed_at = None
                self._stand()
            # Keep polling past the deadline while a landing restore is
            # still pending so the dog always settles back into its
            # default waiting pose before the tool returns.
            while time.time() < deadline or self._landed_at is not None:
                if self.poll() == "fly":
                    flights += 1
                time.sleep(self.POLL_INTERVAL)

        if flights:
            text = (f"Woohoo! I flew like superman {flights} "
                    f"time{'s' if flights != 1 else ''}!")
        else:
            text = ("I was ready to fly but nobody picked me up — "
                    "no superhero action this time.")
        if self._is_up:
            text += " I'm still being held up in my flying pose."
        return FeatureResult(text=text, success=True,
                             extra={"flights": flights})

    # ── transitions (ported from the example) ─────────────────────────
    def _fly(self) -> None:
        log.info("be_superman: flying")
        self.body.light("boom", "red", 3)
        self.body.dog.legs.servo_move(self.FLY_POSE, speed=60)
        self.body.dog.do_action("wag_tail", step_count=10, speed=100)
        self.body.dog.speak("woohoo", volume=80)
        self.body.wait_legs_done()
        time.sleep(1)

    def _stand(self) -> None:
        log.info("be_superman: standing")
        self.body.light("breath", "green", 1)
        self.body.dog.do_action("stand", speed=60)
        self.body.wait_legs_done()
        time.sleep(1)

    def _default_pose(self) -> None:
        """Return to the app's idle waiting state: sit + yellow light.

        Mirrors the posture/light the app establishes at startup and in
        ``_reset_to_startup``. Called ~``restore_delay`` seconds after
        the dog is put down, once it is at rest on the ground.
        """
        log.info("be_superman: back to default waiting state")
        self.body.sit()
        self.body.light("breath", "yellow", 1)
