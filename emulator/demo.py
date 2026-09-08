#!/usr/bin/env python3
"""Demo script for the PiDog emulator.

Runs a short sequence of preset actions on a virtual PiDog and opens the
renderer so you can watch them.  No hardware required.

    python3 -m emulator.demo
or
    python3 emulator/demo.py
"""

import threading
import time

from emulator import Pidog, run_emulator


def action_sequence(dog):
    """Run a sequence of actions; the renderer watches in the main thread."""
    time.sleep(1.0)
    sequence = [
        ("stand", 1, 60),
        ("sit", 1, 60),
        ("half_sit", 1, 60),
        ("lie", 1, 60),
        ("stand", 1, 60),
        ("forward", 6, 90),
        ("turn_left", 4, 90),
        ("turn_right", 4, 90),
        ("trot", 8, 95),
        ("stretch", 1, 40),
        ("push_up", 4, 60),
        ("shake_head", 3, 90),
        ("wag_tail", 10, 100),
        ("tilting_head", 2, 60),
        ("head_bark", 3, 80),
        ("doze_off", 1, 80),
        ("lie", 1, 60),
    ]
    for name, steps, speed in sequence:
        print(f"[demo] {name} x{steps} @ {speed}")
        # stand up before walking actions
        if name in ("forward", "backward", "turn_left", "turn_right", "trot"):
            dog.do_action("stand", speed=60)
            dog.wait_all_done()
        dog.do_action(name, step_count=steps, speed=speed)
        dog.wait_all_done()
        time.sleep(0.4)
    print("[demo] sequence complete (window stays open; close with ESC)")


def main():
    dog = Pidog()
    time.sleep(0.5)
    t = threading.Thread(target=action_sequence, args=(dog,), daemon=True)
    t.start()
    # The renderer must run in the main thread (pygame requirement).
    run_emulator(dog)
    dog.close()


if __name__ == "__main__":
    main()
