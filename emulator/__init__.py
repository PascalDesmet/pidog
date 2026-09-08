#!/usr/bin/env python3
"""PiDog software emulator.

A drop-in, hardware-free replacement for ``pidog.pidog.Pidog`` that drives an
in-memory virtual robot visualised by a pygame renderer.

Quick start
-----------
    from emulator import Pidog, run_emulator
    from time import sleep

    dog = Pidog()
    sleep(0.5)
    dog.do_action('stand', speed=60)
    dog.wait_all_done()
    dog.do_action('forward', step_count=4, speed=90)
    run_emulator(dog)   # opens the window; closes on ESC

To run an existing pidog example without hardware, change::

    from pidog import Pidog
to::
    from emulator import Pidog
and add ``run_emulator(dog)`` (or ``from emulator import run_emulator``) at the
end of the script before ``dog.close()``.
"""

from .pidog_emu import Pidog
from .renderer import Renderer, run_emulator
from .kinematics import Kinematics, Walk, Trot
from .actions import ActionDict

__all__ = ["Pidog", "Renderer", "run_emulator", "Kinematics", "Walk", "Trot", "ActionDict"]
