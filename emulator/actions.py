#!/usr/bin/env python3
"""Preset action dictionary for the PiDog emulator.

This is a faithful re-implementation of ``pidog.actions_dictionary.ActionDict``.
It uses the emulator's :class:`~emulator.kinematics.Kinematics` for the
angle calculations and the :class:`~emulator.kinematics.Walk` / ``Trot`` gait
generators, so the produced angle frames are byte-for-byte equivalent to the
real robot's action dictionary.
"""

from math import sin

from .kinematics import Kinematics, Walk, Trot


class ActionDict(dict):
    """Lookup of preset actions by name (mirrors the real ``ActionDict``).

    ``actions_dict[name]`` returns ``(frames, part)`` where ``frames`` is a list
    of angle frames and ``part`` is one of ``'legs'``, ``'head'`` or ``'tail'``.
    """

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        super().__init__()
        self.barycenter = -15
        self.height = 95

    def __getitem__(self, item):
        return eval("self.%s" % item.replace(" ", "_"))

    def set_height(self, height):
        if height in range(20, 95):
            self.height = height

    def set_barycenter(self, offset):
        if offset in range(-60, 60):
            self.barycenter = offset

    # stand
    @property
    def stand(self):
        x = self.barycenter
        y = 95
        return [
            Kinematics.legs_angle_calculation(
                [[x, y], [x, y], [x + 20, y - 5], [x + 20, y - 5]]),
        ], 'legs'

    # sit
    @property
    def sit(self):
        return [
            [30, 60, -30, -60, 80, -45, -80, 45],
        ], 'legs'

    # lie
    @property
    def lie(self):
        return [
            [45, -45, -45, 45, 45, -45, -45, 45]
        ], 'legs'

    # lie_with_hands_out
    @property
    def lie_with_hands_out(self):
        return [
            [-60, 60, 60, -60, 45, -45, -45, 45],
        ], 'legs'

    # forward
    @property
    def forward(self):
        data = []
        forward = Walk(fb=Walk.FORWARD, lr=Walk.STRAIGHT)
        coords = forward.get_coords()
        for coord in coords:
            data.append(Kinematics.legs_angle_calculation(coord))
        return data, 'legs'

    # backward
    @property
    def backward(self):
        data = []
        backward = Walk(fb=Walk.BACKWARD, lr=Walk.STRAIGHT)
        coords = backward.get_coords()
        for coord in coords:
            data.append(Kinematics.legs_angle_calculation(coord))
        return data, 'legs'

    # turn_left
    @property
    def turn_left(self):
        data = []
        turn_left = Walk(fb=Walk.FORWARD, lr=Walk.LEFT)
        coords = turn_left.get_coords()
        for coord in coords:
            data.append(Kinematics.legs_angle_calculation(coord))
        return data, 'legs'

    # turn_right
    @property
    def turn_right(self):
        data = []
        turn_right = Walk(fb=Walk.FORWARD, lr=Walk.RIGHT)
        coords = turn_right.get_coords()
        for coord in coords:
            data.append(Kinematics.legs_angle_calculation(coord))
        return data, 'legs'

    # trot
    @property
    def trot(self):
        data = []
        trot = Trot(Trot.FORWARD, Trot.STRAIGHT)
        coords = trot.get_coords()
        for coord in coords:
            data.append(Kinematics.legs_angle_calculation(coord))
        return data, 'legs'

    # stretch
    @property
    def stretch(self):
        return [
            [-80, 70, 80, -70, -20, 64, 20, -64],
        ], 'legs'

    # push_up
    @property
    def push_up(self):
        return [
            [90, -30, -90, 30, 80, 70, -80, -70],
            [45, 35, -45, -35, 80, 70, -80, -70]
        ], 'legs'

    # doze_off
    @property
    def doze_off(self):
        start = -30
        am = 20
        angs = []
        t = 4
        for i in range(0, am + 1, 1):  # up
            anl_f = start + i
            anl_b = 45 - i
            angs += [[45, anl_f, -45, -anl_f, 45, -anl_b, -45, anl_b]] * t
        for _ in range(4):  # stop
            anl_f = start + am
            anl_b = 45 - am
            angs += [[45, anl_f, -45, -anl_f, 45, -anl_b, -45, anl_b]] * t
        for i in range(am, -1, -1):  # down
            anl_f = start + i
            anl_b = 45 - i
            angs += [[45, anl_f, -45, -anl_f, 45, -anl_b, -45, anl_b]] * t
        for _ in range(4):  # stop
            anl_f = start
            anl_b = 45
            angs += [[45, anl_f, -45, -anl_f, 45, -anl_b, -45, anl_b]] * t
        return angs, 'legs'

    # nod_lethargy
    @property
    def nod_lethargy(self):
        y = 0
        angs = []
        for i in range(21):
            r = round(10 * sin(i * 0.314), 2)
            p = round(10 * sin(i * 0.628) - 30, 2)
            if r == -10 or r == 10:
                for _ in range(10):
                    angs.append([y, r, p])
            angs.append([y, r, p])
        return angs, 'head'

    # shake_head
    @property
    def shake_head(self):
        amplitude = 60
        angs = []
        for i in range(21):
            y1 = amplitude * sin(i * 0.314)
            angs.append([y1, 0, 0])
        return angs, 'head'

    # tilting_head_left
    @property
    def tilting_head_left(self):
        return [[0, -25, 15]], 'head'

    # tilting_head_right
    @property
    def tilting_head_right(self):
        return [[0, 25, 20]], 'head'

    # tilting_head (left and right)
    @property
    def tilting_head(self):
        yaw = 0
        roll = 22
        pitch = 20
        return [[yaw, roll, pitch]] * 20 \
            + [[yaw, -roll, pitch]] * 20, 'head'

    # head_bark
    @property
    def head_bark(self):
        return [[0, 0, -40],
                [0, 0, -10],
                [0, 0, -10],
                [0, 0, -40]], 'head'

    # wag_tail
    @property
    def wag_tail(self):
        return [[-30], [30]], 'tail'

    # head_up_down
    @property
    def head_up_down(self):
        return [
            [0, 0, 20],
            [0, 0, 20],
            [0, 0, -10]
        ], 'head'

    # half_sit
    @property
    def half_sit(self):
        return [
            [25, 25, -25, -25, 64, -45, -64, 45],
        ], 'legs'
