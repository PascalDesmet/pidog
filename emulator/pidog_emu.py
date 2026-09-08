#!/usr/bin/env python3
"""Software emulator of the PiDog robot dog.

This module provides a :class:`Pidog` class that reproduces the public API of
the real ``pidog.pidog.Pidog`` class (legs/head/tail movement, preset actions,
pose kinematics, wait helpers, sensor stubs) but drives an in-memory *virtual*
servo state instead of real hardware.  A separate renderer (see
:mod:`emulator.renderer`) reads that virtual state and visualises the dog.

The movement model faithfully mirrors the real implementation:

* three worker threads (legs / head / tail) consume angle frames from buffers,
* :meth:`_ServoGroup.servo_move` interpolates from the current angles to the
  target angles using the real speed / max-dps timing, sleeping in 10 ms steps,
* :meth:`wait_legs_done` / :meth:`wait_all_done` block until the buffers drain,
* :meth:`do_action` replays the preset action dictionary.

Because the interpolation uses real ``time.sleep``, the emulator moves in real
time, just like the physical robot.

Drop-in usage: change ``from pidog import Pidog`` to
``from emulator import Pidog`` in any existing example script and start the
renderer (see :func:`emulator.run_emulator` or ``emulator/demo.py``).
"""

import os
import sys
import threading
from time import sleep, time

import numpy as np

from .kinematics import Kinematics, numpy_mat
from .actions import ActionDict


# ----------------------------------------------------------------------
# Virtual servo group (mirrors robot_hat.Robot)
# ----------------------------------------------------------------------
class _ServoGroup:
    """A group of virtual servos with offset, direction and interpolated moves.

    Mirrors the public surface of ``robot_hat.Robot`` that ``pidog.py`` uses:
    ``servo_positions``, ``offset``, ``direction``, ``max_dps``,
    ``servo_move``, ``servo_write_raw``, ``servo_write_all``, ``reset``,
    ``set_offset``.
    """

    def __init__(self, name, pin_num, init_angles=None, max_dps=428):
        self.name = name
        self.pin_num = pin_num
        self.max_dps = max_dps
        self.offset = [0.0] * pin_num
        self.direction = [1] * pin_num
        self.origin_positions = [0.0] * pin_num
        self.calibrate_position = [0.0] * pin_num
        if init_angles is None:
            init_angles = [0.0] * pin_num
        elif len(init_angles) != pin_num:
            raise ValueError("init angles count does not match pin count")
        # current servo positions (the live state the renderer reads)
        self.servo_positions = [float(a) for a in init_angles]

    # -- raw writes (instant) -----------------------------------------
    def servo_write_raw(self, angle_list):
        for i in range(self.pin_num):
            self.servo_positions[i] = float(angle_list[i])

    def servo_write_all(self, angles):
        rel = []
        for i in range(self.pin_num):
            rel.append(
                self.direction[i] * (self.origin_positions[i] + angles[i] + self.offset[i]))
        self.servo_write_raw(rel)

    # -- interpolated move (blocks, real-time) ------------------------
    def servo_move(self, targets, speed=50, bpm=None):
        speed = max(0, min(100, speed))
        step_time = 10  # ms per step
        delta = []
        absdelta = []
        for i in range(self.pin_num):
            value = targets[i] - self.servo_positions[i]
            delta.append(value)
            absdelta.append(abs(value))

        max_delta = max(absdelta) if absdelta else 0
        if max_delta < 1e-9:
            sleep(step_time / 1000.0)
            return

        if bpm:
            total_time = 60 / bpm * 1000
        else:
            total_time = -9.9 * speed + 1000  # ms

        current_max_dps = max_delta / total_time * 1000
        if current_max_dps > self.max_dps:
            total_time = max_delta / self.max_dps * 1000

        max_step = max(1, int(total_time / step_time))
        steps = [float(delta[i]) / max_step for i in range(self.pin_num)]

        for _ in range(max_step):
            start = time()
            for j in range(self.pin_num):
                self.servo_positions[j] += steps[j]
            self.servo_write_all(self.servo_positions)
            delay = step_time / 1000.0 - (time() - start)
            if delay > 0:
                sleep(delay)

    # -- calibration helpers ------------------------------------------
    def set_offset(self, offset_list):
        offset_list = [min(max(o, -20), 20) for o in offset_list]
        self.offset = [float(o) for o in offset_list]

    def reset(self, lst=None):
        if lst is None:
            self.servo_positions = [0.0] * self.pin_num
        else:
            self.servo_positions = [float(a) for a in lst]
        self.servo_write_all(self.servo_positions)


# ----------------------------------------------------------------------
# Emulator Pidog
# ----------------------------------------------------------------------
class Pidog:
    """Software emulator of ``pidog.pidog.Pidog``.

    The public API matches the real class so existing scripts run with a
    one-line import change.  Hardware-only features (IMU, ultrasonic, RGB
    strip, sound) are stubbed so they never crash.
    """

    # structure constants (mirrors the real Pidog)
    LEG = 42
    FOOT = 76
    BODY_LENGTH = 117
    BODY_WIDTH = 98
    HEAD_DPS = 300
    LEGS_DPS = 428
    TAIL_DPS = 500
    DEFAULT_LEGS_PINS = [2, 3, 7, 8, 0, 1, 10, 11]
    DEFAULT_HEAD_PINS = [4, 6, 5]
    DEFAULT_TAIL_PIN = [9]
    HEAD_PITCH_OFFSET = 45
    HEAD_YAW_MIN = -90
    HEAD_YAW_MAX = 90
    HEAD_ROLL_MIN = -70
    HEAD_ROLL_MAX = 70
    HEAD_PITCH_MIN = -45
    HEAD_PITCH_MAX = 30

    @classmethod
    def legs_angle_calculation(cls, coords):
        return Kinematics.legs_angle_calculation(coords)

    def __init__(self, leg_pins=DEFAULT_LEGS_PINS, head_pins=DEFAULT_HEAD_PINS,
                 tail_pin=DEFAULT_TAIL_PIN, leg_init_angles=None,
                 head_init_angles=None, tail_init_angle=None,
                 start_threads=True):
        self.actions_dict = ActionDict()

        self.body_height = 80
        self.pose = numpy_mat([0.0, 0.0, self.body_height]).T
        self.rpy = np.array([0.0, 0.0, 0.0]) * np.pi / 180
        self.leg_point_struc = numpy_mat([
            [-self.BODY_WIDTH / 2, -self.BODY_LENGTH / 2, 0],
            [self.BODY_WIDTH / 2, -self.BODY_LENGTH / 2, 0],
            [-self.BODY_WIDTH / 2, self.BODY_LENGTH / 2, 0],
            [self.BODY_WIDTH / 2, self.BODY_LENGTH / 2, 0]
        ]).T
        self.pitch = 0.0
        self.roll = 0.0
        self.target_rpy = [0, 0, 0]

        if leg_init_angles is None:
            leg_init_angles = self.actions_dict['lie'][0][0]
        if head_init_angles is None:
            head_init_angles = [0, 0, self.HEAD_PITCH_OFFSET]
        else:
            head_init_angles = list(head_init_angles)
            head_init_angles[2] += self.HEAD_PITCH_OFFSET
        if tail_init_angle is None:
            tail_init_angle = [0]

        # virtual servo groups (the live state the renderer reads)
        self.legs = _ServoGroup('legs', len(leg_pins), leg_init_angles, self.LEGS_DPS)
        self.head = _ServoGroup('head', len(head_pins), head_init_angles, self.HEAD_DPS)
        self.tail = _ServoGroup('tail', len(tail_pin), tail_init_angle, self.TAIL_DPS)

        self.legs_action_buffer = []
        self.head_action_buffer = []
        self.tail_action_buffer = []
        self.legs_thread_lock = threading.Lock()
        self.head_thread_lock = threading.Lock()
        self.tail_thread_lock = threading.Lock()

        self.leg_current_angles = list(leg_init_angles)
        self.head_current_angles = list(head_init_angles)
        self.tail_current_angles = list(tail_init_angle)

        self.legs_speed = 90
        self.head_speed = 90
        self.tail_speed = 90

        # bookkeeping for the renderer / introspection
        self.last_action = None
        self.last_part = None

        self.exit_flag = False
        self.thread_list = ["legs", "head", "tail"]
        if start_threads:
            self.action_threads_start()

    # ------------------------------------------------------------------
    # thread management
    # ------------------------------------------------------------------
    def action_threads_start(self):
        if 'legs' in self.thread_list:
            self.legs_thread = threading.Thread(name='legs_thread',
                                                target=self._legs_action_thread, daemon=True)
            self.legs_thread.start()
        if 'head' in self.thread_list:
            self.head_thread = threading.Thread(name='head_thread',
                                                target=self._head_action_thread, daemon=True)
            self.head_thread.start()
        if 'tail' in self.thread_list:
            self.tail_thread = threading.Thread(name='tail_thread',
                                                target=self._tail_action_thread, daemon=True)
            self.tail_thread.start()

    def _legs_action_thread(self):
        while not self.exit_flag:
            try:
                with self.legs_thread_lock:
                    frame = list(self.legs_action_buffer[0])
                self.leg_current_angles = list(frame)
                self.legs.servo_move(frame, self.legs_speed)
                with self.legs_thread_lock:
                    self.legs_action_buffer.pop(0)
            except IndexError:
                sleep(0.001)
            except Exception as e:  # pragma: no cover - defensive
                sys.stderr.write(f"_legs_action_thread Exception:{e}\n")
                break

    def _head_action_thread(self):
        while not self.exit_flag:
            try:
                with self.head_thread_lock:
                    frame = list(self.head_action_buffer[0])
                    self.head_action_buffer.pop(0)
                self.head_current_angles = list(frame)
                _a = list(frame)
                _a[0] = Kinematics.limit(self.HEAD_YAW_MIN, self.HEAD_YAW_MAX, _a[0])
                _a[1] = Kinematics.limit(self.HEAD_ROLL_MIN, self.HEAD_ROLL_MAX, _a[1])
                _a[2] = Kinematics.limit(self.HEAD_PITCH_MIN, self.HEAD_PITCH_MAX, _a[2])
                _a[2] += self.HEAD_PITCH_OFFSET
                self.head.servo_move(_a, self.head_speed)
            except IndexError:
                sleep(0.001)
            except Exception as e:  # pragma: no cover - defensive
                sys.stderr.write(f"_head_action_thread Exception:{e}\n")
                break

    def _tail_action_thread(self):
        while not self.exit_flag:
            try:
                with self.tail_thread_lock:
                    frame = list(self.tail_action_buffer[0])
                    self.tail_action_buffer.pop(0)
                self.tail_current_angles = list(frame)
                self.tail.servo_move(frame, self.tail_speed)
            except IndexError:
                sleep(0.001)
            except Exception as e:  # pragma: no cover - defensive
                sys.stderr.write(f"_tail_action_thread Exception:{e}\n")
                break

    # ------------------------------------------------------------------
    # stop / close
    # ------------------------------------------------------------------
    def close_all_thread(self):
        self.exit_flag = True

    def close(self):
        self.exit_flag = False
        self.action_threads_start()
        self.stop_and_lie()
        self.close_all_thread()
        try:
            self.legs_thread.join(timeout=2)
            self.head_thread.join(timeout=2)
            self.tail_thread.join(timeout=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # direct movement API
    # ------------------------------------------------------------------
    def legs_simple_move(self, angles_list, speed=90):
        tt = time()
        max_delay = 0.05
        min_delay = 0.005
        speed = max(0, min(100, speed))
        delay = (100 - speed) / 100 * (max_delay - min_delay) + min_delay

        rel_angles_list = [angles_list[i] + self.legs.offset[i]
                           for i in range(len(angles_list))]
        self.legs.servo_write_raw(rel_angles_list)
        delay2 = 0.001 * len(angles_list) - (time() - tt)
        if delay2 < -delay:
            delay2 = -delay
        sleep(delay + delay2)

    def legs_switch(self, flag=False):
        self.legs_sw_flag = flag

    def legs_move(self, target_angles, immediately=True, speed=50):
        if immediately:
            self.legs_stop()
        self.legs_speed = speed
        with self.legs_thread_lock:
            self.legs_action_buffer += [list(f) for f in target_angles]

    def head_rpy_to_angle(self, target_yrp, roll_comp=0, pitch_comp=0):
        return Kinematics.head_rpy_to_angle(target_yrp, roll_comp, pitch_comp)

    def head_move(self, target_yrps, roll_comp=0, pitch_comp=0,
                  immediately=True, speed=50):
        if immediately:
            self.head_stop()
        self.head_speed = speed
        angles = [self.head_rpy_to_angle(yrp, roll_comp, pitch_comp)
                  for yrp in target_yrps]
        with self.head_thread_lock:
            self.head_action_buffer += angles
        self.last_action = 'head'
        self.last_part = 'head'

    def head_move_raw(self, target_angles, immediately=True, speed=50):
        if immediately:
            self.head_stop()
        self.head_speed = speed
        with self.head_thread_lock:
            self.head_action_buffer += [list(a) for a in target_angles]
        self.last_action = 'head'
        self.last_part = 'head'

    def tail_move(self, target_angles, immediately=True, speed=50):
        if immediately:
            self.tail_stop()
        self.tail_speed = speed
        with self.tail_thread_lock:
            self.tail_action_buffer += [list(a) for a in target_angles]
        self.last_action = 'tail'
        self.last_part = 'tail'

    # ------------------------------------------------------------------
    # stop / wait helpers
    # ------------------------------------------------------------------
    def legs_stop(self):
        with self.legs_thread_lock:
            self.legs_action_buffer.clear()
        self.wait_legs_done()

    def head_stop(self):
        with self.head_thread_lock:
            self.head_action_buffer.clear()
        self.wait_head_done()

    def tail_stop(self):
        with self.tail_thread_lock:
            self.tail_action_buffer.clear()
        self.wait_tail_done()

    def body_stop(self):
        self.legs_stop()
        self.head_stop()
        self.tail_stop()

    def wait_legs_done(self):
        while not self.is_legs_done():
            sleep(0.001)

    def wait_head_done(self):
        while not self.is_head_done():
            sleep(0.001)

    def wait_tail_done(self):
        while not self.is_tail_done():
            sleep(0.001)

    def wait_all_done(self):
        self.wait_legs_done()
        self.wait_head_done()
        self.wait_tail_done()

    def is_legs_done(self):
        return len(self.legs_action_buffer) == 0

    def is_head_done(self):
        return len(self.head_action_buffer) == 0

    def is_tail_done(self):
        return len(self.tail_action_buffer) == 0

    def is_all_done(self):
        return self.is_legs_done() and self.is_head_done() and self.is_tail_done()

    # ------------------------------------------------------------------
    # preset action
    # ------------------------------------------------------------------
    def do_action(self, action_name, step_count=1, speed=50, pitch_comp=0):
        try:
            actions, part = self.actions_dict[action_name]
            self.last_action = action_name
            self.last_part = part
            if part == 'legs':
                for _ in range(step_count):
                    self.legs_move(actions, immediately=False, speed=speed)
            elif part == 'head':
                for _ in range(step_count):
                    self.head_move(actions, pitch_comp=pitch_comp,
                                   immediately=False, speed=speed)
            elif part == 'tail':
                for _ in range(step_count):
                    self.tail_move(actions, immediately=False, speed=speed)
        except KeyError:
            sys.stderr.write(f"do_action: No such action '{action_name}'\n")
        except Exception as e:
            sys.stderr.write(f"do_action:{e}\n")

    def stop_and_lie(self, speed=85):
        self.body_stop()
        self.legs_move(self.actions_dict['lie'][0], speed)
        self.head_move_raw([[0, 0, 0]], speed)
        self.tail_move([[0, 0, 0]], speed)
        self.wait_all_done()

    # ------------------------------------------------------------------
    # pose / kinematics API (mirrors real Pidog)
    # ------------------------------------------------------------------
    def set_pose(self, x=None, y=None, z=None):
        if x is not None:
            self.pose[0, 0] = float(x)
        if y is not None:
            self.pose[1, 0] = float(y)
        if z is not None:
            self.pose[2, 0] = float(z)

    def set_rpy(self, roll=None, pitch=None, yaw=None, pid=False):
        if roll is None:
            roll = self.rpy[0]
        if pitch is None:
            pitch = self.rpy[1]
        if yaw is None:
            yaw = self.rpy[2]
        # pid balancing is a no-op in the emulator (no IMU)
        self.rpy[0] = roll / 180. * np.pi
        self.rpy[1] = pitch / 180. * np.pi
        self.rpy[2] = yaw / 180. * np.pi

    def set_legs(self, legs_list):
        self.leg_point_struc = numpy_mat([
            [-self.BODY_WIDTH / 2, -self.BODY_LENGTH / 2 + legs_list[0][0],
             self.body_height - legs_list[0][1]],
            [self.BODY_WIDTH / 2, -self.BODY_LENGTH / 2 + legs_list[1][0],
             self.body_height - legs_list[1][1]],
            [-self.BODY_WIDTH / 2, self.BODY_LENGTH / 2 + legs_list[2][0],
             self.body_height - legs_list[2][1]],
            [self.BODY_WIDTH / 2, self.BODY_LENGTH / 2 + legs_list[3][0],
             self.body_height - legs_list[3][1]]
        ]).T

    def pose2coords(self):
        return Kinematics.pose2coords(self.pose, self.rpy,
                                      self.leg_point_struc, self.body_height)

    def pose2legs_angle(self):
        return Kinematics.pose2legs_angle(self.pose, self.rpy,
                                           self.leg_point_struc, self.body_height)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------
    def set_leg_offsets(self, cali_list, reset_list=None):
        self.legs.set_offset(cali_list)
        if reset_list is None:
            self.legs.reset()
            self.leg_current_angles = [0] * 8
        else:
            self.legs.servo_positions = list.copy(reset_list)
            self.leg_current_angles = list.copy(reset_list)
            self.legs.servo_write_all(reset_list)

    def set_head_offsets(self, cali_list):
        self.head.set_offset(cali_list)
        self.head_move([[0] * 3], immediately=True, speed=80)
        self.head_current_angles = [0] * 3

    def set_tail_offset(self, cali_list):
        self.tail.set_offset(cali_list)
        self.tail.reset()
        self.tail_current_angles = [0]

    # ------------------------------------------------------------------
    # sensor / peripheral stubs (so example scripts don't crash)
    # ------------------------------------------------------------------
    def read_distance(self):
        return -1.0

    def read_battery_voltage(self):
        return 8.0

    def speak(self, name, volume=100):
        return False

    def speak_block(self, name, volume=100):
        return False

    # ------------------------------------------------------------------
    # convenience for the renderer
    # ------------------------------------------------------------------
    def get_state(self):
        """Return a snapshot of the live servo angles for the renderer."""
        return {
            'legs': list(self.legs.servo_positions),
            'head': list(self.head.servo_positions),
            'tail': list(self.tail.servo_positions),
            'legs_buffer': len(self.legs_action_buffer),
            'head_buffer': len(self.head_action_buffer),
            'tail_buffer': len(self.tail_action_buffer),
            'last_action': self.last_action,
            'last_part': self.last_part,
        }
