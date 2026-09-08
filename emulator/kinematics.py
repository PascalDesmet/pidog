#!/usr/bin/env python3
"""Pure-math kinematics for the PiDog emulator.

This module is a self-contained re-implementation of the kinematics used by the
real ``pidog.pidog.Pidog`` class (``coord2polar``, ``fieldcoord2polar``,
``legs_angle_calculation``, ``pose2coords``, ``pose2legs_angle``) together with
the gait generators ``Walk`` and ``Trot``.  Everything here is pure Python +
numpy and requires no hardware.

It also provides :func:`angles_to_leg_coord`, a *forward* kinematics helper that
inverts ``coord2polar`` so the renderer can draw a 2-link stick figure of each
leg from the live servo angles.  The forward kinematics is verified to round
trip with ``coord2polar`` (see :func:`self_test`).
"""

from math import pi, sin, cos, sqrt, acos, atan2, radians, degrees

import numpy as np


def numpy_mat(data):
    """Return a matrix view of ``data`` (mirrors the helper used in pidog.py)."""
    return np.asmatrix(data)


class Kinematics:
    """Structure constants and kinematics methods (mirrors ``Pidog`` class)."""

    LEG = 42          # upper leg length (mm)
    FOOT = 76         # lower leg length (mm)
    BODY_LENGTH = 117
    BODY_WIDTH = 98
    BODY_STRUCT = numpy_mat([
        [-BODY_WIDTH / 2, -BODY_LENGTH / 2, 0],
        [BODY_WIDTH / 2, -BODY_LENGTH / 2, 0],
        [-BODY_WIDTH / 2, BODY_LENGTH / 2, 0],
        [BODY_WIDTH / 2, BODY_LENGTH / 2, 0]]).T

    HEAD_PITCH_OFFSET = 45
    HEAD_YAW_MIN = -90
    HEAD_YAW_MAX = 90
    HEAD_ROLL_MIN = -70
    HEAD_ROLL_MAX = 70
    HEAD_PITCH_MIN = -45
    HEAD_PITCH_MAX = 30

    # ------------------------------------------------------------------
    # angle <-> coordinate conversions (copied from pidog.py)
    # ------------------------------------------------------------------
    @staticmethod
    def coord2polar(coord):
        y, z = coord
        u = sqrt(pow(y, 2) + pow(z, 2))
        cos_angle1 = (Kinematics.FOOT ** 2 + Kinematics.LEG ** 2 - u ** 2) / \
            (2 * Kinematics.FOOT * Kinematics.LEG)
        cos_angle1 = min(max(cos_angle1, -1), 1)
        beta = acos(cos_angle1)

        angle1 = atan2(y, z)
        cos_angle2 = (Kinematics.LEG ** 2 + u ** 2 - Kinematics.FOOT ** 2) / \
            (2 * Kinematics.LEG * u)
        cos_angle2 = min(max(cos_angle2, -1), 1)
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1

        alpha = alpha / pi * 180
        beta = beta / pi * 180
        return alpha, beta

    @staticmethod
    def fieldcoord2polar(coord, pitch_rad=0.0):
        y, z = coord
        u = sqrt(pow(y, 2) + pow(z, 2))
        cos_angle1 = (Kinematics.FOOT ** 2 + Kinematics.LEG ** 2 - u ** 2) / \
            (2 * Kinematics.FOOT * Kinematics.LEG)
        cos_angle1 = min(max(cos_angle1, -1), 1)
        beta = acos(cos_angle1)

        angle1 = atan2(y, z)
        cos_angle2 = (Kinematics.LEG ** 2 + u ** 2 - Kinematics.FOOT ** 2) / \
            (2 * Kinematics.LEG * u)
        cos_angle2 = min(max(cos_angle2, -1), 1)
        angle2 = acos(cos_angle2)
        alpha = angle2 + angle1 + pitch_rad

        alpha = alpha / pi * 180
        beta = beta / pi * 180
        return alpha, beta

    @classmethod
    def legs_angle_calculation(cls, coords):
        """Convert 4 leg [y, z] coords into 8 servo angles.

        Mirrors ``Pidog.legs_angle_calculation``.  ``coords`` is a list of 4
        [y, z] pairs (left-front, right-front, left-hind, right-hind).
        """
        translate_list = []
        for i, coord in enumerate(coords):
            leg_angle, foot_angle = cls.coord2polar(coord)
            foot_angle = foot_angle - 90
            if i % 2 != 0:  # right side servos are mirrored
                leg_angle = -leg_angle
                foot_angle = -foot_angle
            translate_list += [leg_angle, foot_angle]
        return translate_list

    # ------------------------------------------------------------------
    # pose / Euler-angle -> leg coordinates (copied from pidog.py)
    # ------------------------------------------------------------------
    @classmethod
    def pose2coords(cls, pose, rpy, leg_point_struc, body_height):
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        rotx = numpy_mat([
            [cos(roll), 0, -sin(roll)],
            [0, 1, 0],
            [sin(roll), 0, cos(roll)]])
        roty = numpy_mat([
            [1, 0, 0],
            [0, cos(-pitch), -sin(-pitch)],
            [0, sin(-pitch), cos(-pitch)]])
        rotz = numpy_mat([
            [cos(yaw), -sin(yaw), 0],
            [sin(yaw), cos(yaw), 0],
            [0, 0, 1]])
        rot_mat = rotx * roty * rotz
        AB = numpy_mat(np.zeros((3, 4)))
        pose_vec = numpy_mat([[float(pose[0])], [float(pose[1])], [float(pose[2])]])
        for i in range(4):
            AB[:, i] = -pose_vec - rot_mat * cls.BODY_STRUCT[:, i] + leg_point_struc[:, i]

        body_coor_list = []
        for i in range(4):
            body_coor_list.append([(leg_point_struc - AB).T[i, 0],
                                   (leg_point_struc - AB).T[i, 1],
                                   (leg_point_struc - AB).T[i, 2]])

        leg_coor_list = []
        for i in range(4):
            leg_coor_list.append(
                [leg_point_struc.T[i, 0], leg_point_struc.T[i, 1], leg_point_struc.T[i, 2]])

        return {"leg": leg_coor_list, "body": body_coor_list}

    @classmethod
    def pose2legs_angle(cls, pose, rpy, leg_point_struc, body_height):
        data = cls.pose2coords(pose, rpy, leg_point_struc, body_height)
        leg_coor_list = data["leg"]
        body_coor_list = data["body"]
        coords = []
        for i in range(4):
            coords.append([
                leg_coor_list[i][1] - body_coor_list[i][1],
                body_coor_list[i][2] - leg_coor_list[i][2]])

        angles = []
        for i, coord in enumerate(coords):
            leg_angle, foot_angle = cls.fieldcoord2polar(coord, rpy[1])
            foot_angle = foot_angle - 90
            if i % 2 != 0:
                leg_angle = -leg_angle
                foot_angle = -foot_angle
            angles += [leg_angle, foot_angle]
        return angles

    # ------------------------------------------------------------------
    # head rpy -> servo angles (copied from pidog.py)
    # ------------------------------------------------------------------
    @staticmethod
    def head_rpy_to_angle(target_yrp, roll_comp=0, pitch_comp=0):
        yaw, roll, pitch = target_yrp
        signed = -1 if yaw < 0 else 1
        ratio = abs(yaw) / 90
        pitch_servo = roll * ratio + pitch * (1 - ratio) + pitch_comp
        roll_servo = -(signed * (roll * (1 - ratio) + pitch * ratio) + roll_comp)
        yaw_servo = yaw
        return [yaw_servo, roll_servo, pitch_servo]

    # ------------------------------------------------------------------
    # forward kinematics for the stick-figure renderer
    # ------------------------------------------------------------------
    @staticmethod
    def angles_to_leg_coord(leg_angle, foot_angle, is_right):
        """Forward kinematics: servo angles -> foot (y, z) relative to the hip.

        Inverts :meth:`coord2polar`; guaranteed to round trip (see
        :func:`self_test`).  ``y`` is the fore/aft offset, ``z`` is the vertical
        offset (positive = up) of the foot relative to the hip.

        Returns ``(foot_y, foot_z, knee_y, knee_z)``.
        """
        if is_right:
            alpha = -leg_angle
            beta = 90 - foot_angle
        else:
            alpha = leg_angle
            beta = foot_angle + 90

        alpha_rad = radians(alpha)
        beta_rad = radians(beta)

        u = sqrt(Kinematics.LEG ** 2 + Kinematics.FOOT ** 2
                 - 2 * Kinematics.LEG * Kinematics.FOOT * cos(beta_rad))
        if u < 1e-6:
            return 0.0, 0.0, 0.0, 0.0
        cos_a2 = (Kinematics.LEG ** 2 + u ** 2 - Kinematics.FOOT ** 2) / \
            (2 * Kinematics.LEG * u)
        cos_a2 = min(max(cos_a2, -1), 1)
        angle2 = acos(cos_a2)
        angle1 = alpha_rad - angle2

        foot_y = u * sin(angle1)
        foot_z = u * cos(angle1)
        knee_y = Kinematics.LEG * sin(alpha_rad)
        knee_z = Kinematics.LEG * cos(alpha_rad)
        return foot_y, foot_z, knee_y, knee_z

    @staticmethod
    def limit(lo, hi, x):
        if x > hi:
            return hi
        elif x < lo:
            return lo
        return x


def _norm_angle(d):
    """Normalize an angle difference to [-180, 180]."""
    return ((d + 180.0) % 360.0) - 180.0


def self_test():
    """Verify angles_to_leg_coord round trips through coord2polar."""
    ok = True
    for is_right in (False, True):
        for leg_angle in (-80, -45, -20, 0, 25, 45, 80):
            for foot_angle in (-70, -45, -20, 0, 30, 60):
                fy, fz, ky, kz = Kinematics.angles_to_leg_coord(
                    leg_angle, foot_angle, is_right)
                a, b = Kinematics.coord2polar([fy, fz])
                fa = b - 90
                la = a
                if is_right:
                    la = -a
                    fa = -fa
                if abs(_norm_angle(la - leg_angle)) > 1e-6 or abs(fa - foot_angle) > 1e-6:
                    print(f"ROUND-TRIP FAIL right={is_right} "
                          f"leg={leg_angle} foot={foot_angle} -> "
                          f"la={la:.4f} fa={fa:.4f}")
                    ok = False
    # also verify legs_angle_calculation inverts cleanly for known coords
    coords = [[-15, 95], [-15, 95], [5, 90], [5, 90]]
    ang = Kinematics.legs_angle_calculation(coords)
    back = []
    for i in range(4):
        fy, fz, ky, kz = Kinematics.angles_to_leg_coord(
            ang[i * 2], ang[i * 2 + 1], is_right=(i % 2 == 1))
        back.append([round(fy, 4), round(fz, 4)])
    for c1, c2 in zip(coords, back):
        if abs(c1[0] - c2[0]) > 1e-3 or abs(c1[1] - c2[1]) > 1e-3:
            print(f"COORD ROUND-TRIP FAIL {c1} != {c2}")
            ok = False
    print("kinematics self-test:", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------
# Gait generators (copied from walk.py / trot.py, pure math)
# ----------------------------------------------------------------------
class Walk:
    FORWARD = 1
    BACKWARD = -1
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    SECTION_COUNT = 8
    STEP_COUNT = 6
    LEG_ORDER = [1, 0, 4, 0, 2, 0, 3, 0]
    LEG_STEP_HEIGHT = 20
    LEG_STEP_WIDTH = 80
    CENTER_OF_GRAVIRTY = -15
    LEG_POSITION_OFFSETS = [-10, -10, 20, 20]
    Z_ORIGIN = 80
    TURNING_RATE = 0.3
    LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
    LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
    LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
    LEG_ORIGINAL_Y_TABLE = [0, 2, 3, 1]
    LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT,
                      LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]

    def __init__(self, fb, lr):
        self.fb = fb
        self.lr = lr
        if self.fb == self.FORWARD:
            self.y_offset = 0 + self.CENTER_OF_GRAVIRTY
        elif self.fb == self.BACKWARD:
            self.y_offset = 0 + self.CENTER_OF_GRAVIRTY
        else:
            self.y_offset = self.CENTER_OF_GRAVIRTY
        self.leg_step_width = [
            self.LEG_STEP_WIDTH * self.LEG_STEP_SCALES[self.lr + 1][i] for i in range(4)]
        self.section_length = [self.leg_step_width[i] /
                               (self.SECTION_COUNT - 1) for i in range(4)]
        self.step_down_length = [
            self.section_length[i] / self.STEP_COUNT for i in range(4)]
        self.leg_origin = [self.leg_step_width[i] / 2 + self.y_offset + (
            self.LEG_POSITION_OFFSETS[i] * self.LEG_STEP_SCALES[self.lr + 1][i])
            for i in range(4)]

    def step_y_func(self, leg, step):
        theta = step * pi / (self.STEP_COUNT - 1)
        temp = (self.leg_step_width[leg] *
                (cos(theta) - self.fb) / 2 * self.fb)
        return self.leg_origin[leg] + temp

    def step_z_func(self, step):
        return self.Z_ORIGIN - (self.LEG_STEP_HEIGHT * step / (self.STEP_COUNT - 1))

    def get_coords(self):
        origin_leg_coord = [[self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i]
                             * 2 * self.section_length[i], self.Z_ORIGIN] for i in range(4)]
        leg_coord = list.copy(origin_leg_coord)
        leg_coords = []
        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                if self.fb == 1:
                    raise_leg = self.LEG_ORDER[section]
                else:
                    raise_leg = self.LEG_ORDER[self.SECTION_COUNT - section - 1]
                for i in range(4):
                    if raise_leg != 0 and i == raise_leg - 1:
                        y = self.step_y_func(i, step)
                        z = self.step_z_func(step)
                    else:
                        y = leg_coord[i][0] + self.step_down_length[i] * self.fb
                        z = self.Z_ORIGIN
                    leg_coord[i] = [y, z]
                leg_coords.append(list.copy(leg_coord))
        leg_coords.append(origin_leg_coord)
        return leg_coords


class Trot:
    FORWARD = 1
    BACKWARD = -1
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

    SECTION_COUNT = 2
    STEP_COUNT = 3
    LEG_RAISE_ORDER = [[1, 4], [2, 3]]
    LEG_STEP_HEIGHT = 20
    LEG_STEP_WIDTH = 100
    CENTER_OF_GRAVITY = -17
    LEG_STAND_OFFSET = 5
    Z_ORIGIN = 80
    TURNING_RATE = 0.5
    LEG_STAND_OFFSET_DIRS = [-1, -1, 1, 1]
    LEG_STEP_SCALES_LEFT = [TURNING_RATE, 1, TURNING_RATE, 1]
    LEG_STEP_SCALES_MIDDLE = [1, 1, 1, 1]
    LEG_STEP_SCALES_RIGHT = [1, TURNING_RATE, 1, TURNING_RATE]
    LEG_ORIGINAL_Y_TABLE = [0, 1, 1, 0]
    LEG_STEP_SCALES = [LEG_STEP_SCALES_LEFT,
                      LEG_STEP_SCALES_MIDDLE, LEG_STEP_SCALES_RIGHT]

    def __init__(self, fb, lr):
        self.fb = fb
        self.lr = lr
        if self.fb == self.FORWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 0 + self.CENTER_OF_GRAVITY
            else:
                self.y_offset = -2 + self.CENTER_OF_GRAVITY
        elif self.fb == self.BACKWARD:
            if self.lr == self.STRAIGHT:
                self.y_offset = 8 + self.CENTER_OF_GRAVITY
            else:
                self.y_offset = 1 + self.CENTER_OF_GRAVITY
        else:
            self.y_offset = self.CENTER_OF_GRAVITY
        self.leg_step_width = [
            self.LEG_STEP_WIDTH * self.LEG_STEP_SCALES[self.lr + 1][i] for i in range(4)]
        self.section_length = [self.leg_step_width[i] /
                               (self.SECTION_COUNT - 1) for i in range(4)]
        self.step_down_length = [
            self.section_length[i] / self.STEP_COUNT for i in range(4)]
        self.leg_offset = [self.LEG_STAND_OFFSET *
                           self.LEG_STAND_OFFSET_DIRS[i] for i in range(4)]
        self.leg_origin = [self.leg_step_width[i] / 2 + self.y_offset + (
            self.leg_offset[i] * self.LEG_STEP_SCALES[self.lr + 1][i]) for i in range(4)]

    def step_y_func(self, leg, step):
        theta = step * pi / (self.STEP_COUNT - 1)
        temp = (self.leg_step_width[leg] *
                (cos(theta) - self.fb) / 2 * self.fb)
        return self.leg_origin[leg] + temp

    def step_z_func(self, step):
        return self.Z_ORIGIN - (self.LEG_STEP_HEIGHT * step / (self.STEP_COUNT - 1))

    def get_coords(self):
        origin_leg_coord = [[self.leg_origin[i] - self.LEG_ORIGINAL_Y_TABLE[i]
                             * self.section_length[i], self.Z_ORIGIN] for i in range(4)]
        leg_coords = []
        for section in range(self.SECTION_COUNT):
            for step in range(self.STEP_COUNT):
                if self.fb == 1:
                    raise_legs = self.LEG_RAISE_ORDER[section]
                else:
                    raise_legs = self.LEG_RAISE_ORDER[self.SECTION_COUNT - section - 1]
                leg_coord = []
                for i in range(4):
                    if i + 1 in raise_legs:
                        y = self.step_y_func(i, step)
                        z = self.step_z_func(step)
                    else:
                        y = origin_leg_coord[i][0] + self.step_down_length[i] * self.fb
                        z = self.Z_ORIGIN
                    leg_coord.append([y, z])
                origin_leg_coord = leg_coord
                leg_coords.append(leg_coord)
        return leg_coords


if __name__ == "__main__":
    self_test()
