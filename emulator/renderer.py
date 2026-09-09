#!/usr/bin/env python3
"""Pygame renderer for the PiDog emulator.

The renderer runs in the main thread (pygame requires it) and polls the
:class:`~emulator.pidog_emu.Pidog` virtual servo state at ~60 fps.  It displays:

* the **body pose** as a photographic background, cross-faded between the two
  nearest of the four reference poses (lie / half_sit / sit / stand) based on
  the live leg angles, for the currently selected camera view;
* a **stick-figure overlay** drawn from the live 12 servo angles (precise
  sagittal kinematics) so you can see the exact motion the code produces;
* a **side panel** with thumbnails of all images for the current view, the
  closest body pose / head / tail reference highlighted;
* a **HUD** with numeric servo angles, buffers, current action and view.

Image filename convention (in ``emulator-images/``):

    body_<pose>_<view>.png      pose: lie half_sit sit stand   (required)
    head_<pos>_<view>.png       pos:  center left right up down (optional)
    tail_<pos>_<view>.png       pos:  center left right         (optional)

    view: front left right rear

If images are missing the renderer still works, showing coloured placeholders
and the stick overlay, so it can be used before any photos are supplied.

Keys
----
    1 / 2 / 3 / 4   front / left / right / rear view
    Arrow Left/Right also switch views
    S               toggle stick overlay
    H               toggle HUD
    T               toggle side panel (thumbnails)
    + / -           zoom the stick overlay
    P               pause rendering (emulator keeps running)
    ESC             quit
"""

import os
import math
import threading

import pygame

from .kinematics import Kinematics

# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "emulator-images")

VIEWS = ["front", "left", "right", "rear"]
DEFAULT_VIEW = "left"

BODY_POSES = ["lie", "half_sit", "sit", "stand"]
HEAD_POS = {
    "center": (0, 0, 0),
    "left": (-45, 0, 0),
    "right": (45, 0, 0),
    "up": (0, 0, -30),
    "down": (0, 0, 20),
}
TAIL_POS = {"center": 0, "left": -30, "right": 30}

# reference leg-angle frames for the four body poses (from the action dict)
def _reference_leg_angles():
    from .actions import ActionDict
    ad = ActionDict()
    return {
        "lie": list(ad["lie"][0][0]),
        "half_sit": list(ad["half_sit"][0][0]),
        "sit": list(ad["sit"][0][0]),
        "stand": list(ad["stand"][0][0]),
    }


# ----------------------------------------------------------------------
# image store
# ----------------------------------------------------------------------
class ImageStore:
    """Loads (or generates placeholders for) all emulator images."""

    def __init__(self, images_dir=IMAGES_DIR, view_size=(560, 420),
                 thumb_size=(120, 90)):
        self.images_dir = images_dir
        self.view_size = view_size
        self.thumb_size = thumb_size
        self.body = {}      # (pose, view) -> Surface (view_size)
        self.body_thumb = {}
        self.head = {}      # (pos, view) -> Surface
        self.tail = {}
        self.head_thumb = {}
        self.tail_thumb = {}
        self.have_body = False
        self.have_head = False
        self.have_tail = False
        self._load()

    def _path(self, name):
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            p = os.path.join(self.images_dir, name + ext)
            if os.path.isfile(p):
                return p
        return None

    def _placeholder(self, size, color, label):
        s = pygame.Surface(size).convert()
        s.fill(color)
        font = pygame.font.SysFont("monospace", 16)
        text = font.render(label, True, (255, 255, 255))
        s.blit(text, text.get_rect(center=(size[0] // 2, size[1] // 2)))
        return s

    def _load_img(self, name, size, fallback_color, fallback_label):
        path = self._path(name)
        if path:
            try:
                img = pygame.image.load(path).convert_alpha()
                img = pygame.transform.smoothscale(img, size)
                return img, True
            except Exception as e:
                print(f"[emulator] failed to load {path}: {e}")
        return self._placeholder(size, fallback_color, fallback_label), False

    def _load(self):
        pose_colors = {
            "lie": (60, 60, 70),
            "half_sit": (70, 50, 60),
            "sit": (50, 60, 80),
            "stand": (40, 80, 70),
        }
        head_color = (80, 80, 50)
        tail_color = (80, 50, 50)
        for pose in BODY_POSES:
            for view in VIEWS:
                img, ok = self._load_img(f"body_{pose}_{view}", self.view_size,
                                         pose_colors[pose], f"{pose}\n{view}")
                self.body[(pose, view)] = img
                self.have_body = self.have_body or ok
                t, _ = self._load_img(f"body_{pose}_{view}", self.thumb_size,
                                      pose_colors[pose], pose)
                self.body_thumb[(pose, view)] = t
        for pos in HEAD_POS:
            for view in VIEWS:
                img, ok = self._load_img(f"head_{pos}_{view}", self.view_size,
                                         head_color, f"head {pos}\n{view}")
                self.head[(pos, view)] = img
                self.have_head = self.have_head or ok
                t, _ = self._load_img(f"head_{pos}_{view}", self.thumb_size,
                                      head_color, pos)
                self.head_thumb[(pos, view)] = t
        for pos in TAIL_POS:
            for view in VIEWS:
                img, ok = self._load_img(f"tail_{pos}_{view}", self.view_size,
                                         tail_color, f"tail {pos}\n{view}")
                self.tail[(pos, view)] = img
                self.have_tail = self.have_tail or ok
                t, _ = self._load_img(f"tail_{pos}_{view}", self.thumb_size,
                                      tail_color, pos)
                self.tail_thumb[(pos, view)] = t

    # -- lookup helpers ------------------------------------------------
    def body_surface(self, pose, view):
        return self.body[(pose, view)]

    def head_surface(self, pos, view):
        if (pos, view) in self.head:
            return self.head[(pos, view)]
        # fall back to left view if missing
        if (pos, "left") in self.head:
            return self.head[(pos, "left")]
        return None

    def tail_surface(self, pos, view):
        if (pos, view) in self.tail:
            return self.tail[(pos, view)]
        if (pos, "left") in self.tail:
            return self.tail[(pos, "left")]
        return None


# ----------------------------------------------------------------------
# pose blending
# ----------------------------------------------------------------------
def pose_weights(leg_angles, reference):
    """Return normalised {pose: weight} for the two nearest body poses."""
    dists = {}
    for pose, ref in reference.items():
        d = sum((a - b) ** 2 for a, b in zip(leg_angles, ref))
        dists[pose] = d + 1e-6
    # inverse-distance, keep top 2
    ordered = sorted(dists.items(), key=lambda kv: kv[1])
    w = {p: 0.0 for p in reference}
    p1, d1 = ordered[0]
    p2, d2 = ordered[1]
    inv1, inv2 = 1.0 / d1, 1.0 / d2
    tot = inv1 + inv2
    w[p1] = inv1 / tot
    w[p2] = inv2 / tot
    return w


def closest_pose(leg_angles, reference):
    best = None
    bd = None
    for pose, ref in reference.items():
        d = sum((a - b) ** 2 for a, b in zip(leg_angles, ref))
        if bd is None or d < bd:
            bd = d
            best = pose
    return best


def closest_head(head_angles):
    """head_angles are the *commanded* [yaw, roll, pitch] (pre-offset)."""
    yaw, roll, pitch = head_angles
    best, bd = None, None
    for pos, (y, r, p) in HEAD_POS.items():
        d = (yaw - y) ** 2 + (roll - r) ** 2 + (pitch - p) ** 2
        if bd is None or d < bd:
            bd = d
            best = pos
    return best


def closest_tail(tail_angle):
    best, bd = None, None
    for pos, a in TAIL_POS.items():
        d = (tail_angle - a) ** 2
        if bd is None or d < bd:
            bd = d
            best = pos
    return best


# ----------------------------------------------------------------------
# display discovery (SSH / headless friendly)
# ----------------------------------------------------------------------
def _ensure_display():
    """Make sure pygame can open a visible window.

    When run over SSH, ``$DISPLAY`` is usually empty and SDL silently falls
    back to a dummy/offscreen driver, so ``set_mode`` succeeds but no window
    appears.  If ``$DISPLAY`` is unset we try to point it at the Pi's local
    X server (``:0``) when its socket exists.
    """
    import os
    disp = os.environ.get("DISPLAY", "").strip()
    if disp:
        return
    # look for an X server socket: /tmp/.X11-unix/X0 -> DISPLAY=:0
    sock_dir = "/tmp/.X11-unix"
    if os.path.isdir(sock_dir):
        sockets = sorted(
            f for f in os.listdir(sock_dir)
            if f.startswith("X") and f[1:].isdigit())
        if sockets:
            n = sockets[0][1:]
            candidate = f":{n}"
            os.environ["DISPLAY"] = candidate
            print(f"[emulator] DISPLAY was unset; using {candidate} "
                  f"(the Pi's local desktop). If you're connected over SSH "
                  f"with X forwarding, run `export DISPLAY=:0` first or use "
                  f"`ssh -X`.")


def _warn_if_dummy_display():
    """Print a clear warning if SDL picked a non-visible video driver."""
    try:
        name = pygame.display.get_wm_info()
        # get_wm_info is unreliable across drivers; use SDL_VIDEODRIVER env
        # and the name pygame reports via get_init is not helpful, so probe
        # by checking the window manager info / driver name where available.
    except Exception:
        pass
    import os
    drv = os.environ.get("SDL_VIDEODRIVER", "")
    # If the user forced a driver, respect it and stay quiet.
    if drv:
        return
    # Heuristic: if DISPLAY is still empty after _ensure_display, we are
    # effectively headless and SDL likely chose 'dummy' or an offscreen driver.
    if not os.environ.get("DISPLAY", "").strip():
        print("[emulator] WARNING: no display found. The window will not be "
              "visible. Options:\n"
              "  - run on the Pi's desktop (or `export DISPLAY=:0`)\n"
              "  - SSH with X forwarding: `ssh -X pds@<pi>` then run the demo\n"
              "  - set SDL_VIDEODRIVER=dummy for headless testing (no window)")


# ----------------------------------------------------------------------
# stick-figure drawing
# ----------------------------------------------------------------------
def draw_stick(surface, rect, state, scale=1.0):
    """Draw a sagittal stick figure of the dog from the live servo angles.

    ``state`` is the dict returned by :meth:`Pidog.get_state`.  The figure is
    drawn within ``rect``.  ``scale`` zooms the skeleton.
    """
    legs = state['legs']
    head = state['head']            # [yaw, roll, pitch+offset]  (raw servo)
    tail = state['tail']

    # commanded head angles (remove the pitch offset to get the logical pitch)
    head_yaw = head[0]
    head_roll = head[1]
    head_pitch = head[2] - Kinematics.HEAD_PITCH_OFFSET
    tail_angle = tail[0]

    cx = rect.centerx
    cy = rect.centery + 30 * scale  # body sits a bit below centre
    px_per_mm = 1.6 * scale

    body_len = Kinematics.BODY_LENGTH * px_per_mm
    body_w = Kinematics.BODY_WIDTH * px_per_mm * 0.35  # foreshortened width

    # body rectangle (side view): front to the right
    body_rect = pygame.Rect(0, 0, int(body_len), int(body_w))
    body_rect.center = (cx, cy)

    # hip positions (screen coords). front = +x (right), up = -y
    half_l = body_len / 2
    hip_z = cy  # hip line vertical centre
    hips = {
        'lf': (cx + half_l, hip_z),   # left front
        'rf': (cx + half_l, hip_z),   # right front (behind, drawn offset)
        'lh': (cx - half_l, hip_z),   # left hind
        'rh': (cx - half_l, hip_z),   # right hind
    }

    # leg drawing: pairs (leg_angle, foot_angle) per leg
    # order: lf, rf, lh, rh  -> indices 0,1,2,3
    leg_pairs = [(legs[0], legs[1]), (legs[2], legs[3]),
                 (legs[4], legs[5]), (legs[6], legs[7])]
    keys = ['lf', 'rf', 'lh', 'rh']
    is_right = [False, True, False, True]

    # ground estimate from average foot depth (positive z = below hip)
    foot_zs = []
    for i, (la, fa) in enumerate(leg_pairs):
        _, fz, _, _ = Kinematics.angles_to_leg_coord(la, fa, is_right[i])
        foot_zs.append(fz)
    ground_offset = (sum(foot_zs) / len(foot_zs)) * px_per_mm
    ground_y = hip_z + ground_offset  # screen y grows downward; z is down-positive

    # draw ground line
    pygame.draw.line(surface, (90, 90, 90),
                     (rect.left + 10, ground_y), (rect.right - 10, ground_y), 1)

    # draw legs (right legs slightly behind = thinner / offset)
    for i, (la, fa) in enumerate(leg_pairs):
        fy, fz, ky, kz = Kinematics.angles_to_leg_coord(la, fa, is_right[i])
        hx, hz = hips[keys[i]]
        # right legs drawn with a small lateral offset so both are visible
        xoff = -6 if is_right[i] else 6
        hip = (hx + xoff, hz)
        # z is down-positive in kinematics; screen y is down-positive too,
        # so add z (not subtract) to put feet below the hips.
        knee = (hx + xoff + ky * px_per_mm, hz + kz * px_per_mm)
        foot = (hx + xoff + fy * px_per_mm, hz + fz * px_per_mm)
        col = (255, 165, 0) if not is_right[i] else (200, 120, 0)
        width = 3 if not is_right[i] else 2
        pygame.draw.line(surface, col, hip, knee, width)
        pygame.draw.line(surface, col, knee, foot, width)
        pygame.draw.circle(surface, col, (int(foot[0]), int(foot[1])), 4)
        pygame.draw.circle(surface, (255, 255, 255), (int(knee[0]), int(knee[1])), 3)

    # body
    pygame.draw.rect(surface, (180, 180, 200), body_rect, 0, border_radius=8)
    pygame.draw.rect(surface, (120, 120, 140), body_rect, 2, border_radius=8)

    # head at the front (right side). Indicate pitch (up/down) and yaw (lateral)
    head_base = (cx + half_l, hip_z - body_w / 2)
    head_len = 46 * px_per_mm
    # pitch: negative = up (e.g. head_bark uses -40). Screen y is down-positive,
    # so head_tip_y = base_y + head_len * sin(pitch): negative pitch -> up.
    pitch_rad = math.radians(head_pitch)
    head_tip = (head_base[0] + head_len * math.cos(pitch_rad),
                head_base[1] + head_len * math.sin(pitch_rad))
    pygame.draw.line(surface, (255, 80, 80), head_base, head_tip, 4)
    pygame.draw.circle(surface, (255, 80, 80), (int(head_tip[0]), int(head_tip[1])), 6)
    # yaw indicator: a small horizontal arrow whose length = yaw
    yaw_len = head_yaw * px_per_mm * 0.8
    yaw_tip = (head_tip[0] + yaw_len, head_tip[1])
    pygame.draw.line(surface, (80, 200, 255), head_tip, yaw_tip, 2)
    # roll indicator: a short tilted segment
    roll_rad = math.radians(head_roll)
    roll_tip = (head_base[0], head_base[1] + 18 * px_per_mm * math.sin(roll_rad) - 4)
    pygame.draw.line(surface, (180, 255, 80), head_base, roll_tip, 2)

    # tail at the rear (left side). tail_angle is lateral (left/right)
    tail_base = (cx - half_l, hip_z + body_w / 4)
    tail_len = 40 * px_per_mm
    tail_rad = math.radians(tail_angle)
    # draw tail swinging left/right (horizontal) so the wag is always visible
    tail_tip = (tail_base[0] - tail_len * math.cos(tail_rad * 0.5),
                tail_base[1] + tail_len * math.sin(tail_rad))
    pygame.draw.line(surface, (255, 200, 80), tail_base, tail_tip, 4)
    pygame.draw.circle(surface, (255, 200, 80),
                       (int(tail_tip[0]), int(tail_tip[1])), 5)


# ----------------------------------------------------------------------
# renderer
# ----------------------------------------------------------------------
class Renderer:
    """Pygame window that visualises a :class:`Pidog` emulator instance."""

    def __init__(self, pidog, images_dir=IMAGES_DIR, width=1100, height=720,
                 fps=60):
        self.pidog = pidog
        self.fps = fps
        self.view = DEFAULT_VIEW
        self.view_index = VIEWS.index(DEFAULT_VIEW)
        self.show_stick = True
        self.show_hud = True
        self.show_thumbs = True
        self.paused = False
        self.stick_scale = 1.0
        self.reference = _reference_leg_angles()

        _ensure_display()
        pygame.init()
        pygame.display.set_caption("PiDog Emulator")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        _warn_if_dummy_display()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_big = pygame.font.SysFont("monospace", 18, bold=True)

        # main view area + side panel layout
        self.layout(width, height)
        self.images = ImageStore(images_dir, view_size=self.main_size,
                                 thumb_size=self.thumb_size)

    def layout(self, width, height):
        self.width = width
        self.height = height
        panel_w = 150
        self.main_rect = pygame.Rect(10, 10, width - panel_w - 30, height - 70)
        self.main_size = (self.main_rect.width, self.main_rect.height)
        self.panel_rect = pygame.Rect(width - panel_w - 10, 10, panel_w, height - 70)
        self.hud_rect = pygame.Rect(10, height - 55, width - 20, 45)
        # thumbnails grid in panel
        self.thumb_size = (panel_w - 20, int((panel_w - 20) * 0.72))
        self.thumbs_per_col = max(1, (self.panel_rect.height - 10) // (self.thumb_size[1] + 6))

    # ------------------------------------------------------------------
    def set_view(self, view):
        if view in VIEWS:
            self.view = view
            self.view_index = VIEWS.index(view)

    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key) and running
                elif event.type == pygame.VIDEORESIZE:
                    self.layout(event.w, event.h)
                    self.images = ImageStore(IMAGES_DIR, view_size=self.main_size,
                                              thumb_size=self.thumb_size)
            if not self.paused:
                self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()

    def _handle_key(self, key):
        if key == pygame.K_ESCAPE:
            return False
        elif key in (pygame.K_1, pygame.K_KP1):
            self.set_view("front")
        elif key in (pygame.K_2, pygame.K_KP2):
            self.set_view("left")
        elif key in (pygame.K_3, pygame.K_KP3):
            self.set_view("right")
        elif key in (pygame.K_4, pygame.K_KP4):
            self.set_view("rear")
        elif key == pygame.K_RIGHT:
            self.view_index = (self.view_index + 1) % len(VIEWS)
            self.set_view(VIEWS[self.view_index])
        elif key == pygame.K_LEFT:
            self.view_index = (self.view_index - 1) % len(VIEWS)
            self.set_view(VIEWS[self.view_index])
        elif key == pygame.K_s:
            self.show_stick = not self.show_stick
        elif key == pygame.K_h:
            self.show_hud = not self.show_hud
        elif key == pygame.K_t:
            self.show_thumbs = not self.show_thumbs
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.stick_scale = min(3.0, self.stick_scale + 0.1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.stick_scale = max(0.3, self.stick_scale - 0.1)
        return True

    # ------------------------------------------------------------------
    def _draw(self):
        self.screen.fill((24, 24, 28))
        state = self.pidog.get_state()
        leg_angles = state['legs']

        # --- main view: cross-faded body pose photo ---
        self._draw_main(state, leg_angles)

        # --- stick overlay ---
        if self.show_stick:
            draw_stick(self.screen, self.main_rect, state, scale=self.stick_scale)

        # --- side panel ---
        if self.show_thumbs:
            self._draw_panel(state, leg_angles)

        # --- HUD ---
        if self.show_hud:
            self._draw_hud(state)

    def _draw_main(self, state, leg_angles):
        weights = pose_weights(leg_angles, self.reference)
        ordered = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        # blit the dominant pose, then the second with alpha
        (p1, w1), (p2, w2) = ordered[0], ordered[1]
        base = self.images.body_surface(p1, self.view)
        self.screen.blit(base, self.main_rect.topleft)
        if w2 > 0.001:
            top = self.images.body_surface(p2, self.view).copy()
            top.set_alpha(int(255 * w2))
            self.screen.blit(top, self.main_rect.topleft)
        # frame
        pygame.draw.rect(self.screen, (80, 80, 90), self.main_rect, 2)
        label = self.font_big.render(
            f"view: {self.view}   pose: {p1} ({w1*100:.0f}%) + {p2} ({w2*100:.0f}%)",
            True, (255, 255, 255))
        bg = pygame.Surface((label.get_width() + 8, label.get_height() + 4))
        bg.set_alpha(160)
        bg.fill((0, 0, 0))
        self.screen.blit(bg, (self.main_rect.left + 4, self.main_rect.top + 4))
        self.screen.blit(label, (self.main_rect.left + 8, self.main_rect.top + 6))

    def _draw_panel(self, state, leg_angles):
        pygame.draw.rect(self.screen, (40, 40, 48), self.panel_rect)
        pygame.draw.rect(self.screen, (80, 80, 90), self.panel_rect, 1)
        title = self.font.render("reference", True, (200, 200, 200))
        self.screen.blit(title, (self.panel_rect.left + 8, self.panel_rect.top + 4))
        y = self.panel_rect.top + 22
        closest_b = closest_pose(leg_angles, self.reference)
        # body poses
        for pose in BODY_POSES:
            thumb = self.images.body_thumb[(pose, self.view)]
            self.screen.blit(thumb, (self.panel_rect.left + 8, y))
            if pose == closest_b:
                pygame.draw.rect(self.screen, (255, 255, 0),
                                 (self.panel_rect.left + 6, y - 2,
                                  self.thumb_size[0] + 4, self.thumb_size[1] + 4), 2)
            y += self.thumb_size[1] + 6
            if y > self.panel_rect.bottom - self.thumb_size[1]:
                break

    def _draw_hud(self, state):
        pygame.draw.rect(self.screen, (30, 30, 36), self.hud_rect)
        pygame.draw.rect(self.screen, (80, 80, 90), self.hud_rect, 1)
        legs = state['legs']
        head = state['head']
        tail = state['tail']
        head_pitch = head[2] - Kinematics.HEAD_PITCH_OFFSET
        lines = [
            f"action: {state['last_action']}  part: {state['last_part']}",
            f"legs: " + " ".join(f"{a:+6.1f}" for a in legs),
            f"head: yaw={head[0]:+6.1f} roll={head[1]:+6.1f} pitch={head_pitch:+6.1f}",
            f"tail: {tail[0]:+6.1f}   buffers L/H/T: "
            f"{state['legs_buffer']}/{state['head_buffer']}/{state['tail_buffer']}",
        ]
        x = self.hud_rect.left + 10
        y = self.hud_rect.top + 4
        for ln in lines:
            t = self.font.render(ln, True, (220, 220, 220))
            self.screen.blit(t, (x, y))
            y += 16
        # key help on the right
        help_txt = "1-4 view | S stick | H hud | T panel | +/- zoom | P pause | ESC quit"
        t = self.font.render(help_txt, True, (150, 150, 150))
        self.screen.blit(t, (self.hud_rect.right - t.get_width() - 10,
                             self.hud_rect.bottom - 18))


# ----------------------------------------------------------------------
# convenience launcher
# ----------------------------------------------------------------------
def run_emulator(pidog, **kwargs):
    """Create a :class:`Renderer` for ``pidog`` and run its main loop.

    This blocks until the window is closed.  Typically called from the main
    thread of your script after starting whatever actions you want on
    ``pidog`` (e.g. in a background thread).
    """
    r = Renderer(pidog, **kwargs)
    r.run()
    return r
