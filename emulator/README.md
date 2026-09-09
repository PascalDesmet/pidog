# PiDog Emulator

A software emulator for the SunFounder PiDog that lets you run the robot's
movement code **without any physical hardware** and watch the result in a
pygame window.  It reproduces the public API of `pidog.pidog.Pidog` (legs /
head / tail movement, preset actions, pose kinematics, wait helpers) and
drives an in-memory *virtual* robot whose servo state is visualised by a
renderer using your own photos of the dog.

## Quick start (from sirus laptop)

```bash
cd /home/pds/pidog
python3 -m emulator.demo
```

This runs a short sequence of preset actions (`stand`, `sit`, `lie`, `forward`,
`trot`, `wag_tail`, ...) and opens the viewer window. Close it with `ESC`.

## Using it with your own / existing scripts

The emulator is a drop-in replacement. In any existing pidog example, change:

```python
from pidog import Pidog
```
to:
```python
from emulator import Pidog, run_emulator
```

and add `run_emulator(dog)` (in the **main** thread) before `dog.close()`. The
renderer must run in the main thread because of pygame. If your script blocks
on the dog's actions, run those actions in a background thread, e.g.:

```python
import threading, time
from emulator import Pidog, run_emulator

dog = Pidog()
time.sleep(0.5)

def work():
    dog.do_action('stand', speed=60); dog.wait_all_done()
    dog.do_action('forward', step_count=8, speed=90); dog.wait_all_done()
    dog.do_action('wag_tail', step_count=20, speed=100); dog.wait_all_done()

threading.Thread(target=work, daemon=True).start()
run_emulator(dog)      # main thread: window loop
dog.close()
```

All movement methods match the real API: `legs_move`, `head_move`,
`head_move_raw`, `tail_move`, `do_action`, `legs_simple_move`, `set_pose`,
`set_rpy`, `set_legs`, `pose2legs_angle`, `wait_all_done`, `body_stop`,
`stop_and_lie`, calibration helpers, etc. Hardware-only peripherals (IMU,
ultrasonic, RGB strip, speaker) are stubbed so scripts don't crash:
`read_distance()` returns `-1`, `read_battery_voltage()` returns `8.0`,
`speak()` is a no-op, and `pitch`/`roll` stay `0` (no balance loop).

## How it works

* Three worker threads (legs / head / tail) consume angle frames from buffers,
  exactly like the real `Pidog`.
* `_ServoGroup.servo_move` interpolates from the current angles to the target
  using the real speed / max-dps timing and sleeps in 10 ms steps, so the
  emulator moves in **real time**, just like the physical robot.
* The kinematics (`coord2polar`, `legs_angle_calculation`, `pose2legs_angle`,
  `head_rpy_to_angle`, `Walk`, `Trot`) are re-implemented in pure Python/numpy
  and verified to round-trip with the real code's math
  (`python3 emulator/kinematics.py` prints `PASS`).
* The renderer polls the virtual servo state at ~60 fps and draws:
  * a **photographic background** cross-faded between the two nearest of the
    four body poses (lie / half_sit / sit / stand) based on the live leg
    angles, for the selected camera view;
  * a **stick-figure overlay** computed from the live 12 servo angles (precise
    sagittal forward kinematics) so you can see the exact motion the code
    produces — this is the part that "mimics the dog's movement from code";
  * a **side panel** with thumbnails of the body-pose photos for the current
    view (closest one highlighted);
  * a **HUD** with numeric servo angles, action buffers, current action/view.

## Controls

| Key | Action |
|-----|--------|
| `1` `2` `3` `4` | front / left / right / rear view |
| `←` `→` | cycle views |
| `S` | toggle stick overlay |
| `H` | toggle HUD |
| `T` | toggle thumbnail side panel |
| `+` / `-` | zoom the stick overlay |
| `P` | pause rendering (emulator keeps running) |
| `ESC` | quit |

Default view is **left side**.

## Image filenames

Put your photos in `/home/pds/pidog/emulator-images/`. The renderer auto-loads
them by filename. Only the **body** images are required; head/tail images are
optional enhancements (the stick overlay always shows head & tail precisely).

### Body poses (required) — `body_<pose>_<view>.png`

`pose` ∈ `lie`, `half_sit`, `sit`, `stand`
`view` ∈ `front`, `left`, `right`, `rear`

```
body_lie_front.png      body_half_sit_front.png   body_sit_front.png   body_stand_front.png
body_lie_left.png       body_half_sit_left.png    body_sit_left.png    body_stand_left.png
body_lie_right.png      body_half_sit_right.png   body_sit_right.png   body_stand_right.png
body_lie_rear.png       body_half_sit_rear.png    body_sit_rear.png    body_stand_rear.png
```

Shoot each pose with a **neutral head and tail**. `half_sit` is the
half-standing pose (between sit and stand).

### Head positions (optional) — `head_<pos>_<view>.png`

`pos` ∈ `center`, `left`, `right`, `up`, `down` (yaw left/right, pitch up/down)

```
head_center_<view>.png  head_left_<view>.png  head_right_<view>.png
head_up_<view>.png      head_down_<view>.png
```

Approximate target angles: center=(0,0,0), left=(yaw -45), right=(yaw +45),
up=(pitch -30), down=(pitch +20). These are full-dog photos; the renderer uses
them as reference (the stick overlay shows the live head motion precisely).

### Tail positions (optional) — `tail_<pos>_<view>.png`

`pos` ∈ `center`, `left`, `right`

```
tail_center_<view>.png  tail_left_<view>.png  tail_right_<view>.png
```

Approximate angles: center=0, left=-30, right=+30.

> Accepted extensions: `.png`, `.jpg`, `.jpeg` (case-insensitive). Missing
> images fall back to coloured placeholders, so the emulator works before any
> photos are supplied.

## Files

| File | Purpose |
|------|---------|
| `kinematics.py` | Pure-math kinematics + `Walk`/`Trot` gait generators + forward kinematics for the stick overlay (self-test included) |
| `actions.py` | Preset action dictionary (mirrors `pidog.actions_dictionary`) |
| `pidog_emu.py` | `Pidog` emulator class + virtual servo groups + threading + sensor stubs |
| `renderer.py` | Pygame renderer (image loading, cross-fade, stick overlay, side panel, HUD, keyboard) |
| `demo.py` | Demo action sequence + viewer |
| `__init__.py` | Public exports |

## Notes / limitations

* The emulator runs in **real time** (servo interpolation uses real sleeps),
  matching the physical robot's timing. There is no physics simulation —
  gravity/balance are not modelled (the IMU stub returns 0).
* The stick overlay is a **sagittal (side) projection** of the 2-DOF legs;
  it is most meaningful in the side views. Front/rear views still show the
  photographic background; the stick is drawn as a side projection for
  consistency.
* During walking/trot the legs move through configurations that aren't close
  to any of the four static body poses, so the photographic background will be
  a blend — the stick overlay carries the actual gait motion.
