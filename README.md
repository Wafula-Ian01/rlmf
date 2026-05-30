# rlmf — Robot Locomotion & Morphology Framework

[![PyPI](https://img.shields.io/pypi/v/rlmf)](https://pypi.org/project/rlmf/)
[![Python](https://img.shields.io/pypi/pyversions/rlmf)](https://pypi.org/project/rlmf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

A universal robot body model and locomotion API.  
Define any legged robot in YAML, then call `robot.walk()`.

```python
from rlmf import Robot

robot = Robot.load("hexapod.yaml")
robot.walk()
robot.turn_left(45)
robot.climb()
robot.follow_path([(0.5, 0), (0.5, 0.5), (0, 0.5)])
```

The API is **identical** regardless of morphology — 4-leg, 6-leg, 8-leg, 12-leg.  
No gait code. No IK math. No synchronization logic. No motor coordination code.

---

## Installation

```bash
pip install rlmf
```

With Raspberry Pi hardware support (PCA9685 + GPIO servos):

```bash
pip install "rlmf[pi]"
```

---

## Quickstart

```python
from rlmf import Robot
import rlmf.robots as robots

# Use a bundled robot definition
robot = Robot.load(robots.get("hexapod"))

# Or load your own YAML
robot = Robot.load("my_robot.yaml")

# Behaviours — same API for any morphology
robot.walk()
robot.turn_left(45)
robot.turn_right(30)
robot.climb()
robot.stop()
robot.follow_path([(0.5, 0.0), (0.5, 0.5)])

# Low-level access
robot.reach("leg_0", (0.20, 0.05, -0.10))   # IK to world target
robot.move_joint("leg_0.hip", 30.0)          # direct joint control
```

---

## Morphology Description Language

Define any robot in YAML — the framework handles the rest:

```yaml
name: HexaBot
body:
  segments: [thorax]
  length: 0.30
  width: 0.18
  height: 0.09
limbs:
  - type: leg
    count: 6
    joints:
      hip:   {min: -90, max: 90}
      knee:  {min: 0,   max: 120}
      ankle: {min: -45, max: 45}
    segment_lengths: [0.07, 0.12, 0.10]
physics:
  mass: 1.2kg
```

After loading, the full model is immediately introspectable:

```python
robot.topology.family        # "hexapod"
robot.topology.limb_count    # 6
robot.topology.total_joints  # 18
robot.mass                   # 1.2
robot.limbs                  # List[Limb]
robot.joints                 # Dict[str, JointDef]
```

---

## Bundled robots

```python
import rlmf.robots as robots

robots.list_robots()          # ['centipede', 'hexapod', 'quadruped', 'spider']
path = robots.get("hexapod")  # Path to bundled hexapod.yaml
```

---

## Architecture

```
User API          robot.walk() / robot.turn_left() / robot.climb()
     ↓
Gait Engine       TripodGait / WaveGait / RippleGait / TrotGait
     ↓
Kinematics        solve_ik() / solve_fk() per limb
     ↓
Balance Engine    support polygon · stability margin · tip risk
     ↓
Motor Layer       SimulatedDriver / PCA9685 / your hardware
```

### Gait engine

| Gait | Class | Best for | Duty factor |
|------|-------|----------|-------------|
| tripod | `TripodGait` | hexapods, fast walking | 0.50 |
| wave | `WaveGait` | any, climbing | 0.83 |
| ripple | `RippleGait` | octopods, medium speed | 0.67 |
| trot | `TrotGait` | quadrupeds | 0.50 |

Gait selection is automatic — `robot.walk()` picks the right pattern for your morphology.  
Override it when needed:

```python
from rlmf import select_gait, TrotGait

gait = select_gait("hexapod", "climb")          # → WaveGait
frames = robot.get_gait_frames("ripple", num_frames=60)
```

### Balance engine

```python
state = robot.balance_state()

state.center_of_mass        # (x, y, z) world space
state.support_polygon       # convex hull of stance feet
state.stability_margin      # metres to nearest polygon edge (>0 = stable)
state.is_stable             # bool
state.tip_risk              # 0.0 (safe) … 1.0 (falling)
```

### Motor abstraction

Swap the hardware driver without touching any robot code:

```python
from rlmf import Robot, MotorAbstractionLayer
from my_hardware import MyServoDriver       # your implementation

robot = Robot(
    Robot.load("hexapod.yaml")._model,
    motor_layer=MotorAbstractionLayer(driver=MyServoDriver()),
)
robot.walk()   # drives your hardware
```

Implement `MotorDriver` for any servo bus:

```python
from rlmf import MotorDriver

class MyDriver(MotorDriver):
    def set_angle(self, joint_id, angle_deg, speed=1.0): ...
    def get_angle(self, joint_id) -> float: ...
    def enable(self, joint_id): ...
    def disable(self, joint_id): ...
```

---

## CLI

```bash
rlmf robots                        # list bundled robots
rlmf describe hexapod              # print topology
rlmf describe path/to/mybot.yaml   # your own robot
rlmf walk hexapod --gait wave      # simulate gait, print stats
rlmf balance quadruped             # print balance state
```

---

## Raspberry Pi deployment

Install with hardware extras:

```bash
pip install "rlmf[pi]"
```

Wire a PCA9685 to the Pi over I2C, then:

```python
from adafruit_servokit import ServoKit
from rlmf import Robot, MotorAbstractionLayer
from rlmf.motors import MotorDriver

class PCA9685Driver(MotorDriver):
    def __init__(self):
        self._kit = ServoKit(channels=16)
        self._map = {}

    def assign(self, joint_id, channel):
        self._map[joint_id] = channel

    def set_angle(self, joint_id, angle_deg, speed=1.0):
        ch = self._map.get(joint_id)
        if ch is not None:
            self._kit.servo[ch].angle = max(0, min(180, angle_deg + 90))

    def get_angle(self, joint_id): return 0.0
    def enable(self, joint_id): pass
    def disable(self, joint_id): pass

driver = PCA9685Driver()
driver.assign("leg_0.hip", 0)
# … assign all 18 joints …

import rlmf.robots as robots
robot = Robot(
    Robot.load(robots.get("hexapod"))._model,
    motor_layer=MotorAbstractionLayer(driver=driver),
)
robot.walk()   # moves physical servos
```

---

## Roadmap

- **v0.1** — Phase 1: 3–12 leg robots, four gait patterns, analytical IK, balance engine, motor abstraction
- **v0.2** — Phase 2: biped support, dynamic balance, weight shifting, fall recovery
- **v0.3** — Phase 3: arbitrary morphologies, terrain adaptation, path planning

---

## Development

```bash
git clone https://github.com/Wafula_Ian01/rlmf
cd rlmf
pip install -e ".[dev]"
pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
