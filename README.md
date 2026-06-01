# rlmf — Robot Locomotion & Morphology Framework

[![PyPI](https://img.shields.io/pypi/v/rlmf)](https://pypi.org/project/rlmf/)
[![Python](https://img.shields.io/pypi/pyversions/rlmf)](https://pypi.org/project/rlmf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()

Universal legged-robot locomotion API.  Define any robot in YAML, call `robot.walk()`.
Works identically for 4-leg, 6-leg, 8-leg, and 12-leg morphologies.

```python
from rlmf import Robot
from rlmf.feedback import FeedbackReader, MPU6050Sensor, FSRContactSensor

# Closed-loop (real hardware)
imu     = MPU6050Sensor(address=0x68)
contact = FSRContactSensor({"leg_0": 0, "leg_1": 1, "leg_2": 2,
                             "leg_3": 3, "leg_4": 4, "leg_5": 5})
fb      = FeedbackReader(imu, contact)

robot = Robot.load("hexapod.yaml", feedback=fb)
robot.safe_startup()
robot.walk()
robot.turn_left(45)
robot.climb()
robot.safe_shutdown()
```

---

## Installation

```bash
pip install rlmf                   # simulation / development
pip install "rlmf[pi]"            # + Raspberry Pi hardware drivers
```

---

## Architecture

```
User API            robot.walk() / robot.climb() / robot.follow_path()
      ↓
Gait Engine         TripodGait / WaveGait / RippleGait / TrotGait
      ↓
Kinematics          Full 3-D IK/FK — exact for all mount angles
      ↓
Balance Controller  PID (roll/pitch/yaw) — IMU + contact feedback
      ↓
Safety Layer        Torque · Thermal · Power · Pose drift detection
      ↓
Motor Layer         JointCalibration → TrajectoryInterpolator → driver
      ↓
Hardware            SimulatedDriver / PCA9685 / ChainedPCA9685 / custom
```

---

## What was fixed in v0.4

**IK sagittal plane error** — v0.3 assumed the knee+ankle plane was vertical
regardless of mount angle.  For legs at ±30°/±60° this caused errors up to
180 mm.  v0.4 projects the target into the leg's own sagittal frame, giving
IK→FK agreement < 0.001 mm on all mount angles.

**Balance controller oscillation** — v0.3 used PI only.  Without the
derivative term, the controller overshoots and rocks the body at 1–3 Hz.
v0.4 is a full PID controller using IMU gyro rate directly as the D signal.

**Thermal model ODE bug** — wrong denominator caused the model to saturate
at ~55 °C instead of the correct ~120 °C, so thermal throttling never fired.
Fixed and validated against SG90 datasheet (t_80 ≈ 90 s at stall).

**Contact threshold auto-calibration** — v0.3 required manual ADC threshold
tuning per foot.  v0.4 adds `ContactCalibrator` which measures rest and loaded
baselines and computes per-foot thresholds automatically.

---

## Morphology Description Language

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
      hip:   {min: -90, max:  90}
      knee:  {min: -90, max: 120}
      ankle: {min: -90, max:  90}
    segment_lengths: [0.07, 0.12, 0.10]
    calibration:
      hip:   {zero_offset: 0.0, direction: 1, speed_dps: 600}
      knee:  {zero_offset: 0.0, direction: 1, speed_dps: 600}
      ankle: {zero_offset: 0.0, direction: 1, speed_dps: 600}
physics:
  mass: 1.2kg
```

---

## Raspberry Pi Setup

### Hardware

| Component | Purpose |
|-----------|---------|
| 2× PCA9685 (0x40, 0x41) | 18-channel servo PWM |
| MPU-6050 (0x68) | IMU — roll/pitch/yaw |
| MCP3008 (SPI) | FSR ADC — foot contact |
| 18× SG90 or MG996R | Joints |
| 5 V ≥ 6 A PSU | Servo power (separate from Pi) |

### Wiring

```
Pi 3.3V  →  PCA9685 VCC (both boards), MPU-6050 VCC
Pi GND   →  common ground (all boards + PSU negative)
Pi GPIO2 →  SDA (PCA9685 A + B, MPU-6050 — parallel)
Pi GPIO3 →  SCL (PCA9685 A + B, MPU-6050 — parallel)
Pi SPI   →  MCP3008 (SCLK=GPIO11, MOSI=GPIO10, MISO=GPIO9, CE0=GPIO8)
5V PSU+  →  PCA9685 V+ (servo power rail)
Board B: solder A0 jumper → address becomes 0x41
```

### Sequence

```bash
bash pi_demo/setup.sh              # one-time system setup
python pi_demo/calibration.py      # measure per-joint servo offsets
python pi_demo/tripod_demo.py      # walk / turn / climb demo
```

---

## Contact calibration

```python
from rlmf import FSRContactSensor, ContactCalibrator

sensor = FSRContactSensor({"leg_0":0,"leg_1":1,...})
cal    = ContactCalibrator(sensor)

# 1. Lift robot (all feet free)
cal.measure_rest()

# 2. Lower robot onto all feet
cal.measure_loaded()

# 3. Apply thresholds
sensor.threshold = cal.thresholds   # per-foot dict
```

---

## CLI

```bash
rlmf robots                         # list bundled robots
rlmf describe hexapod               # topology
rlmf walk hexapod --gait wave       # simulate gait
rlmf balance quadruped              # balance state
```

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
