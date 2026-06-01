"""
RLMF v0.3 Hexapod Demo — Raspberry Pi 5
========================================
Full closed-loop demo: IMU balance correction, FSR contact detection,
torque monitoring, thermal throttling, and power brownout warnings.

Hardware required
-----------------
  2× PCA9685 PWM boards (0x40, 0x41) → 18 SG90/MG996R servos
  1× MPU-6050 IMU          (0x68 on same I2C bus)
  1× MCP3008 ADC via SPI   → 6 FSR sensors (one per foot)
  5V ≥6A PSU for servos (separate from Pi 5V rail)

Run in order
------------
  bash setup.sh              # one-time system setup
  python calibration.py      # measure per-joint zero offsets
  python tripod_demo.py      # full demo
"""
from __future__ import annotations

import sys, os, time, yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlmf import Robot, ChainedPCA9685Driver, SimulatedDriver
from rlmf.motors import MotorAbstractionLayer
from rlmf.feedback import FeedbackReader
import rlmf.robots as robots_pkg

# ── Channel map ───────────────────────────────────────────────────────────────
CHANNEL_MAP = {
    "leg_0.hip":  0,  "leg_0.knee":  1,  "leg_0.ankle":  2,
    "leg_1.hip":  3,  "leg_1.knee":  4,  "leg_1.ankle":  5,
    "leg_2.hip":  6,  "leg_2.knee":  7,  "leg_2.ankle":  8,
    "leg_3.hip":  9,  "leg_3.knee": 10,  "leg_3.ankle": 11,
    "leg_4.hip": 12,  "leg_4.knee": 13,  "leg_4.ankle": 14,
    "leg_5.hip": 15,  "leg_5.knee": 16,  "leg_5.ankle": 17,
}

# FSR channel map: each limb name → MCP3008 ADC channel
FSR_CHANNELS = {
    "leg_0": 0, "leg_1": 1, "leg_2": 2,
    "leg_3": 3, "leg_4": 4, "leg_5": 5,
}

PULSE_MIN    = 500
PULSE_MAX    = 2400
FRAME_TIME   = 0.025
CALIB_FILE   = Path(__file__).parent / "calibration_offsets.yaml"
PSU_LIMIT_A  = 6.0       # change to match your supply
RATED_TORQUE = 180.0     # SG90 @ 6V in N·cm; use 1100 for MG996R


def build_hardware():
    """Try to connect real hardware; fall back to simulation gracefully."""
    hw_driver = None
    imu       = None
    contact   = None

    # ── Servo driver ──────────────────────────────────────────────────────────
    try:
        from rlmf import ChainedPCA9685Driver
        hw_driver = ChainedPCA9685Driver(addr_a=0x40, addr_b=0x41)
        for jname, ch in CHANNEL_MAP.items():
            hw_driver.assign_channel(jname, ch, PULSE_MIN, PULSE_MAX)
        print("[HW]  Servo boards 0x40+0x41 connected")
    except Exception as e:
        print(f"[SIM] Servo hardware not found: {e}")

    # ── IMU ───────────────────────────────────────────────────────────────────
    try:
        from rlmf import MPU6050Sensor
        imu = MPU6050Sensor(address=0x68)
        print("[HW]  MPU-6050 IMU connected at 0x68")
    except Exception as e:
        print(f"[SIM] IMU not found: {e}")
        from rlmf import SimulatedIMU
        imu = SimulatedIMU()

    # ── Contact sensor ────────────────────────────────────────────────────────
    try:
        from rlmf import FSRContactSensor
        contact = FSRContactSensor(FSR_CHANNELS, threshold=200)
        print("[HW]  FSR contact sensors connected via MCP3008")
    except Exception as e:
        print(f"[SIM] Contact sensor not found: {e}")
        from rlmf import SimulatedContactSensor
        contact = SimulatedContactSensor(list(FSR_CHANNELS.keys()))

    driver = hw_driver or SimulatedDriver(default_speed_dps=600.0)
    mal    = MotorAbstractionLayer(driver=driver, frame_time_s=FRAME_TIME)
    fb     = FeedbackReader(imu, contact)

    return mal, fb


def load_calibration(robot: Robot) -> None:
    if CALIB_FILE.exists():
        try:
            offsets = yaml.safe_load(CALIB_FILE.read_text()) or {}
            for limb in robot.limbs:
                for joint in limb.joints:
                    key = f"{limb.name}.{joint.name}"
                    if key in offsets:
                        d = offsets[key]
                        joint.calibration.zero_offset_deg = float(d.get("zero_offset", 0.0))
                        joint.calibration.direction       = float(d.get("direction", 1.0))
                        robot._motor.register_joint(key, calibration=joint.calibration)
            print(f"[CAL] Loaded calibration offsets from {CALIB_FILE}")
        except Exception as e:
            print(f"[CAL] Could not load calibration: {e}")
    else:
        print(f"[CAL] No calibration file. Run calibration.py first.")


def main():
    print("=" * 56)
    print("  RLMF v0.3 Hexapod — Raspberry Pi 5 (closed-loop)")
    print("=" * 56)

    mal, fb = build_hardware()

    robot = Robot(
        Robot.load(robots_pkg.get("hexapod"))._model,
        motor_layer      = mal,
        feedback         = fb,
        frame_time_s     = FRAME_TIME,
        rated_torque_Ncm = RATED_TORQUE,
        psu_current_limit_A = PSU_LIMIT_A,
        warn_temp_C      = 60.0,
        max_temp_C       = 80.0,
    )

    load_calibration(robot)
    print()
    print(robot.describe())

    try:
        print("\n[INIT] Calibrating IMU + homing servos…")
        robot.safe_startup(move_time_s=2.0)
        input("\n  All legs straight? Press Enter to begin walking… ")

        print("\n[WALK] Tripod gait — 120 frames")
        robot.walk(steps=120)

        bal = robot.balance_state()
        print(f"\n[BAL]  stable={bal.is_stable}  "
              f"margin={bal.stability_margin*100:.1f}cm  "
              f"tip_risk={bal.tip_risk*100:.0f}%")

        if robot.state.torque:
            tr = robot.state.torque
            print(f"[TRQ]  max={tr.max_torque_Ncm:.0f}N·cm  "
                  f"margin={tr.safety_margin*100:.0f}%  "
                  f"overloaded={tr.overloaded_joints or 'none'}")

        if robot.state.thermal:
            th = robot.state.thermal
            print(f"[TEMP] throttle={th.throttle_factor:.2f}  "
                  f"hot={th.hot_joints or 'none'}")

        if robot.state.power:
            pw = robot.state.power
            print(f"[PWR]  {pw.estimated_current_A:.1f}A / "
                  f"{pw.estimated_power_W:.1f}W  "
                  f"brownout_risk={pw.brownout_risk}")

        print("\n[TURN] Left 45° then right 45°")
        robot.turn_left(45).walk(steps=30).turn_right(45).walk(steps=30)

        print("\n[CLIMB] Wave gait")
        robot.climb(steps=60)

        print("\n[DONE] Returning to neutral…")
        robot.stop()

        print("\n[LOG]  Last events:")
        for line in robot.state.log[-12:]:
            print(" ", line)

    except KeyboardInterrupt:
        print("\n\n[STOP] Ctrl+C — safe shutdown…")

    finally:
        robot.safe_shutdown()
        print("[DONE] Servo torque released.")


if __name__ == "__main__":
    main()
