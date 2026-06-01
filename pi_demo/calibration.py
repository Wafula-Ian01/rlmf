"""
RLMF Servo Calibration Tool
============================
Run this FIRST, before any gait demo.

For each joint:
  1. The servo moves to its current zero_offset position.
  2. Use +/- keys to find the true mechanical neutral (leg straight, pointing down).
  3. Press S to save that offset.
  4. Press N to advance to the next joint.

Results are saved to calibration_offsets.yaml. Copy those values into your
robot YAML under the calibration: section for each joint.

Controls:
  +        increase angle by 5°
  -        decrease angle by 5°
  .        increase angle by 1°
  ,        decrease angle by 1°
  0        return to 0°
  n / Tab  next joint
  p        previous joint
  s        save calibration.yaml
  q        quit
"""
from __future__ import annotations

import os
import sys
import time
import json
import yaml
import termios
import tty
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlmf import Robot
import rlmf.robots as robots_pkg

# ── Config ────────────────────────────────────────────────────────────────────
ROBOT_NAME  = "hexapod"
OUTPUT_FILE = Path(__file__).parent / "calibration_offsets.yaml"

# Channel map: joint → PCA9685 channel
# Edit this to match your physical wiring.
CHANNEL_MAP: Dict[str, int] = {
    "leg_0.hip": 0,  "leg_0.knee": 1,  "leg_0.ankle": 2,
    "leg_1.hip": 3,  "leg_1.knee": 4,  "leg_1.ankle": 5,
    "leg_2.hip": 6,  "leg_2.knee": 7,  "leg_2.ankle": 8,
    "leg_3.hip": 9,  "leg_3.knee":10,  "leg_3.ankle":11,
    "leg_4.hip":12,  "leg_4.knee":13,  "leg_4.ankle":14,
    "leg_5.hip":15,  "leg_5.knee":16,  "leg_5.ankle":17,
}

PULSE_MIN = 500
PULSE_MAX = 2400


def build_driver():
    try:
        from rlmf.motors import ChainedPCA9685Driver
        driver = ChainedPCA9685Driver(addr_a=0x40, addr_b=0x41)
        for jname, ch in CHANNEL_MAP.items():
            driver.assign_channel(jname, ch, PULSE_MIN, PULSE_MAX)
        print("[HW]  ChainedPCA9685Driver connected")
        return driver
    except Exception as e:
        print(f"[SIM] Hardware not found ({e}). Running in simulation mode.")
        from rlmf.motors import SimulatedDriver
        return SimulatedDriver()


def send_angle(driver, joint_id: str, framework_angle: float,
               zero_offset: float, direction: float) -> None:
    servo_angle = (framework_angle + zero_offset) * direction + 90.0
    servo_angle = max(0.0, min(180.0, servo_angle))
    driver.set_servo_angle(joint_id, servo_angle)


def getch() -> str:
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run():
    driver = build_driver()
    joints = list(CHANNEL_MAP.keys())

    # Load existing offsets if available
    offsets:    Dict[str, float] = {}
    directions: Dict[str, float] = {}

    if OUTPUT_FILE.exists():
        try:
            saved = yaml.safe_load(OUTPUT_FILE.read_text()) or {}
            for jname in joints:
                offsets[jname]    = saved.get(jname, {}).get("zero_offset", 0.0)
                directions[jname] = saved.get(jname, {}).get("direction", 1.0)
            print(f"[CAL] Loaded existing offsets from {OUTPUT_FILE}")
        except Exception:
            pass

    for jname in joints:
        if jname not in offsets:
            offsets[jname]    = 0.0
            directions[jname] = 1.0

    idx = 0
    print("\n" + "="*52)
    print("  RLMF Servo Calibration Tool")
    print("="*52)
    print("  +/-   coarse adjust (5°)   ./,  fine (1°)")
    print("  d     toggle direction      0   reset to 0°")
    print("  n/p   next/prev joint       s   save")
    print("  q     quit")
    print("="*52 + "\n")

    def show():
        jname = joints[idx]
        ch    = CHANNEL_MAP[jname]
        off   = offsets[jname]
        dirn  = directions[jname]
        servo = max(0, min(180, (0.0 + off) * dirn + 90.0))
        sys.stdout.write(
            f"\r  [{idx+1:02d}/{len(joints)}] {jname:<18} "
            f"ch={ch:02d}  offset={off:+.1f}°  dir={int(dirn):+d}  "
            f"servo≈{servo:.0f}°   "
        )
        sys.stdout.flush()
        send_angle(driver, jname, 0.0, off, dirn)

    show()

    while True:
        key = getch()
        jname = joints[idx]

        if key in ('+', '='):
            offsets[jname] += 5.0
        elif key == '-':
            offsets[jname] -= 5.0
        elif key == '.':
            offsets[jname] += 1.0
        elif key == ',':
            offsets[jname] -= 1.0
        elif key == '0':
            offsets[jname] = 0.0
        elif key == 'd':
            directions[jname] *= -1.0
        elif key in ('n', '\t', '\r'):
            idx = (idx + 1) % len(joints)
        elif key == 'p':
            idx = (idx - 1) % len(joints)
        elif key == 's':
            data = {
                jname: {
                    "zero_offset": offsets[jname],
                    "direction":   directions[jname],
                }
                for jname in joints
            }
            OUTPUT_FILE.write_text(yaml.dump(data, default_flow_style=False))
            print(f"\n[SAVED] {OUTPUT_FILE}")
            print("Copy these values into your robot YAML calibration: section.")
        elif key == 'q':
            print("\n[DONE] Exiting calibration.")
            break

        show()


if __name__ == "__main__":
    run()
