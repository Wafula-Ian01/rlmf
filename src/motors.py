"""
Motor Abstraction Layer — hardware-agnostic joint control.
Users call robot.move_joint("hip", 30); the driver handles the rest.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional
import time


class MotorDriver(ABC):
    """Abstract motor driver. Implement for Servo, Stepper, BLDC, Dynamixel, etc."""

    @abstractmethod
    def set_angle(self, joint_id: str, angle_deg: float, speed: float = 1.0):
        ...

    @abstractmethod
    def get_angle(self, joint_id: str) -> float:
        ...

    @abstractmethod
    def enable(self, joint_id: str):
        ...

    @abstractmethod
    def disable(self, joint_id: str):
        ...


class SimulatedDriver(MotorDriver):
    """
    In-memory simulated driver — used for development, testing, and this demo.
    Supports configurable servo speed and angle tracking.
    """
    def __init__(self, default_speed_dps: float = 300.0):
        self._angles: Dict[str, float] = {}
        self._targets: Dict[str, float] = {}
        self._enabled: Dict[str, bool] = {}
        self.speed_dps = default_speed_dps       # degrees per second

    def set_angle(self, joint_id: str, angle_deg: float, speed: float = 1.0):
        self._targets[joint_id] = angle_deg
        # In simulation, snap immediately (call tick() for animation)
        self._angles[joint_id] = angle_deg

    def get_angle(self, joint_id: str) -> float:
        return self._angles.get(joint_id, 0.0)

    def tick(self, dt: float):
        """Simulate servo movement toward targets at rated speed."""
        for jid, target in self._targets.items():
            current = self._angles.get(jid, 0.0)
            delta = target - current
            max_delta = self.speed_dps * dt
            if abs(delta) <= max_delta:
                self._angles[jid] = target
            else:
                self._angles[jid] = current + max_delta * (1 if delta > 0 else -1)

    def enable(self, joint_id: str):
        self._enabled[joint_id] = True

    def disable(self, joint_id: str):
        self._enabled[joint_id] = False

    def state_snapshot(self) -> Dict[str, float]:
        return dict(self._angles)


class PWMServoDriver(MotorDriver):
    """Stub for real PWM servo hardware (e.g. PCA9685)."""
    def __init__(self):
        self._angles: Dict[str, float] = {}

    def set_angle(self, joint_id: str, angle_deg: float, speed: float = 1.0):
        # map to PWM pulse here
        self._angles[joint_id] = angle_deg

    def get_angle(self, joint_id: str) -> float:
        return self._angles.get(joint_id, 0.0)

    def enable(self, joint_id: str): pass
    def disable(self, joint_id: str): pass


class MotorAbstractionLayer:
    """
    Unified joint control API.
    Swap out the driver without changing any robot code.
    """
    def __init__(self, driver: Optional[MotorDriver] = None):
        self.driver = driver or SimulatedDriver()
        self._joint_map: Dict[str, str] = {}       # logical name → hardware id

    def register_joint(self, logical_name: str, hardware_id: Optional[str] = None):
        self._joint_map[logical_name] = hardware_id or logical_name

    def move_joint(self, joint_name: str, angle_deg: float, speed: float = 1.0):
        hw_id = self._joint_map.get(joint_name, joint_name)
        self.driver.set_angle(hw_id, angle_deg, speed)

    def read_joint(self, joint_name: str) -> float:
        hw_id = self._joint_map.get(joint_name, joint_name)
        return self.driver.get_angle(hw_id)

    def move_all(self, angles: Dict[str, float], speed: float = 1.0):
        for name, angle in angles.items():
            self.move_joint(name, angle, speed)
