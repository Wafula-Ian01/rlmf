"""
Robot — the unified public API for the Robot Locomotion & Morphology Framework.

Usage:
    robot = Robot.load("spider.yaml")
    robot.walk()
    robot.turn_left()
    robot.climb()
"""
from __future__ import annotations
import math
import time
from typing import Dict, List, Optional, Tuple, Callable

from rlmf.mdl import MDLParser, RobotModel, Limb
from rlmf.gaits import GaitFrame, select_gait, GAIT_REGISTRY, Gait
from rlmf.balance import BalanceEngine, BalanceState
from rlmf.motors import MotorAbstractionLayer, SimulatedDriver


class RobotState:
    """Mutable runtime state of the robot."""
    def __init__(self):
        self.position: List[float] = [0.0, 0.0, 0.0]      # world x, y, z
        self.yaw: float = 0.0                               # heading in radians
        self.current_gait: Optional[str] = None
        self.is_walking: bool = False
        self.is_climbing: bool = False
        self.current_frame_index: int = 0
        self.gait_frames: List[GaitFrame] = []
        self.balance: Optional[BalanceState] = None
        self.log: List[str] = []

    def record(self, msg: str):
        self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(self.log) > 200:
            self.log = self.log[-200:]


class Robot:
    """
    Universal robot interface.
    Identical API regardless of morphology (3-leg, 4-leg, 6-leg, 8-leg…).
    """

    def __init__(self, model: RobotModel, motor_layer: Optional[MotorAbstractionLayer] = None):
        self._model = model
        self._motor = motor_layer or MotorAbstractionLayer()
        self._balance = BalanceEngine(com_height=model.body.height + 0.10)
        self._state = RobotState()
        self._frame_callbacks: List[Callable] = []

        # Precompute neutral foot positions (rest stance)
        self._neutral_positions: Dict[str, Tuple[float, float, float]] = {}
        for limb in self._model.limbs:
            limb._fk()
            self._neutral_positions[limb.name] = limb.foot_position

        # Register all joints with motor layer
        for key in self._model.joints:
            self._motor.register_joint(key)

        self._state.record(f"Robot '{model.name}' initialized ({model.topology.family}, "
                           f"{model.topology.limb_count} limbs)")

    # ─────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────

    @classmethod
    def load(cls, source) -> "Robot":
        """Load robot from YAML file path, YAML string, or dict."""
        model = MDLParser.load(source)
        return cls(model)

    # ─────────────────────────────────────────────
    # Public behaviour API
    # ─────────────────────────────────────────────

    def walk(self, steps: int = 60, direction: Optional[float] = None) -> "Robot":
        """Walk forward (or in a given direction in radians)."""
        dir_rad = direction if direction is not None else self._state.yaw
        gait_name = self._select_gait_name("walk")
        self._execute_gait(gait_name, dir_rad, steps)
        self._state.record(f"walk() → {gait_name} gait, {steps} frames")
        return self

    def turn_left(self, degrees: float = 45) -> "Robot":
        """Turn left by the specified angle."""
        self._state.yaw += math.radians(degrees)
        self._state.record(f"turn_left({degrees}°) → heading {math.degrees(self._state.yaw):.1f}°")
        return self

    def turn_right(self, degrees: float = 45) -> "Robot":
        """Turn right by the specified angle."""
        self._state.yaw -= math.radians(degrees)
        self._state.record(f"turn_right({degrees}°) → heading {math.degrees(self._state.yaw):.1f}°")
        return self

    def climb(self, steps: int = 60) -> "Robot":
        """Engage climbing mode (slow, maximally stable gait)."""
        self._state.is_climbing = True
        gait_name = self._select_gait_name("climb")
        self._execute_gait(gait_name, self._state.yaw, steps, step_height=0.07)
        self._state.is_climbing = False
        self._state.record(f"climb() → {gait_name} gait")
        return self

    def stop(self) -> "Robot":
        """Halt all motion, return to neutral stance."""
        self._state.is_walking = False
        self._state.gait_frames = []
        self._reset_to_neutral()
        self._state.record("stop() → neutral stance")
        return self

    def reach(self, limb_name: str, target: Tuple[float, float, float]) -> bool:
        """Move a specific limb's foot to a world-space target via IK."""
        limb = self._get_limb(limb_name)
        if limb is None:
            return False
        success = limb.solve_ik(target)
        angles = limb.get_angles()
        for jname, angle in angles.items():
            self._motor.move_joint(f"{limb_name}.{jname}", angle)
        self._state.record(f"reach({limb_name}, {target}) → {'ok' if success else 'out of range'}")
        return success

    def move_joint(self, joint_name: str, angle_deg: float) -> "Robot":
        """Directly drive a single joint by name."""
        self._motor.move_joint(joint_name, angle_deg)
        self._state.record(f"move_joint({joint_name}, {angle_deg}°)")
        return self

    def follow_path(self, waypoints: List[Tuple[float, float]]) -> "Robot":
        """Walk through a sequence of (x, y) waypoints."""
        for wx, wy in waypoints:
            dx = wx - self._state.position[0]
            dy = wy - self._state.position[1]
            target_yaw = math.atan2(dy, dx)
            delta_yaw = target_yaw - self._state.yaw
            if abs(math.degrees(delta_yaw)) > 5:
                if delta_yaw > 0:
                    self.turn_left(math.degrees(delta_yaw))
                else:
                    self.turn_right(-math.degrees(delta_yaw))
            dist = math.sqrt(dx ** 2 + dy ** 2)
            steps_needed = max(30, int(dist / 0.05 * 30))
            self.walk(steps=steps_needed, direction=target_yaw)
            self._state.position[0] = wx
            self._state.position[1] = wy
            self._state.record(f"follow_path → waypoint ({wx:.2f}, {wy:.2f})")
        return self

    # ─────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────

    @property
    def limbs(self):
        return self._model.limbs

    @property
    def joints(self):
        return self._model.joints

    @property
    def mass(self) -> float:
        return self._model.mass

    @property
    def topology(self):
        return self._model.topology

    @property
    def state(self) -> RobotState:
        return self._state

    def solve_ik(self) -> Dict[str, Tuple[float, float, float]]:
        return self._model.solve_ik()

    def solve_fk(self) -> Dict[str, Tuple[float, float, float]]:
        return self._model.solve_fk()

    def balance_state(self) -> BalanceState:
        foot_positions = {l.name: l.foot_position for l in self._model.limbs}
        stance = {l.name: True for l in self._model.limbs}
        return self._balance.analyze(
            tuple(self._state.position),
            foot_positions,
            stance,
            self.mass,
        )

    def get_gait_frames(self, gait_name: str, num_frames: int = 60) -> List[GaitFrame]:
        """Return precomputed gait frames without executing."""
        gait_cls = GAIT_REGISTRY.get(gait_name, list(GAIT_REGISTRY.values())[0])
        gait: Gait = gait_cls()
        return gait.generate_frames(
            [l.name for l in self._model.limbs],
            self._neutral_positions,
            self._state.yaw,
            num_frames,
        )

    def describe(self) -> str:
        t = self._model.topology
        lines = [
            f"Robot: {self._model.name}",
            f"  Family:    {t.family}",
            f"  Limbs:     {t.limb_count}",
            f"  Joints/limb: {t.joint_count_per_limb}",
            f"  Total DOF: {t.total_joints}",
            f"  Mass:      {self.mass} kg",
            f"  Segments:  {', '.join(self._model.body.segments)}",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _select_gait_name(self, behavior: str) -> str:
        gait = select_gait(self._model.topology.family, behavior)
        return gait.name

    def _execute_gait(self, gait_name: str, direction: float, num_frames: int,
                      step_height: float = 0.04):
        gait_cls = GAIT_REGISTRY.get(gait_name, list(GAIT_REGISTRY.values())[0])
        gait: Gait = gait_cls(step_height=step_height)
        frames = gait.generate_frames(
            [l.name for l in self._model.limbs],
            self._neutral_positions,
            direction,
            num_frames,
        )
        self._state.gait_frames = frames
        self._state.current_gait = gait_name
        self._state.is_walking = True

        for fi, frame in enumerate(frames):
            self._state.current_frame_index = fi
            # Apply IK to each limb
            for limb in self._model.limbs:
                target = frame.foot_targets[limb.name]
                limb.solve_ik(target)
                angles = limb.get_angles()
                for jname, angle in angles.items():
                    self._motor.move_joint(f"{limb.name}.{jname}", angle)

            # Balance analysis on every frame
            foot_pos = {l.name: l.foot_position for l in self._model.limbs}
            self._state.balance = self._balance.analyze(
                tuple(self._state.position),
                foot_pos,
                frame.stance_mask,
                self.mass,
            )

            # Advance body position
            step_dist = 0.05 / num_frames
            self._state.position[0] += step_dist * math.cos(direction)
            self._state.position[1] += step_dist * math.sin(direction)

        self._state.is_walking = False

    def _reset_to_neutral(self):
        for limb in self._model.limbs:
            for joint in limb.joints:
                joint.set_angle(0.0)
                self._motor.move_joint(f"{limb.name}.{joint.name}", 0.0)
            limb._fk()

    def _get_limb(self, name: str) -> Optional[Limb]:
        for limb in self._model.limbs:
            if limb.name == name or limb.name.startswith(name):
                return limb
        return None
