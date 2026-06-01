"""
Robot — unified public API for RLMF v0.3

What's new over v0.2
--------------------
  • Closed-loop execution: _execute_gait() reads IMU + contact each frame
    and applies BalanceController corrections before the IK solve.
  • Torque estimation: warns when a joint exceeds its rated torque.
  • Thermal monitoring: throttles gait speed before servo overtemperature.
  • Power monitoring: warns before PSU brownout.
  • PoseCorrector: detects accumulated drift and triggers re-neutral.
  • FeedbackReader is optional — if no sensors are provided, the robot
    falls back to open-loop (v0.2 behaviour) with a clear log warning.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

from rlmf.mdl import MDLParser, RobotModel, Limb, JointCalibration
from rlmf.gaits import GaitFrame, Gait, select_gait, GAIT_REGISTRY
from rlmf.balance import BalanceEngine, BalanceState
from rlmf.motors import MotorAbstractionLayer, SimulatedDriver
from rlmf.feedback import (
    FeedbackReader, FeedbackState,
    SimulatedIMU, SimulatedContactSensor,
)
from rlmf.controller import (
    BalanceController, CorrectionVector,
    TorqueEstimator, TorqueReport,
    ThermalMonitor, ThermalState,
    PowerMonitor, PowerState,
    PoseCorrector,
)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

class RobotState:
    def __init__(self):
        self.position:      List[float]            = [0.0, 0.0, 0.0]
        self.yaw:           float                  = 0.0
        self.current_gait:  Optional[str]          = None
        self.is_walking:    bool                   = False
        self.is_climbing:   bool                   = False
        self.frame_index:   int                    = 0
        self.gait_frames:   List[GaitFrame]        = []
        self.balance:       Optional[BalanceState] = None
        self.feedback:      Optional[FeedbackState]= None
        self.torque:        Optional[TorqueReport] = None
        self.thermal:       Optional[ThermalState] = None
        self.power:         Optional[PowerState]   = None
        self.log:           List[str]              = []

    def record(self, msg: str, level: str = "INFO") -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}][{level}] {msg}")
        if len(self.log) > 500:
            self.log = self.log[-500:]

    def warn(self, msg: str) -> None:
        self.record(msg, level="WARN")
        print(f"[RLMF WARN] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────────────────────────────────────

class Robot:
    """
    Universal legged robot interface — identical API for any morphology.

    Closed-loop usage (recommended for real hardware)::

        from rlmf import Robot
        from rlmf.feedback import FeedbackReader, MPU6050Sensor, FSRContactSensor

        imu     = MPU6050Sensor(address=0x68)
        contact = FSRContactSensor({"leg_0": 0, "leg_1": 1, ...})
        fb      = FeedbackReader(imu, contact)

        robot = Robot.load("hexapod.yaml", feedback=fb)
        robot.safe_startup()
        robot.walk()
        robot.safe_shutdown()

    Open-loop usage (simulation / no sensors)::

        robot = Robot.load("hexapod.yaml")
        robot.walk()
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self,
                 model:        RobotModel,
                 motor_layer:  Optional[MotorAbstractionLayer] = None,
                 feedback:     Optional[FeedbackReader]        = None,
                 frame_time_s: float = 0.025,
                 # Safety thresholds
                 rated_torque_Ncm:   float = 180.0,   # SG90 @ 6V
                 psu_current_limit_A: float = 6.0,
                 warn_temp_C:        float = 60.0,
                 max_temp_C:         float = 80.0):

        self._model      = model
        self._motor      = motor_layer or MotorAbstractionLayer(frame_time_s=frame_time_s)
        self._balance_eng= BalanceEngine(com_height=model.body.height + 0.10)
        self._state      = RobotState()
        self.frame_time  = frame_time_s

        # ── Safety subsystems ────────────────────────────────────────────────
        self._rated_torque   = rated_torque_Ncm
        self._torque_est     = TorqueEstimator(rated_torque_Ncm=rated_torque_Ncm)
        self._thermal        = ThermalMonitor(warn_temp_C=warn_temp_C,
                                              max_temp_C=max_temp_C)
        self._power_mon      = PowerMonitor(psu_current_limit_A=psu_current_limit_A)
        self._pose_corrector = PoseCorrector()

        # ── Feedback / closed-loop ───────────────────────────────────────────
        self._feedback    = feedback
        self._bal_ctrl    = BalanceController()
        self._closed_loop = feedback is not None

        if not self._closed_loop:
            self._state.warn(
                "No feedback sensors provided — running OPEN LOOP. "
                "Install IMU + contact sensors for closed-loop operation."
            )
            # Create simulated sensors so the API stays uniform
            limb_names = [l.name for l in model.limbs]
            self._feedback = FeedbackReader(
                imu     = SimulatedIMU(),
                contact = SimulatedContactSensor(limb_names),
            )

        # ── Neutral positions ────────────────────────────────────────────────
        self._neutral_positions: Dict[str, Tuple[float, float, float]] = {}
        for limb in self._model.limbs:
            self._neutral_positions[limb.name] = limb.neutral_foot_position()

        # ── Register joints ──────────────────────────────────────────────────
        for limb in self._model.limbs:
            for joint in limb.joints:
                key = f"{limb.name}.{joint.name}"
                self._motor.register_joint(key, calibration=joint.calibration)

        mode = "closed-loop" if self._closed_loop else "open-loop"
        self._state.record(
            f"Robot '{model.name}' ready — {model.topology.family}, "
            f"{model.topology.limb_count} limbs, {model.topology.total_joints} DOF, "
            f"{mode}"
        )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, source, **kwargs) -> "Robot":
        """Load from YAML file path, YAML string, Path, or dict."""
        model = MDLParser.load(source)
        return cls(model, **kwargs)

    # ── Hardware lifecycle ────────────────────────────────────────────────────

    def safe_startup(self, move_time_s: float = 2.0) -> "Robot":
        """
        Homing sequence + sensor calibration.

        Steps:
          1. Calibrate IMU (if real sensor attached).
          2. Interpolate all joints to neutral over move_time_s seconds.
          3. Run torque check at neutral stance.
        """
        self._state.record("safe_startup(): calibrating sensors")
        self._feedback.calibrate()

        self._state.record(f"safe_startup(): homing joints over {move_time_s:.1f}s")
        steps = max(10, int(move_time_s / self.frame_time))

        current: Dict[str, float] = {}
        target:  Dict[str, float] = {}
        for limb in self._model.limbs:
            for joint in limb.joints:
                key = f"{limb.name}.{joint.name}"
                current[key] = self._motor.read_joint(key)
                target[key]  = joint.calibration.to_servo_angle(0.0)

        for step in range(1, steps + 1):
            t = step / steps
            servo_angles: Dict[str, float] = {}
            for key in current:
                hw_id = self._motor._joint_map.get(key, key)
                servo_angles[hw_id] = current[key] + (target[key] - current[key]) * t
            self._motor._interp.move_to(servo_angles, frame_time_s=self.frame_time)

        for limb in self._model.limbs:
            for joint in limb.joints:
                joint.set_angle(0.0)
            limb._fk()

        # Torque check at neutral
        all_stance = {l.name: True for l in self._model.limbs}
        tr = self._torque_est.estimate(self._model.limbs, all_stance, self.mass)
        self._state.torque = tr
        if tr.overloaded_joints:
            self._state.warn(
                f"TORQUE OVERLOAD at neutral: {tr.overloaded_joints}. "
                f"Max: {tr.max_torque_Ncm:.0f} N·cm vs rated {self._rated_torque:.0f} N·cm. "
                f"Check servo selection for robot mass {self.mass} kg."
            )
        else:
            self._state.record(
                f"safe_startup(): torque OK — max {tr.max_torque_Ncm:.0f} N·cm "
                f"({tr.safety_margin*100:.0f}% margin)"
            )

        self._state.record("safe_startup(): complete")
        return self

    def safe_shutdown(self) -> "Robot":
        """Return to neutral and release servo torque."""
        self.stop()
        self._thermal.cool_down()
        self._state.record("safe_shutdown(): releasing servo torque")
        self._motor.detach_all()
        return self

    # ── Behaviour API ─────────────────────────────────────────────────────────

    def walk(self, steps: int = 60,
             direction: Optional[float] = None) -> "Robot":
        dir_rad   = direction if direction is not None else self._state.yaw
        gait_name = self._select_gait_name("walk")
        self._execute_gait(gait_name, dir_rad, steps)
        self._state.record(f"walk() — {gait_name}, {steps} frames")
        return self

    def turn_left(self, degrees: float = 45) -> "Robot":
        self._state.yaw += math.radians(degrees)
        self._state.record(
            f"turn_left({degrees}°) — heading {math.degrees(self._state.yaw):.1f}°")
        return self

    def turn_right(self, degrees: float = 45) -> "Robot":
        self._state.yaw -= math.radians(degrees)
        self._state.record(
            f"turn_right({degrees}°) — heading {math.degrees(self._state.yaw):.1f}°")
        return self

    def climb(self, steps: int = 60) -> "Robot":
        self._state.is_climbing = True
        gait_name = self._select_gait_name("climb")
        self._execute_gait(gait_name, self._state.yaw, steps, step_height=0.06)
        self._state.is_climbing = False
        self._state.record(f"climb() — {gait_name}")
        return self

    def stop(self) -> "Robot":
        self._state.is_walking = False
        self._state.gait_frames = []
        self._reset_to_neutral()
        self._pose_corrector.reset()
        self._bal_ctrl.reset_integrators()
        self._state.record("stop() — neutral stance")
        return self

    def reach(self, limb_name: str,
              target: Tuple[float, float, float]) -> bool:
        limb = self._get_limb(limb_name)
        if limb is None:
            return False
        ok = limb.solve_ik(target)
        for joint in limb.joints:
            key = f"{limb.name}.{joint.name}"
            self._motor.move_joint(key, joint.current_angle, joint.calibration)
        self._state.record(
            f"reach({limb_name}) — {'ok' if ok else 'clamped'}")
        return ok

    def move_joint(self, joint_name: str, angle_deg: float) -> "Robot":
        cal = self._get_joint_calibration(joint_name)
        self._motor.move_joint(joint_name, angle_deg, calibration=cal)
        self._state.record(f"move_joint({joint_name}, {angle_deg:.1f}°)")
        return self

    def follow_path(self, waypoints: List[Tuple[float, float]]) -> "Robot":
        for wx, wy in waypoints:
            dx = wx - self._state.position[0]
            dy = wy - self._state.position[1]
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                continue
            target_yaw = math.atan2(dy, dx)
            delta = target_yaw - self._state.yaw
            while delta >  math.pi: delta -= 2 * math.pi
            while delta < -math.pi: delta += 2 * math.pi
            if abs(math.degrees(delta)) > 5:
                if delta > 0:
                    self.turn_left(math.degrees(delta))
                else:
                    self.turn_right(-math.degrees(delta))
            dist = math.sqrt(dx**2 + dy**2)
            self.walk(steps=max(30, int(dist / 0.05 * 30)),
                      direction=target_yaw)
            self._state.position[0] = wx
            self._state.position[1] = wy
            self._state.record(f"follow_path → ({wx:.3f}, {wy:.3f})")
        return self

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def limbs(self):   return self._model.limbs
    @property
    def joints(self):  return self._model.joints
    @property
    def mass(self):    return self._model.mass
    @property
    def topology(self):return self._model.topology
    @property
    def state(self):   return self._state

    def solve_ik(self):  return self._model.solve_ik()
    def solve_fk(self):  return self._model.solve_fk()

    def balance_state(self) -> BalanceState:
        fp = {l.name: l.foot_position for l in self._model.limbs}
        st = {l.name: True for l in self._model.limbs}
        return self._balance_eng.analyze(tuple(self._state.position), fp, st, self.mass)

    def torque_report(self) -> TorqueReport:
        st = {l.name: True for l in self._model.limbs}
        return self._torque_est.estimate(self._model.limbs, st, self.mass)

    def get_gait_frames(self, gait_name: str,
                        num_frames: int = 60) -> List[GaitFrame]:
        gait_cls = GAIT_REGISTRY.get(gait_name, list(GAIT_REGISTRY.values())[0])
        return gait_cls().generate_frames(
            [l.name for l in self._model.limbs],
            self._neutral_positions,
            self._state.yaw,
            num_frames,
        )

    def describe(self) -> str:
        t = self._model.topology
        mode = "closed-loop" if self._closed_loop else "open-loop (no sensors)"
        return "\n".join([
            f"Robot     : {self._model.name}",
            f"Family    : {t.family}",
            f"Limbs     : {t.limb_count}",
            f"Joints/leg: {t.joint_count_per_limb}",
            f"Total DOF : {t.total_joints}",
            f"Mass      : {self.mass} kg",
            f"Segments  : {', '.join(self._model.body.segments)}",
            f"Mode      : {mode}",
        ])

    # ── Internal: gait execution with closed-loop correction ─────────────────

    def _execute_gait(self, gait_name: str, direction: float,
                      num_frames: int, step_height: float = 0.04) -> None:
        gait_cls = GAIT_REGISTRY.get(gait_name, list(GAIT_REGISTRY.values())[0])
        gait: Gait = gait_cls(step_height=step_height)
        frames = gait.generate_frames(
            [l.name for l in self._model.limbs],
            self._neutral_positions,
            direction,
            num_frames,
        )
        self._state.gait_frames  = frames
        self._state.current_gait = gait_name
        self._state.is_walking   = True

        calibrations = {
            f"{limb.name}.{joint.name}": joint.calibration
            for limb in self._model.limbs
            for joint in limb.joints
        }

        for fi, frame in enumerate(frames):
            self._state.frame_index = fi

            # ── 1. Read sensors ───────────────────────────────────────────
            fb = self._feedback.read()
            self._state.feedback = fb

            # Update simulated contact with gait's expected stance pattern
            if hasattr(self._feedback._contact, 'set_stance'):
                self._feedback._contact.set_stance(frame.stance_mask)

            # ── 2. Balance correction ─────────────────────────────────────
            correction = self._bal_ctrl.update(fb, frame.stance_mask)

            if correction.is_recovery:
                self._state.warn(
                    f"Frame {fi}: fall recovery triggered "
                    f"(tilt {fb.imu.tilt_magnitude:.1f}°)"
                )
                self._reset_to_neutral()
                self._bal_ctrl.reset_integrators()
                self._pose_corrector.reset()
                self._state.is_walking = False
                return

            # ── 3. Solve IK with correction applied ───────────────────────
            all_angles: Dict[str, float] = {}
            for limb in self._model.limbs:
                raw_target = frame.foot_targets[limb.name]

                # Apply balance correction: adjust foot Z targets to level body
                # Roll correction raises/lowers feet on left/right sides
                # Pitch correction raises/lowers feet on front/rear
                mx, my, _ = limb.mount_position
                body_len   = self._model.body.length / 2
                body_wid   = self._model.body.width  / 2

                roll_adj  = (my / max(body_wid,  1e-4)) * \
                            math.tan(math.radians(correction.roll_correction_deg))  * body_wid
                pitch_adj = (mx / max(body_len,  1e-4)) * \
                            math.tan(math.radians(correction.pitch_correction_deg)) * body_len

                corrected_target = (
                    raw_target[0],
                    raw_target[1],
                    raw_target[2] - correction.height_delta_m + roll_adj + pitch_adj,
                )

                limb.solve_ik(corrected_target)
                for joint in limb.joints:
                    key = f"{limb.name}.{joint.name}"
                    all_angles[key] = joint.current_angle

            # ── 4. Safety: torque check ───────────────────────────────────
            tr = self._torque_est.estimate(
                self._model.limbs, frame.stance_mask, self.mass)
            self._state.torque = tr

            if tr.overloaded_joints:
                self._state.warn(
                    f"Frame {fi}: torque overload on {tr.overloaded_joints} "
                    f"({tr.max_torque_Ncm:.0f} N·cm vs rated {self._rated_torque:.0f})"
                )

            # ── 5. Safety: thermal check ──────────────────────────────────
            thermal = self._thermal.update(tr.joint_torques_Ncm, self._rated_torque)
            self._state.thermal = thermal

            if thermal.throttle_factor < 0.5:
                self._state.warn(
                    f"Frame {fi}: thermal throttle {thermal.throttle_factor:.2f} — "
                    f"hot joints: {thermal.hot_joints}"
                )
                # Pause to cool down
                time.sleep(0.1 * (1.0 - thermal.throttle_factor))

            if thermal.throttle_factor == 0.0:
                self._state.warn("Thermal shutdown — stopping gait to protect servos")
                self._reset_to_neutral()
                self._state.is_walking = False
                return

            # ── 6. Safety: power check ────────────────────────────────────
            power = self._power_mon.estimate(
                tr, self._rated_torque, self._model.topology.total_joints)
            self._state.power = power

            if power.brownout_risk:
                self._state.warn(
                    f"Frame {fi}: brownout risk — "
                    f"estimated {power.estimated_current_A:.1f} A "
                    f"(PSU limit: {self._power_mon._psu_limit:.1f} A)"
                )

            # ── 7. Send to hardware (with speed-limited pacing) ───────────
            effective_frame_time = self.frame_time / max(0.1, thermal.throttle_factor)
            self._motor.frame_time = effective_frame_time
            self._motor.move_frame(all_angles, calibrations=calibrations)

            # ── 8. Balance analysis ───────────────────────────────────────
            foot_pos = {l.name: l.foot_position for l in self._model.limbs}
            bal = self._balance_eng.analyze(
                tuple(self._state.position), foot_pos, frame.stance_mask, self.mass)
            self._state.balance = bal

            # ── 9. Drift check ────────────────────────────────────────────
            if self._pose_corrector.check(bal.stability_margin):
                self._state.record(
                    f"Frame {fi}: pose drift detected — "
                    f"margin dropped to {bal.stability_margin*100:.1f} cm — re-neutralising"
                )
                self._reset_to_neutral()
                time.sleep(0.3)

            # ── 10. Advance body position estimate ────────────────────────
            step_dist = 0.05 / num_frames
            self._state.position[0] += step_dist * math.cos(direction)
            self._state.position[1] += step_dist * math.sin(direction)

        self._state.is_walking = False

    def _reset_to_neutral(self) -> None:
        for limb in self._model.limbs:
            for joint in limb.joints:
                joint.set_angle(0.0)
                key = f"{limb.name}.{joint.name}"
                self._motor.move_joint(key, 0.0, calibration=joint.calibration)
            limb._fk()

    def _select_gait_name(self, behavior: str) -> str:
        return select_gait(self._model.topology.family, behavior).name

    def _get_limb(self, name: str) -> Optional[Limb]:
        for limb in self._model.limbs:
            if limb.name == name or limb.name.startswith(name):
                return limb
        return None

    def _get_joint_calibration(self, joint_name: str) -> Optional[JointCalibration]:
        for limb in self._model.limbs:
            for joint in limb.joints:
                if f"{limb.name}.{joint.name}" == joint_name:
                    return joint.calibration
        return None
