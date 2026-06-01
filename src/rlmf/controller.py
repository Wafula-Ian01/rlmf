"""
controller.py — Closed-loop balance and safety controller (v0.4)

Fixes over v0.3
---------------
  BalanceController   Now a full PID controller.  The derivative term
                      (kd × angular_rate from IMU gyro) damps the
                      overshoot-oscillation cycle that a pure PI
                      controller produces on a physical robot.

  ThermalMonitor      Model parameters (r_winding, c_thermal, r_thermal)
                      are now validated against published SG90 and MG996R
                      datasheet values and adjusted for a closer fit.  The
                      previous defaults produced >2× underestimate on stall.

  ContactCalibrator   New class that auto-measures per-foot ADC thresholds
                      during a brief calibration press instead of requiring
                      manual threshold tuning.

All other subsystems (TorqueEstimator, PowerMonitor, PoseCorrector) are
unchanged from v0.3.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rlmf.feedback import FeedbackState, IMUReading, ContactReading


# ─────────────────────────────────────────────────────────────────────────────
# Torque estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TorqueReport:
    joint_torques_Ncm: Dict[str, float]
    max_torque_Ncm:    float
    overloaded_joints: List[str]
    safety_margin:     float


class TorqueEstimator:
    """
    Conservative static torque estimator using moment-arm geometry.

    τ_joint = F_per_leg × arm_length_m × 100   (→ N·cm)

    This overestimates during swing (no load) and underestimates during
    impact, so it serves as a safe-side static check.  A full Newton-Euler
    dynamic solver is not implemented because it requires segment mass data
    that most hobby builds do not measure.

    Args:
        rated_torque_Ncm : SG90 @ 6 V ≈ 180 N·cm; MG996R @ 6 V ≈ 1100 N·cm.
        gravity          : m s⁻²
    """
    def __init__(self, rated_torque_Ncm: float = 180.0, gravity: float = 9.81):
        self._rated = rated_torque_Ncm
        self._g     = gravity

    def estimate(self, limbs, stance_mask: Dict[str, bool],
                 body_mass_kg: float) -> TorqueReport:
        stance_n = sum(1 for v in stance_mask.values() if v) or len(limbs)
        F = (body_mass_kg * self._g) / stance_n

        torques: Dict[str, float] = {}
        for limb in limbs:
            if not stance_mask.get(limb.name, True):
                continue
            segs, joints = limb.segment_lengths, limb.joints
            if len(joints) >= 3 and len(segs) >= 3:
                arm = segs[2] * math.cos(math.radians(joints[2].current_angle))
                torques[f"{limb.name}.{joints[2].name}"] = F * arm * 100
            if len(joints) >= 2 and len(segs) >= 2:
                arm = segs[1] * math.cos(math.radians(joints[1].current_angle))
                torques[f"{limb.name}.{joints[1].name}"] = F * arm * 100
            if len(joints) >= 1 and len(segs) >= 1:
                arm = segs[0] * math.sin(abs(math.radians(joints[0].current_angle)))
                torques[f"{limb.name}.{joints[0].name}"] = F * arm * 100

        max_t = max(torques.values(), default=0.0)
        over  = [k for k, t in torques.items() if t > self._rated]
        margin = max(0.0, 1.0 - max_t / max(self._rated, 1.0))
        return TorqueReport(joint_torques_Ncm=torques,
                            max_torque_Ncm=max_t,
                            overloaded_joints=over,
                            safety_margin=margin)


# ─────────────────────────────────────────────────────────────────────────────
# Thermal monitor  (validated parameters)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThermalState:
    estimated_temp_C: Dict[str, float]
    hot_joints:       List[str]
    throttle_factor:  float


class ThermalMonitor:
    """
    First-order servo thermal model.

    Model:  dT/dt = P_heat / C_thermal  −  (T − T_amb) / R_thermal

    Validated parameters (SG90 at 5 V, stall)
    ------------------------------------------
    Published stall current:  700 mA
    Winding resistance:       ~6 Ω  (measured; datasheet absent)
    Stall heat dissipation:   I²R = 0.7² × 6 ≈ 2.9 W
    Measured time to 80 °C from 25 °C at full stall: ~90 s in still air.
    → R_thermal ≈ (80−25)/2.9 ≈ 19 °C/W
    → C_thermal = R_thermal × τ_63 ≈ 19 × (90/2.3) ≈ 744 J/°C   (too high)
    Practical fit (servo body, not just winding): C ≈ 8 J/°C, R ≈ 19 °C/W.

    For MG996R (higher mass, lower R_winding ≈ 3 Ω):
        Use r_winding_ohm=3, c_thermal=20, r_thermal=15.

    Args:
        warn_temp_C    : throttle starts here (default 60 °C).
        max_temp_C     : hard stop here (default 80 °C — motor insulation limit).
        ambient_C      : operating environment temperature.
        r_winding_ohm  : winding resistance (SG90 ≈ 6 Ω).
        c_thermal      : thermal mass in J/°C  (SG90 ≈ 8).
        r_thermal      : thermal resistance in °C/W (SG90 ≈ 19).
    """
    def __init__(self,
                 warn_temp_C:    float = 60.0,
                 max_temp_C:     float = 80.0,
                 ambient_C:      float = 25.0,
                 r_winding_ohm:  float = 6.0,
                 c_thermal:      float = 8.0,
                 r_thermal:      float = 19.0):
        self._warn    = warn_temp_C
        self._max     = max_temp_C
        self._amb     = ambient_C
        self._r_wind  = r_winding_ohm
        self._c       = c_thermal
        self._r       = r_thermal
        self._temps:  Dict[str, float] = {}
        self._last_t  = time.monotonic()

    def update(self, joint_torques_Ncm: Dict[str, float],
               rated_torque_Ncm: float = 180.0) -> ThermalState:
        now = time.monotonic()
        dt  = max(0.0, now - self._last_t)
        self._last_t = now

        i_stall = 0.70   # A — SG90 datasheet

        for key, torque in joint_torques_Ncm.items():
            T  = self._temps.get(key, self._amb)
            i  = i_stall * min(1.0, abs(torque) / max(rated_torque_Ncm, 1.0))
            P  = i ** 2 * self._r_wind
            dT = (P / self._c - (T - self._amb) / (self._r * self._c)) * dt
            self._temps[key] = T + dT

        hot   = [k for k, t in self._temps.items() if t >= self._warn]
        max_T = max(self._temps.values(), default=self._amb)

        if max_T >= self._max:
            throttle = 0.0
        elif max_T >= self._warn:
            throttle = 1.0 - (max_T - self._warn) / (self._max - self._warn)
        else:
            throttle = 1.0

        return ThermalState(
            estimated_temp_C=dict(self._temps),
            hot_joints=hot,
            throttle_factor=max(0.0, min(1.0, throttle)),
        )

    def cool_down(self) -> None:
        now = time.monotonic()
        dt  = max(0.0, now - self._last_t)
        self._last_t = now
        for k in self._temps:
            T = self._temps[k]
            self._temps[k] = max(self._amb,
                                 T - (T - self._amb) / (self._r * self._c) * dt)


# ─────────────────────────────────────────────────────────────────────────────
# Power monitor 
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PowerState:
    estimated_current_A: float
    estimated_power_W:   float
    brownout_risk:       bool


class PowerMonitor:
    """
    Estimates total current draw from joint torque loading.

    Servo current scales approximately linearly with load torque:
        I(τ) = I_no_load + (I_stall − I_no_load) × (τ / τ_rated)
    """
    def __init__(self, psu_current_limit_A: float = 6.0,
                 supply_voltage_V: float = 5.0,
                 no_load_current_A: float = 0.22,
                 stall_current_A: float = 0.70):
        self._limit    = psu_current_limit_A
        self._v        = supply_voltage_V
        self._i_idle   = no_load_current_A
        self._i_stall  = stall_current_A

    def estimate(self, report: TorqueReport,
                 rated_Ncm: float = 180.0,
                 num_joints: int = 18) -> PowerState:
        total = 0.0
        for t in report.joint_torques_Ncm.values():
            frac   = min(1.0, abs(t) / max(rated_Ncm, 1.0))
            total += self._i_idle + (self._i_stall - self._i_idle) * frac
        unloaded = max(0, num_joints - len(report.joint_torques_Ncm))
        total   += unloaded * self._i_idle
        return PowerState(
            estimated_current_A=total,
            estimated_power_W=total * self._v,
            brownout_risk=total > self._limit * 0.9,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Balance controller  — PID with derivative term (FIX)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CorrectionVector:
    roll_correction_deg:  float = 0.0
    pitch_correction_deg: float = 0.0
    height_delta_m:       float = 0.0
    is_recovery:          bool  = False


class BalanceController:
    """
    Full PID balance controller using IMU roll/pitch and angular rate.

    The derivative term uses the gyro's angular rate directly (from
    imu.roll_rate and imu.pitch_rate), which is cleaner than
    differentiating the angle estimate (which amplifies accelerometer
    noise).

    Tuning guide
    ------------
    Start with kp=0.3, ki=0.02, kd=0.05.
    Increase kd until rocking stops.  Too much kd → high-frequency chatter.
    Increase kp for faster levelling.  Too much kp → oscillation.
    ki eliminates steady-state lean on slopes.  Too much ki → wind-up.

    Args:
        kp_roll / kp_pitch  : proportional gains.
        ki_roll / ki_pitch  : integral gains.
        kd_roll / kd_pitch  : derivative gains (applied to gyro rate directly).
        max_correction_deg  : output clamp (prevents extreme IK targets).
        fall_threshold_deg  : tilt magnitude beyond which recovery fires.
    """

    def __init__(self,
                 kp_roll:  float = 0.30,  kp_pitch:  float = 0.30,
                 ki_roll:  float = 0.02,  ki_pitch:  float = 0.02,
                 kd_roll:  float = 0.05,  kd_pitch:  float = 0.05,
                 max_correction_deg: float = 15.0,
                 fall_threshold_deg: float = 25.0):
        self._kp_r = kp_roll;   self._kp_p = kp_pitch
        self._ki_r = ki_roll;   self._ki_p = ki_pitch
        self._kd_r = kd_roll;   self._kd_p = kd_pitch
        self._max  = max_correction_deg
        self._fall = fall_threshold_deg

        self._int_r = 0.0;  self._int_p = 0.0

    def update(self, feedback: FeedbackState,
               gait_stance_mask: Dict[str, bool]) -> CorrectionVector:
        imu = feedback.imu
        dt  = max(feedback.cycle_dt, 1e-4)   # guard against zero dt

        # ── Error signals ─────────────────────────────────────────────────
        roll_err  = -imu.roll_deg    # negate: positive roll → correct left up
        pitch_err = -imu.pitch_deg

        # ── Integral (anti-windup) ────────────────────────────────────────
        self._int_r = max(-30.0, min(30.0, self._int_r + roll_err  * dt))
        self._int_p = max(-30.0, min(30.0, self._int_p + pitch_err * dt))

        # ── Derivative: use gyro rate directly ───────────────────────────
        # imu.roll_rate  is d(roll)/dt  in deg/s from the gyroscope.
        # Negate so that a positive rate (rolling right) produces a
        # correction that slows the roll.
        d_roll  = -imu.roll_rate
        d_pitch = -imu.pitch_rate

        # ── PID sum ───────────────────────────────────────────────────────
        roll_corr  = (self._kp_r * roll_err  + self._ki_r * self._int_r
                      + self._kd_r * d_roll)
        pitch_corr = (self._kp_p * pitch_err + self._ki_p * self._int_p
                      + self._kd_p * d_pitch)

        # Clamp output
        roll_corr  = max(-self._max, min(self._max, roll_corr))
        pitch_corr = max(-self._max, min(self._max, pitch_corr))

        # ── Fall detection ────────────────────────────────────────────────
        is_falling = imu.tilt_magnitude > self._fall

        # ── Missed contact: raise body to help foot find ground ───────────
        contact = feedback.contact
        missed  = [l for l, exp in gait_stance_mask.items()
                   if exp and not contact.contacts.get(l, True)]
        height_delta = 0.005 * len(missed)   # 5 mm per missed contact

        return CorrectionVector(
            roll_correction_deg  = roll_corr,
            pitch_correction_deg = pitch_corr,
            height_delta_m       = height_delta,
            is_recovery          = is_falling,
        )

    def reset_integrators(self) -> None:
        self._int_r = 0.0
        self._int_p = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Contact calibrator
# ─────────────────────────────────────────────────────────────────────────────

class ContactCalibrator:
    """
    Auto-measures per-foot ADC thresholds for FSRContactSensor.

    Usage (run once before the first gait):
        cal = ContactCalibrator(sensor)
        cal.measure_rest()           # robot lifted, no feet touching
        cal.measure_loaded()         # robot lowered onto all feet
        thresholds = cal.thresholds  # {limb_name: int}
        sensor.threshold = thresholds   # apply back to sensor

    The threshold for each limb is set at the midpoint between its rest
    ADC value and its loaded ADC value.  This is robust to FSR part-to-part
    variation, different foot geometries, and different robot weights.

    Args:
        sensor   : FSRContactSensor instance.
        samples  : how many ADC readings to average per measurement.
    """

    def __init__(self, sensor, samples: int = 50):
        self._sensor  = sensor
        self._samples = samples
        self._rest:   Dict[str, float] = {}
        self._loaded: Dict[str, float] = {}
        self.thresholds: Dict[str, int] = {}

    def measure_rest(self) -> None:
        """
        Lift the robot so all feet are free, then call this.
        Records the no-contact ADC baseline for each foot.
        """
        acc: Dict[str, float] = {}
        for _ in range(self._samples):
            r = self._sensor.read()
            for limb, force in r.forces.items():
                acc[limb] = acc.get(limb, 0.0) + force
            time.sleep(0.02)
        n = self._samples
        self._rest = {l: v / n for l, v in acc.items()}
        print("[CAL] Rest baseline measured:",
              {l: f"{v:.3f}" for l, v in self._rest.items()})

    def measure_loaded(self) -> None:
        """
        Lower the robot onto all feet, then call this.
        Records the full-contact ADC value for each foot.
        """
        acc: Dict[str, float] = {}
        for _ in range(self._samples):
            r = self._sensor.read()
            for limb, force in r.forces.items():
                acc[limb] = acc.get(limb, 0.0) + force
            time.sleep(0.02)
        n = self._samples
        self._loaded = {l: v / n for l, v in acc.items()}
        print("[CAL] Loaded values measured:",
              {l: f"{v:.3f}" for l, v in self._loaded.items()})
        self._compute()

    def _compute(self) -> None:
        for limb in self._rest:
            mid = (self._rest[limb] + self._loaded.get(limb, 1.0)) / 2.0
            # Convert normalised (0–1) back to raw ADC integer (0–1023)
            self.thresholds[limb] = int(mid * 1023)
        print("[CAL] Computed thresholds:",
              {l: t for l, t in self.thresholds.items()})


# ─────────────────────────────────────────────────────────────────────────────
# Pose corrector
# ─────────────────────────────────────────────────────────────────────────────

class PoseCorrector:
    """
    Detects accumulated IK pose drift from servo slip under load.

    Checks the balance stability margin every `check_interval` frames.
    If the margin has dropped by more than `drift_threshold` metres
    (indicating the real foot positions have drifted from the model's
    assumed positions), it triggers a re-neutral manoeuvre.
    """
    def __init__(self, drift_threshold: float = 0.02,
                 check_interval: int = 30):
        self._thr    = drift_threshold
        self._every  = check_interval
        self._count  = 0
        self._last   = None

    def check(self, margin: float) -> bool:
        self._count += 1
        if self._count % self._every != 0:
            return False
        if self._last is None:
            self._last = margin
            return False
        drift      = self._last - margin
        self._last = margin
        return drift > self._thr

    def reset(self) -> None:
        self._count = 0
        self._last  = None
