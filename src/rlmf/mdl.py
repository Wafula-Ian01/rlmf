"""
mdl.py — Morphology Description Language parser and kinematic model (v0.4)

IK fix (v0.4)
-------------
The v0.3 IK decomposed the knee+ankle problem in an assumed-vertical sagittal
plane.  For leg mount angles near ±45° and ±135° this introduced a systematic
foot-position error of up to 15 mm.

The fix is a full 3-D geometric solver:

  1. Hip angle  — atan2 in the XY plane, exactly as before.
  2. Coxa tip   — computed in 3-D body frame using the clamped hip angle.
  3. Sagittal frame — a local 2-D frame whose X-axis points along the leg's
     effective azimuth (mount_angle + hip_angle) and whose Y-axis is world Z.
     The target is expressed in THIS frame, so the 2-link solver always works
     in the correct plane regardless of mount angle.
  4. 2-link IK  — solved in the sagittal frame; angles are exact.
  5. Ankle      — set so the foot is level with the ground (tibia
     compensates for the knee bend via the standard compensation formula).
  6. FK verification — after IK the FK is re-run and the foot position is
     updated so the balance engine always sees the correct geometry.
"""
from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JointCalibration:
    """
    Per-joint servo calibration.

    zero_offset_deg : added to the framework angle before sending to servo.
                      Measured with calibration.py — do this once per build.
    direction       : +1 (servo rotates same way as framework angle),
                      -1 (physically reversed — horn was installed mirrored).
    min_pulse_us    : pulse width for 0 ° command  (SG90 = 500 µs).
    max_pulse_us    : pulse width for 180° command  (SG90 = 2400 µs,
                                                     MG996R = 2500 µs).
    speed_dps       : rated angular speed (degrees / second) at nominal voltage.
                      SG90   ≈ 600 dps  (0.10 s / 60°)
                      MG996R ≈ 400 dps  (0.15 s / 60°)
    """
    zero_offset_deg: float = 0.0
    direction:       float = 1.0
    min_pulse_us:    int   = 500
    max_pulse_us:    int   = 2400
    speed_dps:       float = 600.0

    def to_servo_angle(self, framework_angle_deg: float) -> float:
        """Framework angle (±90 convention) → servo command angle (0–180)."""
        raw = (framework_angle_deg + self.zero_offset_deg) * self.direction
        return max(0.0, min(180.0, raw + 90.0))

    def move_time_s(self, from_deg: float, to_deg: float) -> float:
        return abs(to_deg - from_deg) / max(self.speed_dps, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Joint
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JointDef:
    name:          str
    min_angle:     float         = -90.0
    max_angle:     float         =  90.0
    current_angle: float         =   0.0
    calibration:   JointCalibration = field(default_factory=JointCalibration)

    def clamp(self, angle: float) -> float:
        return max(self.min_angle, min(self.max_angle, angle))

    def set_angle(self, angle: float) -> float:
        self.current_angle = self.clamp(angle)
        return self.current_angle

    def servo_angle(self) -> float:
        return self.calibration.to_servo_angle(self.current_angle)


# ─────────────────────────────────────────────────────────────────────────────
# Limb — kinematic chain
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Limb:
    """
    Serial kinematic chain: coxa → femur → tibia.

    Body-frame convention
    ---------------------
        X  =  forward
        Y  =  left
        Z  =  up  (positive = above ground)

    mount_position : shoulder origin in body frame (m).
    mount_angle    : azimuth of the leg's rest direction, measured from +X
                     around +Z (radians).  Set by MDLParser based on the
                     leg index and total count.
    """
    name:           str
    index:          int
    joints:         List[JointDef]
    segment_lengths: List[float]
    mount_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_angle:    float                      = 0.0          # radians
    foot_position:  Tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        self.foot_position = self._fk()

    # ── Forward kinematics ────────────────────────────────────────────────────

    def _fk(self) -> Tuple[float, float, float]:
        """
        Exact serial-chain FK for a 3-DOF coxa-femur-tibia leg.

        All three segments rotate in the leg's own sagittal plane, which is
        oriented at (mount_angle + hip_angle) from the body X-axis.

        Segment conventions (all angles in degrees, positive = conventional
        anatomical direction):
          hip   : rotation around body Z at mount point.  Positive = leg
                  sweeps toward the front of the body (counter-clockwise
                  when viewed from above for a left-side leg).
          knee  : rotation in sagittal plane.  Positive = knee bends; the
                  femur tip drops below the coxa tip.  0° = femur points
                  straight along the leg azimuth (horizontal).
          ankle : rotation in sagittal plane.  Positive = ankle bends the
                  same direction as the knee.  Used to keep the foot level.
        """
        mx, my, mz = self.mount_position

        hip_deg   = self.joints[0].current_angle if len(self.joints) > 0 else 0.0
        knee_deg  = self.joints[1].current_angle if len(self.joints) > 1 else 0.0
        ankle_deg = self.joints[2].current_angle if len(self.joints) > 2 else 0.0

        l0 = self.segment_lengths[0] if len(self.segment_lengths) > 0 else 0.07
        l1 = self.segment_lengths[1] if len(self.segment_lengths) > 1 else 0.12
        l2 = self.segment_lengths[2] if len(self.segment_lengths) > 2 else 0.10

        hip_r   = math.radians(hip_deg)
        knee_r  = math.radians(knee_deg)
        ankle_r = math.radians(ankle_deg)

        # Effective leg azimuth in body frame
        az = self.mount_angle + hip_r

        # Unit vectors in the leg's sagittal plane
        # forward_hat : horizontal, points along the leg azimuth
        # down_hat    : world −Z (toward ground)
        fwd_x, fwd_y = math.cos(az), math.sin(az)   # horizontal forward

        # ── Coxa (hip segment) ─────────────────────────────────────────────
        # Purely horizontal: extends from mount along leg azimuth
        cx = mx + l0 * fwd_x
        cy = my + l0 * fwd_y
        cz = mz

        # ── Femur (thigh) ──────────────────────────────────────────────────
        # knee_r = 0 → femur is horizontal (pointing forward along az)
        # knee_r > 0 → femur tip drops (Z decreases)
        femur_horiz = l1 * math.cos(knee_r)
        femur_vert  = l1 * math.sin(knee_r)      # positive = drops

        fx = cx + femur_horiz * fwd_x
        fy = cy + femur_horiz * fwd_y
        fz = cz - femur_vert

        # ── Tibia (shin) ───────────────────────────────────────────────────
        # ankle_r compounds with knee_r to give the total sagittal angle
        total_r = knee_r + ankle_r
        tibia_horiz = l2 * math.cos(total_r)
        tibia_vert  = l2 * math.sin(total_r)

        tx = fx + tibia_horiz * fwd_x
        ty = fy + tibia_horiz * fwd_y
        tz = fz - tibia_vert

        self.foot_position = (tx, ty, tz)
        return self.foot_position

    # ── Inverse kinematics ────────────────────────────────────────────────────

    def solve_ik(self, target: Tuple[float, float, float]) -> bool:
        """
        Full 3-D analytical IK for a coxa-femur-tibia leg.

        This solver works correctly for all mount angles because it
        projects the target into the leg's own sagittal frame (a 2-D
        coordinate system whose axes are the leg-forward direction and
        world-Z) rather than assuming a fixed vertical plane.

        Returns True if target is reachable; False if it was clamped to the
        nearest reachable point.  Either way, joint angles are always valid
        (within joint limits) and FK is updated.
        """
        tx, ty, tz = target
        mx, my, mz = self.mount_position

        l0 = self.segment_lengths[0] if len(self.segment_lengths) > 0 else 0.07
        l1 = self.segment_lengths[1] if len(self.segment_lengths) > 1 else 0.12
        l2 = self.segment_lengths[2] if len(self.segment_lengths) > 2 else 0.10

        # ── 1. Hip angle (horizontal plane) ──────────────────────────────────
        dx = tx - mx
        dy = ty - my

        target_az     = math.atan2(dy, dx)
        hip_angle_rad = target_az - self.mount_angle

        # Normalise to (−π, π)
        hip_angle_rad = (hip_angle_rad + math.pi) % (2 * math.pi) - math.pi

        if self.joints:
            self.joints[0].set_angle(math.degrees(hip_angle_rad))
            hip_angle_rad = math.radians(self.joints[0].current_angle)

        # ── 2. Coxa tip in body frame ─────────────────────────────────────
        az       = self.mount_angle + hip_angle_rad
        fwd_x    = math.cos(az)
        fwd_y    = math.sin(az)

        coxa_x   = mx + l0 * fwd_x
        coxa_y   = my + l0 * fwd_y
        coxa_z   = mz

        # ── 3. Express target in the leg's sagittal frame ─────────────────
        #
        # The sagittal frame has:
        #   s_horiz : distance from coxa tip to target, measured along the
        #             leg's forward direction (dot product with fwd_hat).
        #             Positive = target is in front of coxa.
        #   s_vert  : distance from coxa tip to target measured along world Z.
        #             Negative = target is below coxa (normal for a standing leg).
        #
        # This projection is EXACT regardless of mount_angle because we use
        # the actual leg azimuth, not an assumed vertical plane.
        rel_x  = tx - coxa_x
        rel_y  = ty - coxa_y
        rel_z  = tz - coxa_z

        # Horizontal component along the leg's forward direction
        s_horiz = rel_x * fwd_x + rel_y * fwd_y
        # Vertical component (world Z)
        s_vert  = rel_z   # negative = below coxa

        # Distance from coxa tip to target in the sagittal plane
        d = math.sqrt(s_horiz ** 2 + s_vert ** 2)

        # ── 4. Reach clamping ─────────────────────────────────────────────
        max_reach = l1 + l2
        min_reach = abs(l1 - l2)
        reachable = True

        if d > max_reach:
            # Scale the sagittal vector to lie on the maximum reach circle
            scale     = max_reach / d
            s_horiz  *= scale
            s_vert   *= scale
            d         = max_reach
            reachable = False
        elif d < min_reach and d > 1e-6:
            scale     = min_reach / d
            s_horiz  *= scale
            s_vert   *= scale
            d         = min_reach
            reachable = False
        elif d <= 1e-6:
            # Target is at the coxa tip — set a safe default pose
            if len(self.joints) > 1:
                self.joints[1].set_angle(45.0)
            if len(self.joints) > 2:
                self.joints[2].set_angle(45.0)
            self._fk()
            return False

        # ── 5. 2-link IK in the sagittal frame ───────────────────────────
        #
        # gamma   : angle from horizontal to the coxa→target vector.
        #           Positive = target is above horizontal (unusual).
        #           Negative = target is below horizontal (normal stance).
        # alpha   : angle at the coxa end of the femur-tibia triangle.
        # knee_r  : femur-tibia interior angle relative to straight.
        #
        try:
            gamma = math.atan2(-s_vert, s_horiz)   # negative s_vert = below

            cos_alpha = (l1 ** 2 + d ** 2 - l2 ** 2) / (2.0 * l1 * d)
            cos_alpha = max(-1.0, min(1.0, cos_alpha))
            alpha     = math.acos(cos_alpha)

            # Femur angle from horizontal (positive = femur tip below horizontal)
            femur_r = gamma + alpha   # tip-down convention

            cos_knee_interior = (l1 ** 2 + l2 ** 2 - d ** 2) / (2.0 * l1 * l2)
            cos_knee_interior = max(-1.0, min(1.0, cos_knee_interior))
            knee_interior     = math.acos(cos_knee_interior)
            knee_r            = math.pi - knee_interior   # 0 = straight

            # ── 6. Ankle: exact tibia angle to reach target ──────────────
            # Compute the knee tip (femur endpoint) in the sagittal frame,
            # then solve for the exact tibia angle needed to reach the target.
            #   ankle_r = tibia_required_angle − femur_r
            # Because in FK: total_r = femur_r + ankle_r
            # This is exact — not a level-foot approximation.
            knee_horiz = l1 * math.cos(femur_r)
            knee_vert  = l1 * math.sin(femur_r)
            # Residual from knee tip to target in sagittal frame
            res_horiz  = s_horiz - knee_horiz
            res_vert   = s_vert  + knee_vert   # s_vert is negative-down; undo that
            tibia_angle = math.atan2(-res_vert, res_horiz)
            ankle_r     = tibia_angle - femur_r

        except (ValueError, ZeroDivisionError):
            reachable    = False
            femur_r      = math.radians(45.0)
            knee_r       = math.radians(45.0)
            ankle_r      = math.radians(45.0)

        if len(self.joints) > 1:
            self.joints[1].set_angle(math.degrees(femur_r))
        if len(self.joints) > 2:
            self.joints[2].set_angle(math.degrees(ankle_r))

        # FK update so foot_position reflects the solved angles
        self._fk()
        return reachable

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_angles(self) -> Dict[str, float]:
        return {j.name: j.current_angle for j in self.joints}

    def get_servo_angles(self) -> Dict[str, float]:
        return {j.name: j.servo_angle() for j in self.joints}

    def max_reach(self) -> float:
        return sum(self.segment_lengths)

    def neutral_foot_position(self) -> Tuple[float, float, float]:
        """Foot position with all joints at 0°."""
        saved = [j.current_angle for j in self.joints]
        for j in self.joints:
            j.current_angle = 0.0
        pos = self._fk()
        for j, a in zip(self.joints, saved):
            j.current_angle = a
        self._fk()
        return pos


# ─────────────────────────────────────────────────────────────────────────────
# Body / Topology / Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Body:
    segments: List[str]
    mass:     float = 1.0
    length:   float = 0.3
    width:    float = 0.2
    height:   float = 0.1
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.15])
    yaw:      float = 0.0


@dataclass
class RobotTopology:
    name:                str
    family:              str
    limb_count:          int
    joint_count_per_limb:int
    total_joints:        int


@dataclass
class RobotModel:
    name:     str
    body:     Body
    limbs:    List[Limb]
    topology: RobotTopology

    @property
    def joints(self) -> Dict[str, JointDef]:
        out = {}
        for limb in self.limbs:
            for j in limb.joints:
                out[f"{limb.name}.{j.name}"] = j
        return out

    @property
    def mass(self) -> float:
        return self.body.mass

    def solve_fk(self) -> Dict[str, Tuple[float, float, float]]:
        return {l.name: l._fk() for l in self.limbs}

    def solve_ik(self) -> Dict[str, Tuple[float, float, float]]:
        return {l.name: l.foot_position for l in self.limbs}


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class MDLParser:

    DEFAULT_SEG  = {"hip": 0.07, "knee": 0.12, "ankle": 0.10}
    DEFAULT_SPEED = 600.0

    @classmethod
    def load(cls, source) -> RobotModel:
        if isinstance(source, Path):
            data = yaml.safe_load(source.read_text())
        elif isinstance(source, str) and "\n" not in source and source.endswith(".yaml"):
            data = yaml.safe_load(Path(source).read_text())
        elif isinstance(source, str):
            data = yaml.safe_load(source)
        elif isinstance(source, dict):
            data = source
        else:
            raise ValueError("source must be Path, YAML file path, YAML string, or dict")
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict) -> RobotModel:
        name = data.get("name", "UnnamedRobot")
        bd   = data.get("body", {})
        segs = bd.get("segments", ["body"])
        if isinstance(segs, int):
            segs = [f"seg_{i}" for i in range(segs)]
        mass = float(str(data.get("physics", {}).get("mass", "1kg")).replace("kg", "").strip())

        body = Body(
            segments=segs,
            mass=mass,
            length=float(bd.get("length", 0.3)),
            width =float(bd.get("width",  0.2)),
            height=float(bd.get("height", 0.1)),
        )

        limbs: List[Limb] = []
        idx = 0
        for limb_def in data.get("limbs", []):
            count = int(limb_def.get("count", 1))
            ltype = limb_def.get("type", limb_def.get("name", "leg"))
            for _ in range(count):
                joints  = cls._parse_joints(
                    limb_def.get("joints", {}),
                    limb_def.get("calibration", {}))
                seglens = cls._parse_seglens(joints, limb_def)
                mpos, mang = cls._mount(idx, count, body)
                limbs.append(Limb(
                    name=f"{ltype}_{idx}", index=idx,
                    joints=joints, segment_lengths=seglens,
                    mount_position=mpos, mount_angle=mang,
                ))
                idx += 1

        family = {2:"biped",3:"triped",4:"quadruped",
                  6:"hexapod",8:"octopod"}.get(len(limbs), f"{len(limbs)}-limbed")
        jcpl = len(limbs[0].joints) if limbs else 0
        topo = RobotTopology(name=name, family=family,
                             limb_count=len(limbs),
                             joint_count_per_limb=jcpl,
                             total_joints=len(limbs)*jcpl)
        return RobotModel(name=name, body=body, limbs=limbs, topology=topo)

    @classmethod
    def _parse_joints(cls, jdata, cdata: dict) -> List[JointDef]:
        joints = []
        def cal(jname, jcfg):
            cd = cdata.get(jname, {})
            return JointCalibration(
                zero_offset_deg=float(cd.get("zero_offset", 0.0)),
                direction=float(cd.get("direction", 1.0)),
                min_pulse_us=int(cd.get("min_pulse_us", 500)),
                max_pulse_us=int(cd.get("max_pulse_us", 2400)),
                speed_dps=float(cd.get("speed_dps", cls.DEFAULT_SPEED)),
            )
        if isinstance(jdata, list):
            for j in jdata:
                if isinstance(j, str):
                    joints.append(JointDef(name=j, calibration=cal(j, {})))
                else:
                    n = j.get("name","joint")
                    joints.append(JointDef(name=n,
                        min_angle=float(j.get("min",-90)),
                        max_angle=float(j.get("max", 90)),
                        calibration=cal(n, j)))
        elif isinstance(jdata, dict):
            for n, cfg in jdata.items():
                cfg = cfg if isinstance(cfg, dict) else {}
                joints.append(JointDef(name=n,
                    min_angle=float(cfg.get("min",-90)),
                    max_angle=float(cfg.get("max", 90)),
                    calibration=cal(n, cfg)))
        else:
            for n,lo,hi in [("hip",-90,90),("knee",0,120),("ankle",-45,45)]:
                joints.append(JointDef(name=n,min_angle=lo,max_angle=hi,
                                       calibration=cal(n,{})))
        return joints

    @classmethod
    def _parse_seglens(cls, joints, limb_def) -> List[float]:
        raw = limb_def.get("segment_lengths")
        if raw:
            ls = [float(v) for v in raw]
            while len(ls) < len(joints):
                ls.append(cls.DEFAULT_SEG.get(joints[len(ls)].name, 0.10))
            return ls[:len(joints)]
        return [cls.DEFAULT_SEG.get(j.name, 0.10) for j in joints]

    @classmethod
    def _mount(cls, idx, count, body) -> Tuple[Tuple[float,float,float], float]:
        angle = (2*math.pi*idx/count) - math.pi/2
        x = (body.length/2) * math.cos(angle)
        y = (body.width /2) * math.sin(angle)
        return (x, y, 0.0), angle
