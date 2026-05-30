"""
Morphology Description Language (MDL) parser.
Converts YAML robot definitions into structured Robot objects.
"""
from __future__ import annotations
import math
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class JointDef:
    name: str
    min_angle: float = -90.0
    max_angle: float = 90.0
    current_angle: float = 0.0

    def clamp(self, angle: float) -> float:
        return max(self.min_angle, min(self.max_angle, angle))

    def set_angle(self, angle: float):
        self.current_angle = self.clamp(angle)


@dataclass
class Limb:
    name: str
    index: int
    joints: List[JointDef]
    segment_lengths: List[float]       # length per segment (per joint)
    mount_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # x, y, z from body center
    mount_angle: float = 0.0          # radians, around Z axis
    foot_position: Tuple[float, float, float] = field(init=False)

    def __post_init__(self):
        self.foot_position = self._fk()

    def _fk(self) -> Tuple[float, float, float]:
        """Forward kinematics: joint angles → foot position."""
        x, y, z = self.mount_position
        angle_accum = self.mount_angle

        for i, joint in enumerate(self.joints):
            rad = math.radians(joint.current_angle)
            seg_len = self.segment_lengths[i] if i < len(self.segment_lengths) else 1.0

            if joint.name == "hip":
                angle_accum += rad
                x += seg_len * math.cos(angle_accum)
                y += seg_len * math.sin(angle_accum)
            elif joint.name in ("knee", "ankle"):
                z -= seg_len * math.sin(rad)
                x += seg_len * math.cos(rad) * math.cos(angle_accum)
                y += seg_len * math.cos(rad) * math.sin(angle_accum)

        self.foot_position = (x, y, z)
        return self.foot_position

    def solve_ik(self, target: Tuple[float, float, float]) -> bool:
        """
        Inverse kinematics: foot target → joint angles.
        Uses analytical 2-link IK for the vertical plane (knee+ankle joints),
        and direct atan2 for the horizontal hip joint.
        Returns True if target is reachable.
        """
        tx, ty, tz = target
        mx, my, mz = self.mount_position

        dx = tx - mx
        dy = ty - my
        dz = tz - mz

        # Hip: horizontal angle
        hip_angle = math.degrees(math.atan2(dy, dx)) - math.degrees(self.mount_angle)
        if self.joints:
            self.joints[0].set_angle(hip_angle)

        # Reach in the sagittal plane for knee+ankle
        horiz_dist = math.sqrt(dx ** 2 + dy ** 2)
        if len(self.joints) >= 2 and len(self.segment_lengths) >= 2:
            l1 = self.segment_lengths[1] if len(self.segment_lengths) > 1 else 1.0
            l2 = self.segment_lengths[2] if len(self.segment_lengths) > 2 else 1.0
            d = math.sqrt(horiz_dist ** 2 + dz ** 2)
            if d > l1 + l2:
                d = l1 + l2  # clamp to max reach
                reachable = False
            else:
                reachable = True

            try:
                cos_knee = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
                cos_knee = max(-1.0, min(1.0, cos_knee))
                knee_angle = math.degrees(math.acos(cos_knee)) - 180.0

                cos_a1 = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
                cos_a1 = max(-1.0, min(1.0, cos_a1))
                a1 = math.degrees(math.atan2(-dz, horiz_dist))
                a2 = math.degrees(math.acos(cos_a1))
                ankle_angle = a1 + a2

                self.joints[1].set_angle(knee_angle)
                if len(self.joints) >= 3:
                    self.joints[2].set_angle(ankle_angle)
            except (ValueError, ZeroDivisionError):
                reachable = False
        else:
            reachable = True

        self._fk()
        return reachable

    def get_angles(self) -> Dict[str, float]:
        return {j.name: j.current_angle for j in self.joints}


@dataclass
class Body:
    segments: List[str]
    mass: float = 1.0
    length: float = 0.3
    width: float = 0.2
    height: float = 0.1
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.15])
    yaw: float = 0.0


@dataclass
class RobotTopology:
    name: str
    family: str       # "hexapod", "quadruped", "biped", "octopod", etc.
    limb_count: int
    joint_count_per_limb: int
    total_joints: int


class MDLParser:
    DEFAULT_SEGMENT_LENGTHS = {
        "hip": 0.08,
        "knee": 0.12,
        "ankle": 0.10,
    }

    @classmethod
    def load(cls, source) -> "RobotModel":
        if isinstance(source, str) and source.endswith(".yaml"):
            with open(source) as f:
                data = yaml.safe_load(f)
        elif isinstance(source, str):
            data = yaml.safe_load(source)
        elif isinstance(source, dict):
            data = source
        else:
            raise ValueError("source must be a YAML string, file path, or dict")

        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict) -> "RobotModel":
        name = data.get("name", "UnnamedRobot")

        # Parse body
        body_data = data.get("body", {})
        segments = body_data.get("segments", ["body"])
        if isinstance(segments, int):
            segments = [f"segment_{i}" for i in range(segments)]
        mass_raw = data.get("physics", {}).get("mass", "1kg")
        mass = float(str(mass_raw).replace("kg", "").strip())
        body = Body(
            segments=segments,
            mass=mass,
            length=body_data.get("length", 0.3),
            width=body_data.get("width", 0.2),
            height=body_data.get("height", 0.1),
        )

        # Parse limbs
        limbs_data = data.get("limbs", [])
        limbs = []
        limb_index = 0

        for limb_def in limbs_data:
            count = limb_def.get("count", 1)
            ltype = limb_def.get("type", limb_def.get("name", "limb"))
            joints_data = limb_def.get("joints", {})

            for i in range(count):
                joints = cls._parse_joints(joints_data, ltype)
                seg_lengths = cls._segment_lengths(joints, limb_def)
                mount_pos, mount_angle = cls._mount_position(
                    limb_index, count, body, ltype
                )
                limb = Limb(
                    name=f"{ltype}_{limb_index}",
                    index=limb_index,
                    joints=joints,
                    segment_lengths=seg_lengths,
                    mount_position=mount_pos,
                    mount_angle=mount_angle,
                )
                limbs.append(limb)
                limb_index += 1

        # Topology
        limb_count = len(limbs)
        jcpl = len(limbs[0].joints) if limbs else 0
        family = cls._infer_family(limb_count)
        topology = RobotTopology(
            name=name,
            family=family,
            limb_count=limb_count,
            joint_count_per_limb=jcpl,
            total_joints=limb_count * jcpl,
        )

        return RobotModel(name=name, body=body, limbs=limbs, topology=topology)

    @classmethod
    def _parse_joints(cls, joints_data, limb_type: str) -> List[JointDef]:
        joints = []
        if isinstance(joints_data, list):
            for j in joints_data:
                if isinstance(j, str):
                    joints.append(JointDef(name=j))
                elif isinstance(j, dict):
                    name = j.get("name", "joint")
                    joints.append(JointDef(
                        name=name,
                        min_angle=j.get("min", -90),
                        max_angle=j.get("max", 90),
                    ))
        elif isinstance(joints_data, dict):
            for jname, jcfg in joints_data.items():
                if isinstance(jcfg, dict):
                    joints.append(JointDef(
                        name=jname,
                        min_angle=jcfg.get("min", -90),
                        max_angle=jcfg.get("max", 90),
                    ))
                else:
                    joints.append(JointDef(name=jname))
        else:
            # default 3-DOF leg
            joints = [
                JointDef("hip", -90, 90),
                JointDef("knee", 0, 120),
                JointDef("ankle", -45, 45),
            ]
        return joints

    @classmethod
    def _segment_lengths(cls, joints: List[JointDef], limb_def: dict) -> List[float]:
        lengths = limb_def.get("segment_lengths", None)
        if lengths:
            return [float(l) for l in lengths]
        return [cls.DEFAULT_SEGMENT_LENGTHS.get(j.name, 0.10) for j in joints]

    @classmethod
    def _mount_position(cls, index: int, count: int, body: Body, ltype: str):
        """Evenly distribute limbs around the body perimeter."""
        angle = (2 * math.pi * index / count) - math.pi / 2
        r_x = body.length / 2
        r_y = body.width / 2
        x = r_x * math.cos(angle)
        y = r_y * math.sin(angle)
        z = 0.0
        return (x, y, z), angle

    @classmethod
    def _infer_family(cls, limb_count: int) -> str:
        families = {
            2: "biped",
            3: "triped",
            4: "quadruped",
            6: "hexapod",
            8: "octopod",
        }
        return families.get(limb_count, f"{limb_count}-limbed")


@dataclass
class RobotModel:
    name: str
    body: Body
    limbs: List[Limb]
    topology: RobotTopology

    @property
    def joints(self) -> Dict[str, JointDef]:
        result = {}
        for limb in self.limbs:
            for j in limb.joints:
                result[f"{limb.name}.{j.name}"] = j
        return result

    @property
    def mass(self) -> float:
        return self.body.mass

    def solve_ik(self) -> Dict[str, Tuple[float, float, float]]:
        return {limb.name: limb.foot_position for limb in self.limbs}

    def solve_fk(self) -> Dict[str, Tuple[float, float, float]]:
        return {limb.name: limb._fk() for limb in self.limbs}
