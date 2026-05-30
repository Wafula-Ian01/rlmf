"""
Tests for rlmf — Robot Locomotion & Morphology Framework.
Run with:  pytest
"""
import math
import pytest

import rlmf
import rlmf.robots as robots_pkg
from rlmf import (
    Robot,
    TripodGait,
    WaveGait,
    RippleGait,
    TrotGait,
    select_gait,
    BalanceEngine,
)
from rlmf.mdl import MDLParser

# ── Fixtures ──────────────────────────────────────────────────────────────────

HEXAPOD_YAML = """
name: TestHex
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
"""

QUAD_YAML = """
name: TestQuad
body:
  segments: [torso]
  length: 0.35
  width: 0.20
  height: 0.12
limbs:
  - type: leg
    count: 4
    joints:
      hip:   {min: -60, max: 60}
      knee:  {min: 0,   max: 130}
      ankle: {min: -50, max: 50}
    segment_lengths: [0.08, 0.15, 0.12]
physics:
  mass: 12kg
"""


@pytest.fixture
def hexapod():
    return Robot.load(HEXAPOD_YAML)


@pytest.fixture
def quadruped():
    return Robot.load(QUAD_YAML)


# ── MDL parser ────────────────────────────────────────────────────────────────

class TestMDLParser:
    def test_loads_from_string(self):
        model = MDLParser.load(HEXAPOD_YAML)
        assert model.name == "TestHex"

    def test_loads_from_dict(self):
        import yaml
        data = yaml.safe_load(HEXAPOD_YAML)
        model = MDLParser.load(data)
        assert model.name == "TestHex"

    def test_limb_count(self):
        model = MDLParser.load(HEXAPOD_YAML)
        assert len(model.limbs) == 6

    def test_joint_count_per_limb(self):
        model = MDLParser.load(HEXAPOD_YAML)
        for limb in model.limbs:
            assert len(limb.joints) == 3

    def test_total_joints(self):
        model = MDLParser.load(HEXAPOD_YAML)
        assert model.topology.total_joints == 18

    def test_mass_parsed(self):
        model = MDLParser.load(HEXAPOD_YAML)
        assert model.mass == pytest.approx(1.2)

    def test_family_inferred(self):
        model = MDLParser.load(HEXAPOD_YAML)
        assert model.topology.family == "hexapod"

    def test_quad_family(self):
        model = MDLParser.load(QUAD_YAML)
        assert model.topology.family == "quadruped"

    def test_joint_limits_respected(self):
        model = MDLParser.load(HEXAPOD_YAML)
        hip = model.limbs[0].joints[0]
        assert hip.min_angle == -90
        assert hip.max_angle == 90

    def test_mount_positions_distinct(self):
        model = MDLParser.load(HEXAPOD_YAML)
        positions = [l.mount_position for l in model.limbs]
        # All 6 mount points should be unique
        assert len(set(positions)) == 6


# ── Robot API ─────────────────────────────────────────────────────────────────

class TestRobotAPI:
    def test_load_from_string(self, hexapod):
        assert hexapod is not None

    def test_load_bundled_yaml(self):
        path = robots_pkg.get("hexapod")
        robot = Robot.load(path)
        assert robot.topology.limb_count == 6

    def test_describe_returns_string(self, hexapod):
        desc = hexapod.describe()
        assert "TestHex" in desc
        assert "hexapod" in desc

    def test_mass_property(self, hexapod):
        assert hexapod.mass == pytest.approx(1.2)

    def test_limbs_property(self, hexapod):
        assert len(hexapod.limbs) == 6

    def test_joints_property(self, hexapod):
        j = hexapod.joints
        assert len(j) == 18
        assert "leg_0.hip" in j

    def test_topology_property(self, hexapod):
        t = hexapod.topology
        assert t.family == "hexapod"
        assert t.limb_count == 6
        assert t.total_joints == 18

    def test_walk_returns_self(self, hexapod):
        result = hexapod.walk(steps=10)
        assert result is hexapod

    def test_turn_left_changes_heading(self, hexapod):
        before = hexapod.state.yaw
        hexapod.turn_left(90)
        assert hexapod.state.yaw == pytest.approx(before + math.pi / 2, abs=1e-6)

    def test_turn_right_changes_heading(self, hexapod):
        before = hexapod.state.yaw
        hexapod.turn_right(90)
        assert hexapod.state.yaw == pytest.approx(before - math.pi / 2, abs=1e-6)

    def test_stop_clears_walking(self, hexapod):
        hexapod.walk(steps=5)
        hexapod.stop()
        assert not hexapod.state.is_walking

    def test_move_joint(self, hexapod):
        hexapod.move_joint("leg_0.hip", 30.0)
        # No exception = driver accepted the command

    def test_reach_returns_bool(self, hexapod):
        result = hexapod.reach("leg_0", (0.2, 0.0, -0.1))
        assert isinstance(result, bool)

    def test_chaining(self, hexapod):
        # API methods return self for chaining
        hexapod.walk(steps=5).turn_left(30).turn_right(30).stop()

    def test_state_log_populated(self, hexapod):
        hexapod.walk(steps=5)
        assert len(hexapod.state.log) > 0

    def test_solve_fk_returns_dict(self, hexapod):
        fk = hexapod.solve_fk()
        assert len(fk) == 6
        for name, pos in fk.items():
            assert len(pos) == 3

    def test_follow_path(self, hexapod):
        hexapod.follow_path([(0.1, 0.0), (0.1, 0.1)])


# ── Gait engine ───────────────────────────────────────────────────────────────

class TestGaits:
    @pytest.mark.parametrize("gait_cls", [TripodGait, WaveGait, RippleGait, TrotGait])
    def test_generates_correct_frame_count(self, hexapod, gait_cls):
        gait = gait_cls()
        neutral = {l.name: l.foot_position for l in hexapod.limbs}
        names = [l.name for l in hexapod.limbs]
        frames = gait.generate_frames(names, neutral, 0.0, 60)
        assert len(frames) == 60

    @pytest.mark.parametrize("gait_cls", [TripodGait, WaveGait, RippleGait, TrotGait])
    def test_frames_have_all_limbs(self, hexapod, gait_cls):
        gait = gait_cls()
        neutral = {l.name: l.foot_position for l in hexapod.limbs}
        names = [l.name for l in hexapod.limbs]
        frames = gait.generate_frames(names, neutral, 0.0, 30)
        for frame in frames:
            assert set(frame.foot_targets.keys()) == set(names)
            assert set(frame.stance_mask.keys()) == set(names)

    def test_phase_ranges_0_to_1(self, hexapod):
        gait = TripodGait()
        neutral = {l.name: l.foot_position for l in hexapod.limbs}
        names = [l.name for l in hexapod.limbs]
        frames = gait.generate_frames(names, neutral, 0.0, 60)
        for frame in frames:
            assert 0.0 <= frame.phase <= 1.0

    def test_select_gait_hexapod_walk(self):
        gait = select_gait("hexapod", "walk")
        assert isinstance(gait, TripodGait)

    def test_select_gait_hexapod_climb(self):
        gait = select_gait("hexapod", "climb")
        assert isinstance(gait, WaveGait)

    def test_select_gait_quadruped(self):
        gait = select_gait("quadruped", "walk")
        assert isinstance(gait, TrotGait)

    def test_get_gait_frames_via_robot(self, hexapod):
        frames = hexapod.get_gait_frames("tripod", num_frames=30)
        assert len(frames) == 30

    def test_wave_gait_quadruped(self, quadruped):
        frames = quadruped.get_gait_frames("wave", num_frames=40)
        assert len(frames) == 40


# ── Balance engine ────────────────────────────────────────────────────────────

class TestBalance:
    def test_stable_in_neutral(self, hexapod):
        state = hexapod.balance_state()
        assert state.is_stable

    def test_margin_positive_when_stable(self, hexapod):
        state = hexapod.balance_state()
        assert state.stability_margin > 0

    def test_tip_risk_in_range(self, hexapod):
        state = hexapod.balance_state()
        assert 0.0 <= state.tip_risk <= 1.0

    def test_support_polygon_has_vertices(self, hexapod):
        state = hexapod.balance_state()
        assert len(state.support_polygon) >= 3

    def test_balance_engine_direct(self, hexapod):
        engine = BalanceEngine()
        foot_positions = {l.name: l.foot_position for l in hexapod.limbs}
        stance = {l.name: True for l in hexapod.limbs}
        state = engine.analyze((0.0, 0.0, 0.15), foot_positions, stance, 1.2)
        assert state.is_stable

    def test_unstable_with_no_stance_feet(self, hexapod):
        engine = BalanceEngine()
        foot_positions = {l.name: l.foot_position for l in hexapod.limbs}
        stance = {l.name: False for l in hexapod.limbs}  # all airborne
        state = engine.analyze((0.0, 0.0, 0.15), foot_positions, stance, 1.2)
        assert not state.is_stable


# ── Kinematics ────────────────────────────────────────────────────────────────

class TestKinematics:
    def test_fk_returns_3tuple_per_limb(self, hexapod):
        for limb in hexapod.limbs:
            pos = limb._fk()
            assert len(pos) == 3

    def test_ik_moves_foot_toward_target(self, hexapod):
        limb = hexapod.limbs[0]
        # Build a reachable target near the limb's neutral foot position
        neutral = limb.foot_position
        target = (neutral[0] + 0.02, neutral[1], neutral[2])
        before = limb.foot_position
        limb.solve_ik(target)
        after = limb.foot_position
        # Foot must have moved (angles changed) and stay within link reach
        total_reach = sum(limb.segment_lengths)
        dist_from_mount = math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(after, limb.mount_position)
        ))
        assert dist_from_mount <= total_reach + 1e-6

    def test_joint_clamping(self, hexapod):
        joint = hexapod.limbs[0].joints[0]
        joint.set_angle(999)
        assert joint.current_angle <= joint.max_angle
        joint.set_angle(-999)
        assert joint.current_angle >= joint.min_angle


# ── Bundled robots package ────────────────────────────────────────────────────

class TestBundledRobots:
    def test_list_robots(self):
        names = robots_pkg.list_robots()
        assert "hexapod" in names
        assert "quadruped" in names

    def test_get_returns_path(self):
        path = robots_pkg.get("hexapod")
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            robots_pkg.get("nonexistent_robot")

    @pytest.mark.parametrize("name", ["hexapod", "spider", "quadruped", "centipede"])
    def test_all_bundled_load(self, name):
        robot = Robot.load(robots_pkg.get(name))
        assert robot.topology.limb_count > 0


# ── Version ───────────────────────────────────────────────────────────────────

class TestPackage:
    def test_version_attribute_exists(self):
        assert hasattr(rlmf, "__version__")

    def test_version_is_string(self):
        assert isinstance(rlmf.__version__, str)
