"""
Gait Engine — where walking emerges from joint coordination.
Generates footstep sequences and phase offsets for different gait patterns.
"""
from __future__ import annotations
import math
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class GaitFrame:
    """One snapshot of a gait cycle."""
    foot_targets: Dict[str, Tuple[float, float, float]]   # limb_name → (x,y,z)
    stance_mask: Dict[str, bool]                           # limb_name → on_ground
    phase: float                                           # 0.0 → 1.0 in cycle
    timestamp: float = field(default_factory=time.time)


class Gait(ABC):
    """Base class for all gait patterns."""
    name: str = "base"

    def __init__(self, step_length: float = 0.05, step_height: float = 0.04,
                 cycle_time: float = 1.0):
        self.step_length = step_length
        self.step_height = step_height
        self.cycle_time = cycle_time

    @abstractmethod
    def generate_frames(
        self,
        limb_names: List[str],
        neutral_positions: Dict[str, Tuple[float, float, float]],
        direction: float = 0.0,          # radians, 0 = forward
        num_frames: int = 60,
    ) -> List[GaitFrame]:
        ...

    def _swing_trajectory(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        t: float,                         # 0..1 within swing phase
    ) -> Tuple[float, float, float]:
        """Smooth foot trajectory during swing (lifted phase)."""
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        # Raised parabola
        z = start[2] + self.step_height * math.sin(math.pi * t)
        return (x, y, z)

    def _step_target(
        self,
        neutral: Tuple[float, float, float],
        direction: float,
        phase: float,            # -0.5 .. +0.5 relative to neutral
    ) -> Tuple[float, float, float]:
        dx = self.step_length * phase * math.cos(direction)
        dy = self.step_length * phase * math.sin(direction)
        return (neutral[0] + dx, neutral[1] + dy, neutral[2])


class TripodGait(Gait):
    """
    Classic hexapod tripod: legs split into two groups of 3.
    Group A swings while Group B stances, then alternate.
    Duty factor ≈ 0.5, fastest stable hexapod gait.
    """
    name = "tripod"

    def generate_frames(self, limb_names, neutral_positions, direction=0.0, num_frames=60):
        n = len(limb_names)
        group_a = [limb_names[i] for i in range(0, n, 2)]
        group_b = [limb_names[i] for i in range(1, n, 2)]

        frames = []
        for fi in range(num_frames):
            phase = fi / num_frames       # 0..1
            foot_targets = {}
            stance_mask = {}

            for name in limb_names:
                neutral = neutral_positions[name]
                in_group_a = name in group_a

                # Group A swings in first half, B in second half
                if in_group_a:
                    local_phase = phase * 2 % 1.0
                    is_swing = phase < 0.5
                else:
                    local_phase = (phase * 2 + 1.0) % 1.0
                    is_swing = phase >= 0.5

                if is_swing:
                    t = local_phase
                    start = self._step_target(neutral, direction, -0.5)
                    end = self._step_target(neutral, direction, +0.5)
                    pos = self._swing_trajectory(start, end, t)
                else:
                    # Stance: foot moves backward relative to body
                    stance_t = local_phase
                    pos = self._step_target(neutral, direction, 0.5 - stance_t)

                foot_targets[name] = pos
                stance_mask[name] = not is_swing

            frames.append(GaitFrame(foot_targets, stance_mask, phase))

        return frames


class WaveGait(Gait):
    """
    Wave gait: one leg lifts at a time in sequence.
    Most stable (5 legs always on ground for hexapod), slowest.
    """
    name = "wave"

    def generate_frames(self, limb_names, neutral_positions, direction=0.0, num_frames=60):
        n = len(limb_names)
        duty = (n - 1) / n          # fraction of time each leg is in stance

        frames = []
        for fi in range(num_frames):
            phase = fi / num_frames
            foot_targets = {}
            stance_mask = {}

            for idx, name in enumerate(limb_names):
                neutral = neutral_positions[name]
                leg_phase = (phase + idx / n) % 1.0
                swing_fraction = 1.0 - duty

                is_swing = leg_phase >= duty
                if is_swing:
                    t = (leg_phase - duty) / swing_fraction
                    start = self._step_target(neutral, direction, -0.5)
                    end = self._step_target(neutral, direction, +0.5)
                    pos = self._swing_trajectory(start, end, t)
                else:
                    stance_t = leg_phase / duty
                    pos = self._step_target(neutral, direction, 0.5 - stance_t)

                foot_targets[name] = pos
                stance_mask[name] = not is_swing

            frames.append(GaitFrame(foot_targets, stance_mask, phase))

        return frames


class RippleGait(Gait):
    """
    Ripple gait: legs swing in overlapping sequences.
    Balance between wave (stability) and tripod (speed).
    """
    name = "ripple"

    def generate_frames(self, limb_names, neutral_positions, direction=0.0, num_frames=60):
        n = len(limb_names)
        # Divide legs into 3 groups, each offset by 1/3 of cycle
        groups = [[], [], []]
        for i, name in enumerate(limb_names):
            groups[i % 3].append(name)

        frames = []
        for fi in range(num_frames):
            phase = fi / num_frames
            foot_targets = {}
            stance_mask = {}

            for gidx, group in enumerate(groups):
                group_phase = (phase + gidx / 3) % 1.0
                swing_fraction = 1.0 / 3.0
                is_swing = group_phase < swing_fraction

                for name in group:
                    neutral = neutral_positions[name]
                    if is_swing:
                        t = group_phase / swing_fraction
                        start = self._step_target(neutral, direction, -0.5)
                        end = self._step_target(neutral, direction, +0.5)
                        pos = self._swing_trajectory(start, end, t)
                    else:
                        stance_t = (group_phase - swing_fraction) / (1 - swing_fraction)
                        pos = self._step_target(neutral, direction, 0.5 - stance_t)

                    foot_targets[name] = pos
                    stance_mask[name] = not is_swing

            frames.append(GaitFrame(foot_targets, stance_mask, phase))

        return frames


class TrotGait(Gait):
    """
    Diagonal trot: for quadrupeds, diagonal pairs move together.
    Fast and efficient.
    """
    name = "trot"

    def generate_frames(self, limb_names, neutral_positions, direction=0.0, num_frames=60):
        n = len(limb_names)
        # Diagonal pairs: 0+3, 1+2 for quadruped
        pair_a = [limb_names[i] for i in [0, 3] if i < n]
        pair_b = [limb_names[i] for i in [1, 2] if i < n]
        # Fall back to even/odd split
        if not pair_a:
            pair_a = [limb_names[i] for i in range(0, n, 2)]
            pair_b = [limb_names[i] for i in range(1, n, 2)]

        frames = []
        for fi in range(num_frames):
            phase = fi / num_frames
            foot_targets = {}
            stance_mask = {}

            for name in limb_names:
                neutral = neutral_positions[name]
                is_pair_a = name in pair_a
                local_phase = phase if is_pair_a else (phase + 0.5) % 1.0
                is_swing = local_phase < 0.4

                if is_swing:
                    t = local_phase / 0.4
                    start = self._step_target(neutral, direction, -0.5)
                    end = self._step_target(neutral, direction, +0.5)
                    pos = self._swing_trajectory(start, end, t)
                else:
                    stance_t = (local_phase - 0.4) / 0.6
                    pos = self._step_target(neutral, direction, 0.5 - stance_t)

                foot_targets[name] = pos
                stance_mask[name] = not is_swing

            frames.append(GaitFrame(foot_targets, stance_mask, phase))

        return frames


# Registry
GAIT_REGISTRY: Dict[str, type] = {
    "tripod": TripodGait,
    "wave": WaveGait,
    "ripple": RippleGait,
    "trot": TrotGait,
}


def select_gait(family: str, behavior: str = "walk") -> Gait:
    """Select the most appropriate gait for a given morphology family."""
    rules = {
        ("hexapod", "walk"):  TripodGait,
        ("hexapod", "slow"):  WaveGait,
        ("hexapod", "climb"): WaveGait,
        ("quadruped", "walk"): TrotGait,
        ("quadruped", "slow"): WaveGait,
        ("octopod", "walk"):  RippleGait,
        ("triped", "walk"):   TripodGait,
    }
    gait_cls = rules.get((family, behavior), TripodGait)
    return gait_cls()
