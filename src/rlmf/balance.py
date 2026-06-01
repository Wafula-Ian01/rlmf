"""
Balance Engine — stability analysis for legged robots.
Tracks center of mass, support polygon, and stability margin.
"""
from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class BalanceState:
    center_of_mass: Tuple[float, float, float]
    support_polygon: List[Tuple[float, float]]     # convex hull of stance feet (x, y)
    stability_margin: float                         # min distance from CoM projection to polygon edge
    is_stable: bool
    tip_risk: float                                 # 0..1


def _cross_2d(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain convex hull."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_to_segment_distance(px, py, ax, ay, bx, by) -> float:
    """Signed distance from point (px,py) to line segment (a→b)."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)


def point_in_polygon(px, py, polygon: List[Tuple[float, float]]) -> bool:
    """Ray casting algorithm."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def stability_margin(com_xy: Tuple[float, float],
                     polygon: List[Tuple[float, float]]) -> float:
    """
    Minimum distance from CoM projection to any edge of the support polygon.
    Negative if CoM is outside (unstable).
    """
    if len(polygon) < 3:
        return -1.0

    inside = point_in_polygon(com_xy[0], com_xy[1], polygon)
    min_dist = float("inf")

    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        d = point_to_segment_distance(com_xy[0], com_xy[1], ax, ay, bx, by)
        min_dist = min(min_dist, d)

    return min_dist if inside else -min_dist


class BalanceEngine:
    def __init__(self, com_height: float = 0.15):
        self.com_height = com_height

    def analyze(
        self,
        body_position: Tuple[float, float, float],
        foot_positions: Dict[str, Tuple[float, float, float]],
        stance_mask: Dict[str, bool],
        body_mass: float = 1.0,
    ) -> BalanceState:
        # Center of mass: body + weighted foot positions
        stance_feet = [
            pos for name, pos in foot_positions.items()
            if stance_mask.get(name, True)
        ]

        if not stance_feet:
            return BalanceState(
                center_of_mass=body_position,
                support_polygon=[],
                stability_margin=-1.0,
                is_stable=False,
                tip_risk=1.0,
            )

        # Simple CoM: body position projected down
        com = (body_position[0], body_position[1], body_position[2])

        # Support polygon from stance feet
        foot_2d = [(f[0], f[1]) for f in stance_feet]
        polygon = convex_hull(foot_2d)

        margin = stability_margin((com[0], com[1]), polygon)
        is_stable = margin > 0.005   # 5mm threshold

        # Tip risk: inversely proportional to margin, normalized
        max_expected_margin = 0.12   # 12cm is "very stable"
        tip_risk = max(0.0, min(1.0, 1.0 - (margin / max_expected_margin))) if is_stable else 1.0

        return BalanceState(
            center_of_mass=com,
            support_polygon=polygon,
            stability_margin=margin,
            is_stable=is_stable,
            tip_risk=tip_risk,
        )
