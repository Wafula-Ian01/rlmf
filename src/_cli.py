"""
rlmf command-line interface.

Usage examples:
    rlmf describe hexapod
    rlmf describe path/to/mybot.yaml
    rlmf robots
    rlmf walk hexapod --gait tripod --steps 60
    rlmf balance hexapod
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve(name_or_path: str) -> str | Path:
    """Accept either a bundled robot name or a file path."""
    p = Path(name_or_path)
    if p.exists():
        return p
    try:
        import rlmf.robots as robots_pkg
        return robots_pkg.get(name_or_path)
    except KeyError:
        print(f"Error: '{name_or_path}' is not a file path or a known bundled robot.", file=sys.stderr)
        print("Run  rlmf robots  to list bundled robots.", file=sys.stderr)
        sys.exit(1)


def cmd_robots(_args):
    import rlmf.robots as robots_pkg
    names = robots_pkg.list_robots()
    print("Bundled robots:")
    for n in names:
        print(f"  {n}")


def cmd_describe(args):
    from rlmf import Robot
    robot = Robot.load(_resolve(args.robot))
    print(robot.describe())
    print()
    t = robot.topology
    print(f"  Gait family:  {t.family}")
    print(f"  Limb count:   {t.limb_count}")
    print(f"  Total joints: {t.total_joints}")


def cmd_walk(args):
    from rlmf import Robot
    robot = Robot.load(_resolve(args.robot))
    print(f"Running {args.gait} gait for {args.steps} frames…")
    frames = robot.get_gait_frames(args.gait, num_frames=args.steps)
    print(f"Generated {len(frames)} frames.")
    state = robot.balance_state()
    print(f"Balance: stable={state.is_stable}, margin={state.stability_margin*100:.1f} cm")


def cmd_balance(args):
    from rlmf import Robot
    robot = Robot.load(_resolve(args.robot))
    state = robot.balance_state()
    print(f"Center of mass:   ({state.center_of_mass[0]:.3f}, {state.center_of_mass[1]:.3f}, {state.center_of_mass[2]:.3f})")
    print(f"Support polygon:  {len(state.support_polygon)} vertices")
    print(f"Stability margin: {state.stability_margin * 100:.1f} cm")
    print(f"Stable:           {state.is_stable}")
    print(f"Tip risk:         {state.tip_risk * 100:.0f}%")


def main():
    parser = argparse.ArgumentParser(
        prog="rlmf",
        description="Robot Morphology & Locomotion Framework CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # rlmf robots
    sub.add_parser("robots", help="List bundled robot definitions")

    # rlmf describe <robot>
    p_desc = sub.add_parser("describe", help="Print robot topology")
    p_desc.add_argument("robot", help="Bundled name (e.g. hexapod) or path to YAML")

    # rlmf walk <robot>
    p_walk = sub.add_parser("walk", help="Simulate a gait and report stats")
    p_walk.add_argument("robot", help="Bundled name or YAML path")
    p_walk.add_argument("--gait",  default="tripod",
                        choices=["tripod", "wave", "ripple", "trot"],
                        help="Gait pattern (default: tripod)")
    p_walk.add_argument("--steps", default=60, type=int,
                        help="Number of gait frames (default: 60)")

    # rlmf balance <robot>
    p_bal = sub.add_parser("balance", help="Print balance state in neutral stance")
    p_bal.add_argument("robot", help="Bundled name or YAML path")

    args = parser.parse_args()
    dispatch = {
        "robots":   cmd_robots,
        "describe": cmd_describe,
        "walk":     cmd_walk,
        "balance":  cmd_balance,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
