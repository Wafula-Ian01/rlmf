"""
Bundled robot YAML definitions.
Access with:
    import rlmf.robots as robots
    path = robots.get("hexapod")   # returns Path to hexapod.yaml
    robot = Robot.load(path)
"""
from pathlib import Path

_ROBOTS_DIR = Path(__file__).parent

_BUNDLED = {p.stem: p for p in _ROBOTS_DIR.glob("*.yaml")}


def get(name: str) -> Path:
    """
    Return the Path to a bundled robot YAML by name (without extension).

    Available: hexapod, spider, quadruped, centipede

    Example::

        import rlmf.robots as robots
        from rlmf import Robot

        robot = Robot.load(robots.get("hexapod"))
    """
    if name not in _BUNDLED:
        available = ", ".join(sorted(_BUNDLED))
        raise KeyError(f"No bundled robot '{name}'. Available: {available}")
    return _BUNDLED[name]


def list_robots() -> list:
    """Return the names of all bundled robot definitions."""
    return sorted(_BUNDLED.keys())
