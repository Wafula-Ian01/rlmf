"""
rlmf — Robot Locomotion & Morphology Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A universal robot body model and locomotion API.
Define any legged robot in YAML, then call robot.walk().

    from rlmf import Robot

    robot = Robot.load("hexapod.yaml")
    robot.walk()
    robot.turn_left(45)
    robot.climb()

The API is identical regardless of morphology (3-leg, 4-leg, 6-leg, 8-leg…).
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rlmf")
except PackageNotFoundError:
    __version__ = "0.0.0"

from rlmf.robot import Robot, RobotState
from rlmf.mdl import (
    MDLParser,
    RobotModel,
    RobotTopology,
    Limb,
    JointDef,
    Body,
)
from rlmf.gaits import (
    Gait,
    GaitFrame,
    TripodGait,
    WaveGait,
    RippleGait,
    TrotGait,
    select_gait,
    GAIT_REGISTRY,
)
from rlmf.balance import BalanceEngine, BalanceState
from rlmf.motors import MotorAbstractionLayer, SimulatedDriver, MotorDriver

__all__ = [
    # Top-level
    "Robot",
    "RobotState",
    # MDL
    "MDLParser",
    "RobotModel",
    "RobotTopology",
    "Limb",
    "JointDef",
    "Body",
    # Gaits
    "Gait",
    "GaitFrame",
    "TripodGait",
    "WaveGait",
    "RippleGait",
    "TrotGait",
    "select_gait",
    "GAIT_REGISTRY",
    # Balance
    "BalanceEngine",
    "BalanceState",
    # Motors
    "MotorAbstractionLayer",
    "SimulatedDriver",
    "MotorDriver",
]
