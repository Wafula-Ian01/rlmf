"""
rlmf — Robot Morphology & Locomotion Framework v0.4
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("rlmf")
except PackageNotFoundError:
    __version__ = "0.4.0"

from rlmf.robot import Robot, RobotState
from rlmf.mdl import (
    MDLParser, RobotModel, RobotTopology,
    Limb, JointDef, Body, JointCalibration,
)
from rlmf.gaits import (
    Gait, GaitFrame,
    TripodGait, WaveGait, RippleGait, TrotGait,
    select_gait, GAIT_REGISTRY,
)
from rlmf.balance import BalanceEngine, BalanceState
from rlmf.motors import (
    MotorDriver, MotorAbstractionLayer, SimulatedDriver,
    PCA9685Driver, ChainedPCA9685Driver, TrajectoryInterpolator,
)
from rlmf.feedback import (
    IMUSensor, MPU6050Sensor, SimulatedIMU,
    ContactSensor, FSRContactSensor, CurrentContactSensor,
    SimulatedContactSensor, FeedbackReader,
    FeedbackState, IMUReading, ContactReading,
)
from rlmf.controller import (
    BalanceController, CorrectionVector,
    TorqueEstimator, TorqueReport,
    ThermalMonitor, ThermalState,
    PowerMonitor, PowerState,
    PoseCorrector, ContactCalibrator,
)

__all__ = [
    "Robot", "RobotState",
    "MDLParser", "RobotModel", "RobotTopology",
    "Limb", "JointDef", "Body", "JointCalibration",
    "Gait", "GaitFrame",
    "TripodGait", "WaveGait", "RippleGait", "TrotGait",
    "select_gait", "GAIT_REGISTRY",
    "BalanceEngine", "BalanceState",
    "MotorDriver", "MotorAbstractionLayer", "SimulatedDriver",
    "PCA9685Driver", "ChainedPCA9685Driver", "TrajectoryInterpolator",
    "IMUSensor", "MPU6050Sensor", "SimulatedIMU",
    "ContactSensor", "FSRContactSensor", "CurrentContactSensor",
    "SimulatedContactSensor", "FeedbackReader",
    "FeedbackState", "IMUReading", "ContactReading",
    "BalanceController", "CorrectionVector",
    "TorqueEstimator", "TorqueReport",
    "ThermalMonitor", "ThermalState",
    "PowerMonitor", "PowerState",
    "PoseCorrector", "ContactCalibrator",
]
