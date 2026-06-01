"""
Motor Abstraction Layer

Key fixes over v0.1:
  - set_angle() now takes the FRAMEWORK angle; the driver applies
    JointCalibration.to_servo_angle() before sending to hardware,
    so calibration offsets and direction flips are handled automatically.
  - TrajectoryInterpolator: enforces servo speed limits so the gait engine
    cannot advance faster than the hardware can move.
  - ChainedPCA9685Driver: supports >16 channels via two boards (0x40 + 0x41),
    eliminating the channel-17 unconnected bug.
  - SimulatedDriver: realistic speed simulation with tick() for test loops.
"""
from __future__ import annotations

import time
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class MotorDriver(ABC):
    """
    Hardware driver interface.

    IMPORTANT: set_angle() receives the SERVO angle (0–180°), already converted
    from the framework angle by JointCalibration. Drivers should NOT do any
    further angle mapping unless they have hardware-specific requirements.
    """

    @abstractmethod
    def set_servo_angle(self, joint_id: str, servo_angle_deg: float) -> None:
        """Send a 0–180° command to the physical servo."""
        ...

    @abstractmethod
    def get_servo_angle(self, joint_id: str) -> float:
        """Read back the last commanded servo angle (0–180°)."""
        ...

    @abstractmethod
    def enable(self, joint_id: str) -> None: ...

    @abstractmethod
    def disable(self, joint_id: str) -> None: ...

    # Convenience shim so old call-sites keep working
    def set_angle(self, joint_id: str, angle_deg: float, speed: float = 1.0) -> None:
        self.set_servo_angle(joint_id, angle_deg)

    def get_angle(self, joint_id: str) -> float:
        return self.get_servo_angle(joint_id)


# ─────────────────────────────────────────────────────────────────────────────
# Simulated driver (dev / test)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedDriver(MotorDriver):
    """
    In-memory driver that simulates realistic servo movement speed.

    Call tick(dt) in your control loop to advance servo positions.
    This catches gait-vs-servo timing bugs before they reach hardware.
    """

    def __init__(self, default_speed_dps: float = 600.0):
        self._targets:  Dict[str, float] = {}
        self._current:  Dict[str, float] = {}
        self._enabled:  Dict[str, bool]  = {}
        self.speed_dps = default_speed_dps

    def set_servo_angle(self, joint_id: str, servo_angle_deg: float) -> None:
        servo_angle_deg = max(0.0, min(180.0, servo_angle_deg))
        self._targets[joint_id] = servo_angle_deg
        # Snap _current immediately so get_servo_angle reflects the command right away.
        # tick() advances _current at rated speed_dps for animation/timing tests.
        self._current[joint_id] = servo_angle_deg

    def get_servo_angle(self, joint_id: str) -> float:
        return self._current.get(joint_id, 90.0)

    def tick(self, dt: float) -> Dict[str, float]:
        """
        Advance all servo positions toward their targets.
        Returns dict of joints that have not yet reached their target.
        """
        still_moving: Dict[str, float] = {}
        for jid, target in self._targets.items():
            current = self._current.get(jid, 90.0)
            delta   = target - current
            max_d   = self.speed_dps * dt
            if abs(delta) <= max_d:
                self._current[jid] = target
            else:
                self._current[jid] = current + math.copysign(max_d, delta)
                still_moving[jid] = abs(delta) - max_d
        return still_moving

    def wait_for_completion(self, tolerance_deg: float = 1.0,
                            timeout_s: float = 5.0) -> bool:
        """Simulate waiting until all servos reach targets (for test scripts)."""
        dt = 0.01
        elapsed = 0.0
        while elapsed < timeout_s:
            remaining = self.tick(dt)
            if not remaining:
                return True
            elapsed += dt
        return False

    def enable(self, joint_id: str)  -> None: self._enabled[joint_id] = True
    def disable(self, joint_id: str) -> None: self._enabled[joint_id] = False

    def state_snapshot(self) -> Dict[str, float]:
        return dict(self._current)


# ─────────────────────────────────────────────────────────────────────────────
# PCA9685 single-board driver
# ─────────────────────────────────────────────────────────────────────────────

class PCA9685Driver(MotorDriver):
    """
    Driver for the PCA9685 16-channel PWM servo board over I2C.

    Handles up to 16 channels per board (channels 0–15).
    For 18 servos, use ChainedPCA9685Driver below.

    Wiring (Raspberry Pi 5):
        Pi 3.3V → PCA9685 VCC   (logic supply)
        Pi GND  → PCA9685 GND
        Pi SDA  (GPIO 2, pin 3) → PCA9685 SDA
        Pi SCL  (GPIO 3, pin 5) → PCA9685 SCL
        5V PSU+ → PCA9685 V+   (servo power — separate from Pi 5V!)
        5V PSU− → PCA9685 GND  (common ground with Pi)

    Install:
        pip install adafruit-circuitpython-servokit

    Args:
        i2c_address : I2C address of the board (default 0x40).
        frequency   : PWM frequency in Hz (50 is standard for most servos).
    """

    def __init__(self, i2c_address: int = 0x40, frequency: int = 50):
        try:
            from adafruit_servokit import ServoKit
        except ImportError as e:
            raise ImportError(
                "adafruit-circuitpython-servokit is required.\n"
                "Run: pip install adafruit-circuitpython-servokit"
            ) from e

        self._kit = ServoKit(channels=16, address=i2c_address)
        # Set PWM frequency if supported by the board
        try:
            self._kit.frequency = frequency
        except AttributeError:
            pass

        self._channel_map:  Dict[str, int]   = {}
        self._last_angles:  Dict[str, float] = {}
        self._pulse_ranges: Dict[int, Tuple[int,int]] = {}

    def assign_channel(self, joint_id: str, channel: int,
                       min_pulse_us: int = 500,
                       max_pulse_us: int = 2400) -> None:
        """
        Map a logical joint name to a hardware channel.

        Args:
            joint_id     : e.g. "leg_0.hip"
            channel      : PCA9685 channel 0–15
            min_pulse_us : pulse width for 0° (typically 500 µs)
            max_pulse_us : pulse width for 180° (SG90: 2400, MG996R: 2500)
        """
        if not (0 <= channel <= 15):
            raise ValueError(f"Channel {channel} out of range 0–15 for single PCA9685")
        self._kit.servo[channel].set_pulse_width_range(min_pulse_us, max_pulse_us)
        self._channel_map[joint_id] = channel
        self._pulse_ranges[channel] = (min_pulse_us, max_pulse_us)

    def set_servo_angle(self, joint_id: str, servo_angle_deg: float) -> None:
        channel = self._channel_map.get(joint_id)
        if channel is None:
            return   # joint not assigned — silently skip
        servo_angle_deg = max(0.0, min(180.0, servo_angle_deg))
        self._kit.servo[channel].angle = servo_angle_deg
        self._last_angles[joint_id] = servo_angle_deg

    def get_servo_angle(self, joint_id: str) -> float:
        return self._last_angles.get(joint_id, 90.0)

    def enable(self, joint_id: str) -> None:
        pass  # PCA9685 is always powered; torque is always on

    def disable(self, joint_id: str) -> None:
        """Detach servo (releases torque)."""
        channel = self._channel_map.get(joint_id)
        if channel is not None:
            self._kit.servo[channel].angle = None

    def detach_all(self) -> None:
        """Release torque on all assigned servos (safe shutdown)."""
        for joint_id in list(self._channel_map):
            self.disable(joint_id)


# ─────────────────────────────────────────────────────────────────────────────
# Chained dual-board driver (fixes the 18-servo channel bug)
# ─────────────────────────────────────────────────────────────────────────────

class ChainedPCA9685Driver(MotorDriver):
    """
    Supports up to 32 channels by chaining two PCA9685 boards.

    Board A at 0x40 handles channels  0–15.
    Board B at 0x41 handles channels 16–31 (addressed as 0–15 on the board).

    Wiring for board B:
        Solder the A0 jumper on the second PCA9685 board → its I2C address
        becomes 0x41. Connect SDA/SCL/GND/VCC in parallel with board A.

    Usage:
        driver = ChainedPCA9685Driver()
        driver.assign_channel("leg_5.ankle", 17, min_pulse_us=500, max_pulse_us=2400)
        # Channel 17 is automatically routed to board B channel 1.
    """

    def __init__(self, addr_a: int = 0x40, addr_b: int = 0x41,
                 frequency: int = 50):
        self._board_a = PCA9685Driver(i2c_address=addr_a, frequency=frequency)
        try:
            self._board_b = PCA9685Driver(i2c_address=addr_b, frequency=frequency)
            self._has_b = True
        except Exception:
            self._has_b = False
            self._board_b = None

    def assign_channel(self, joint_id: str, channel: int,
                       min_pulse_us: int = 500,
                       max_pulse_us: int = 2400) -> None:
        if channel < 16:
            self._board_a.assign_channel(joint_id, channel, min_pulse_us, max_pulse_us)
        else:
            if not self._has_b:
                raise RuntimeError(
                    f"Channel {channel} requires a second PCA9685 at 0x41, "
                    f"which was not found on the I2C bus."
                )
            self._board_b.assign_channel(joint_id, channel - 16, min_pulse_us, max_pulse_us)

    def set_servo_angle(self, joint_id: str, servo_angle_deg: float) -> None:
        self._board_a.set_servo_angle(joint_id, servo_angle_deg)
        if self._has_b:
            self._board_b.set_servo_angle(joint_id, servo_angle_deg)

    def get_servo_angle(self, joint_id: str) -> float:
        a = self._board_a.get_servo_angle(joint_id)
        if a != 90.0:
            return a
        return self._board_b.get_servo_angle(joint_id) if self._has_b else 90.0

    def enable(self, joint_id: str)  -> None: pass
    def disable(self, joint_id: str) -> None:
        self._board_a.disable(joint_id)
        if self._has_b:
            self._board_b.disable(joint_id)

    def detach_all(self) -> None:
        self._board_a.detach_all()
        if self._has_b:
            self._board_b.detach_all()


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory interpolator
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryInterpolator:
    """
    Enforces servo speed limits between consecutive gait frames.

    The gait engine produces frames at a fixed rate. Without interpolation,
    a large angle jump gets sent as a single instant command, causing the
    servo to slam to position (stall current spike, violent motion, possible
    brownout on Pi).

    Usage:
        interp = TrajectoryInterpolator(driver)
        interp.move_to(angles_dict, frame_time_s=0.02)
        # Blocks until all servos are within tolerance or time expires.
    """

    def __init__(self, driver: MotorDriver,
                 default_speed_dps: float = 400.0,
                 poll_interval_s: float = 0.005):
        self._driver        = driver
        self._default_speed = default_speed_dps
        self._poll           = poll_interval_s
        self._last_angles:   Dict[str, float] = {}

    def move_to(self,
                servo_angles: Dict[str, float],
                frame_time_s: float = 0.02,
                speed_override_dps: Optional[float] = None) -> float:
        """
        Command all joints to move to new servo angles within frame_time_s.

        Computes the required speed for the largest joint movement and applies
        that as the frame pace, clamped to the servo's rated speed.

        Returns actual elapsed time.
        """
        speed = speed_override_dps or self._default_speed

        # Find the largest delta
        max_delta = 0.0
        for jid, target in servo_angles.items():
            last = self._last_angles.get(jid, 90.0)
            max_delta = max(max_delta, abs(target - last))

        if max_delta < 0.5:
            # Nothing significant to move; send immediately and return
            for jid, angle in servo_angles.items():
                self._driver.set_servo_angle(jid, angle)
                self._last_angles[jid] = angle
            return 0.0

        # Time needed for the largest movement at rated speed
        move_time = max_delta / speed
        # Use the longer of frame_time or move_time
        wait_time = max(frame_time_s, move_time)

        t0 = time.monotonic()
        for jid, angle in servo_angles.items():
            self._driver.set_servo_angle(jid, angle)
            self._last_angles[jid] = angle

        # Busy-wait remainder of the frame to enforce pacing
        elapsed = time.monotonic() - t0
        remaining = wait_time - elapsed
        if remaining > 0.001:
            time.sleep(remaining)

        return time.monotonic() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Motor Abstraction Layer
# ─────────────────────────────────────────────────────────────────────────────

class MotorAbstractionLayer:
    """
    Unified joint control API used by Robot.

    Receives FRAMEWORK angles (degrees, ±90 convention) from the gait engine,
    applies per-joint calibration (zero_offset + direction flip) to produce
    SERVO angles (0–180°), then forwards to the driver.

    This is the layer that translates the IK output into real hardware commands.
    """

    def __init__(self, driver: Optional[MotorDriver] = None,
                 frame_time_s: float = 0.02):
        self.driver     = driver or SimulatedDriver()
        self._joint_map: Dict[str, str]          = {}
        self._calibs:    Dict[str, object]        = {}  # joint_id → JointCalibration
        self._interp     = TrajectoryInterpolator(self.driver)
        self.frame_time  = frame_time_s

    def register_joint(self, logical_name: str,
                       hardware_id: Optional[str] = None,
                       calibration=None) -> None:
        """Register a joint with optional calibration override."""
        self._joint_map[logical_name] = hardware_id or logical_name
        if calibration is not None:
            self._calibs[logical_name] = calibration

    def move_joint(self, joint_name: str, framework_angle_deg: float,
                   calibration=None) -> None:
        """
        Move one joint.

        framework_angle_deg : angle from the IK solver (±90° convention)
        calibration         : JointCalibration instance, or None to use stored.
        """
        hw_id = self._joint_map.get(joint_name, joint_name)
        cal   = calibration or self._calibs.get(joint_name)

        if cal is not None:
            servo_angle = cal.to_servo_angle(framework_angle_deg)
        else:
            # No calibration: simple 0–180 mapping, neutral at 90°
            servo_angle = max(0.0, min(180.0, framework_angle_deg + 90.0))

        self.driver.set_servo_angle(hw_id, servo_angle)

    def move_frame(self, joint_angles: Dict[str, float],
                   calibrations: Optional[Dict[str, object]] = None) -> float:
        """
        Move all joints in one gait frame simultaneously with speed-limited pacing.

        joint_angles  : {joint_name: framework_angle_deg}
        calibrations  : optional dict of JointCalibration per joint name

        Returns actual elapsed time.
        """
        servo_angles: Dict[str, float] = {}
        cals = calibrations or {}

        for jname, fw_angle in joint_angles.items():
            hw_id = self._joint_map.get(jname, jname)
            cal   = cals.get(jname) or self._calibs.get(jname)
            if cal is not None:
                servo_angle = cal.to_servo_angle(fw_angle)
            else:
                servo_angle = max(0.0, min(180.0, fw_angle + 90.0))
            servo_angles[hw_id] = servo_angle

        return self._interp.move_to(servo_angles, frame_time_s=self.frame_time)

    def move_all(self, angles: Dict[str, float], speed: float = 1.0) -> None:
        """Legacy convenience: move all joints without frame pacing."""
        for name, angle in angles.items():
            self.move_joint(name, angle)

    def read_joint(self, joint_name: str) -> float:
        hw_id = self._joint_map.get(joint_name, joint_name)
        return self.driver.get_servo_angle(hw_id)

    def detach_all(self) -> None:
        """Safely release all servo torque."""
        for hw_id in self._joint_map.values():
            try:
                self.driver.disable(hw_id)
            except Exception:
                pass
