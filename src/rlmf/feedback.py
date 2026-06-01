"""
feedback.py — Closed-loop sensor feedback for RLMF v0.3

Provides two orthogonal feedback channels:

  IMUSensor          Body orientation (roll/pitch/yaw) via MPU-6050 over I2C.
  ContactSensor      Per-foot ground contact via FSR (force-sensitive resistors)
                     or servo current sensing via INA219 current monitors.
  FeedbackState      Snapshot produced every control cycle, consumed by the
                     BalanceController and gait corrector.
  SimulatedFeedback  Deterministic in-memory implementation for testing.

Design principle
----------------
Every sensor class exposes the same interface:

    sensor.read() -> FeedbackState

The Robot's control loop calls this once per gait frame and feeds the result
to the balance controller.  No polling, no threads — caller decides the rate.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUReading:
    """Body orientation and angular rates from an IMU."""
    roll_deg:   float = 0.0   # + = right side down
    pitch_deg:  float = 0.0   # + = nose up
    yaw_deg:    float = 0.0   # heading, 0 = forward
    roll_rate:  float = 0.0   # deg/s
    pitch_rate: float = 0.0
    yaw_rate:   float = 0.0
    timestamp:  float = field(default_factory=time.monotonic)

    @property
    def is_level(self) -> bool:
        """True when body tilt is within ±5° in both axes."""
        return abs(self.roll_deg) < 5.0 and abs(self.pitch_deg) < 5.0

    @property
    def tilt_magnitude(self) -> float:
        """Combined tilt angle in degrees."""
        return math.sqrt(self.roll_deg**2 + self.pitch_deg**2)


@dataclass
class ContactReading:
    """Per-limb ground contact state."""
    contacts: Dict[str, bool]   = field(default_factory=dict)  # limb_name → in_contact
    forces:   Dict[str, float]  = field(default_factory=dict)  # limb_name → force (N)
    timestamp: float            = field(default_factory=time.monotonic)

    def stance_limbs(self) -> List[str]:
        return [name for name, c in self.contacts.items() if c]

    def swing_limbs(self) -> List[str]:
        return [name for name, c in self.contacts.items() if not c]

    def contact_count(self) -> int:
        return sum(1 for c in self.contacts.values() if c)


@dataclass
class FeedbackState:
    """Complete sensor snapshot for one control cycle."""
    imu:      IMUReading
    contact:  ContactReading
    cycle_dt: float = 0.025     # actual elapsed time since last read (s)

    @property
    def is_stable(self) -> bool:
        return self.imu.is_level and self.contact.contact_count() >= 3


# ─────────────────────────────────────────────────────────────────────────────
# IMU sensor
# ─────────────────────────────────────────────────────────────────────────────

class IMUSensor(ABC):
    @abstractmethod
    def read(self) -> IMUReading: ...
    @abstractmethod
    def calibrate(self) -> None: ...


class MPU6050Sensor(IMUSensor):
    """
    Real MPU-6050 / MPU-6500 IMU over I2C.

    Wiring (Raspberry Pi 5):
        Pi 3.3V → MPU-6050 VCC
        Pi GND  → MPU-6050 GND
        Pi SDA (GPIO 2) → MPU-6050 SDA
        Pi SCL (GPIO 3) → MPU-6050 SCL
        MPU-6050 AD0 → GND  (I2C address 0x68)

    Install:
        pip install mpu6050-raspberrypi

    The sensor is mounted flat on the robot body, arrow pointing forward.
    Calibrate once per session before moving (takes ~2 s).
    """

    def __init__(self, address: int = 0x68,
                 alpha: float = 0.96):
        """
        Args:
            address : I2C address (0x68 default, 0x69 if AD0 is pulled high).
            alpha   : complementary filter coefficient (0.95–0.98 typical).
                      Higher = trust gyro more, lower = trust accel more.
        """
        try:
            from mpu6050 import mpu6050
            self._mpu = mpu6050(address)
        except ImportError as e:
            raise ImportError(
                "mpu6050-raspberrypi is required.\n"
                "Run: pip install mpu6050-raspberrypi"
            ) from e

        self._alpha       = alpha
        self._roll        = 0.0
        self._pitch       = 0.0
        self._yaw         = 0.0
        self._last_time   = time.monotonic()
        self._offsets     = {"ax": 0.0, "ay": 0.0, "az": 0.0,
                             "gx": 0.0, "gy": 0.0, "gz": 0.0}
        self._calibrated  = False

    def calibrate(self, samples: int = 200) -> None:
        """
        Measure static offsets with the robot standing still on a level surface.
        Call once before starting any gait.
        """
        print(f"[IMU] Calibrating ({samples} samples)… keep robot still.")
        acc_sum  = {"x": 0.0, "y": 0.0, "z": 0.0}
        gyro_sum = {"x": 0.0, "y": 0.0, "z": 0.0}
        for _ in range(samples):
            a = self._mpu.get_accel_data()
            g = self._mpu.get_gyro_data()
            for k in ("x", "y", "z"):
                acc_sum[k]  += a[k]
                gyro_sum[k] += g[k]
            time.sleep(0.01)
        n = samples
        self._offsets = {
            "ax": acc_sum["x"] / n,
            "ay": acc_sum["y"] / n,
            "az": acc_sum["z"] / n - 9.81,   # remove gravity from Z
            "gx": gyro_sum["x"] / n,
            "gy": gyro_sum["y"] / n,
            "gz": gyro_sum["z"] / n,
        }
        self._calibrated = True
        print("[IMU] Calibration complete.")

    def read(self) -> IMUReading:
        now = time.monotonic()
        dt  = now - self._last_time
        self._last_time = now

        a = self._mpu.get_accel_data()
        g = self._mpu.get_gyro_data()

        # Remove calibration offsets
        ax = a["x"] - self._offsets["ax"]
        ay = a["y"] - self._offsets["ay"]
        az = a["z"] - self._offsets["az"]
        gx = g["x"] - self._offsets["gx"]
        gy = g["y"] - self._offsets["gy"]
        gz = g["z"] - self._offsets["gz"]

        # Accelerometer angles
        accel_roll  = math.degrees(math.atan2(ay, az))
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))

        # Complementary filter
        self._roll  = self._alpha * (self._roll  + gx * dt) + (1 - self._alpha) * accel_roll
        self._pitch = self._alpha * (self._pitch + gy * dt) + (1 - self._alpha) * accel_pitch
        self._yaw  += gz * dt

        return IMUReading(
            roll_deg   = self._roll,
            pitch_deg  = self._pitch,
            yaw_deg    = self._yaw,
            roll_rate  = gx,
            pitch_rate = gy,
            yaw_rate   = gz,
            timestamp  = now,
        )


class SimulatedIMU(IMUSensor):
    """
    Deterministic IMU for testing. Returns level readings unless overridden.
    Inject tilt to test the balance controller recovery logic.
    """

    def __init__(self):
        self.roll_deg  = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg   = 0.0

    def calibrate(self) -> None:
        pass   # no-op in simulation

    def inject_tilt(self, roll: float, pitch: float) -> None:
        """Simulate a tilt event (e.g. terrain slope)."""
        self.roll_deg  = roll
        self.pitch_deg = pitch

    def read(self) -> IMUReading:
        return IMUReading(
            roll_deg=self.roll_deg, pitch_deg=self.pitch_deg,
            yaw_deg=self.yaw_deg,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Contact sensor
# ─────────────────────────────────────────────────────────────────────────────

class ContactSensor(ABC):
    @abstractmethod
    def read(self) -> ContactReading: ...


class FSRContactSensor(ContactSensor):
    """
    Force-sensitive resistor contact detection via MCP3008 ADC over SPI.

    One FSR per foot.  The FSR sits between the foot tip and the foot pad.
    At rest (no contact): high resistance → low ADC voltage → value near 0.
    Under load (contact): low resistance → high voltage → value near 1023.

    Wiring:
        Pi SPI (GPIO 8 CE0, GPIO 11 CLK, GPIO 9 MISO, GPIO 10 MOSI)
        → MCP3008 CS/CLK/DOUT/DIN
        MCP3008 VDD/VREF → 3.3V
        MCP3008 AGND/DGND → GND
        Each FSR between CH0…CH7 and GND (with 10kΩ pull-down to GND).

    Install:
        pip install spidev

    Args:
        limb_channels : {limb_name: MCP3008_channel}  e.g. {"leg_0": 0}
        threshold     : ADC value (0–1023) above which contact is detected.
                        Tune by pressing foot and observing raw values.
    """

    def __init__(self, limb_channels: Dict[str, int],
                 threshold: int = 200,
                 spi_bus: int = 0, spi_device: int = 0):
        try:
            import spidev
            self._spi = spidev.SpiDev()
            self._spi.open(spi_bus, spi_device)
            self._spi.max_speed_hz = 1_350_000
        except ImportError as e:
            raise ImportError("spidev is required.\nRun: pip install spidev") from e
        except Exception as e:
            raise RuntimeError(f"SPI init failed: {e}") from e

        self._channels = limb_channels
        self._threshold = threshold

    def _read_channel(self, ch: int) -> int:
        assert 0 <= ch <= 7
        r = self._spi.xfer2([1, (8 + ch) << 4, 0])
        return ((r[1] & 3) << 8) + r[2]

    def read(self) -> ContactReading:
        contacts: Dict[str, bool]  = {}
        forces:   Dict[str, float] = {}
        for limb, ch in self._channels.items():
            raw = self._read_channel(ch)
            contacts[limb] = raw >= self._threshold
            forces[limb]   = float(raw) / 1023.0   # normalised 0..1
        return ContactReading(contacts=contacts, forces=forces)


class CurrentContactSensor(ContactSensor):
    """
    Contact detection via servo current sensing using INA219 power monitors.

    When a foot touches the ground, the servo holding it draws more current.
    A sharp current spike above `threshold_ma` indicates contact.

    One INA219 per leg (or share with I2C address assignment).
    INA219 I2C addresses: 0x40–0x4F (set via A0/A1 pins).

    Wiring:
        INA219 VCC → 3.3V
        INA219 GND → GND
        INA219 SDA/SCL → Pi SDA/SCL (shared I2C bus)
        INA219 VIN+ → servo power rail (+5V)
        INA219 VIN- → servo power input

    Install:
        pip install pi-ina219

    Args:
        limb_addresses : {limb_name: i2c_address}  e.g. {"leg_0": 0x40}
        threshold_ma   : current spike above this = foot contact detected.
                         Start at 200 mA and tune for your servo+load.
    """

    def __init__(self, limb_addresses: Dict[str, int],
                 threshold_ma: float = 200.0):
        try:
            from ina219 import INA219
        except ImportError as e:
            raise ImportError("pi-ina219 is required.\nRun: pip install pi-ina219") from e

        from ina219 import INA219
        self._monitors: Dict[str, INA219] = {}
        for limb, addr in limb_addresses.items():
            ina = INA219(0.1, address=addr)   # 0.1Ω shunt resistor
            ina.configure()
            self._monitors[limb] = ina
        self._threshold = threshold_ma

    def read(self) -> ContactReading:
        contacts: Dict[str, bool]  = {}
        forces:   Dict[str, float] = {}
        for limb, ina in self._monitors.items():
            try:
                ma = ina.current()
                contacts[limb] = ma >= self._threshold
                forces[limb]   = ma
            except Exception:
                contacts[limb] = False
                forces[limb]   = 0.0
        return ContactReading(contacts=contacts, forces=forces)


class SimulatedContactSensor(ContactSensor):
    """
    Deterministic contact sensor for testing.
    By default respects the gait stance_mask passed in.
    Can inject false contacts or missed contacts to test error recovery.
    """

    def __init__(self, limb_names: List[str]):
        self._limbs   = limb_names
        self._override: Optional[Dict[str, bool]] = None

    def set_stance(self, stance_mask: Dict[str, bool]) -> None:
        """Called by the gait engine each frame to inject realistic contact pattern."""
        self._override = dict(stance_mask)

    def inject_missed_contact(self, limb: str) -> None:
        """Simulate a foot that failed to touch down."""
        if self._override:
            self._override[limb] = False

    def read(self) -> ContactReading:
        if self._override is not None:
            return ContactReading(
                contacts=dict(self._override),
                forces={l: (1.0 if self._override.get(l, False) else 0.0)
                        for l in self._limbs},
            )
        # Default: all in contact
        return ContactReading(
            contacts={l: True for l in self._limbs},
            forces  ={l: 1.0  for l in self._limbs},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unified feedback reader
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackReader:
    """
    Reads IMU and contact sensors together and produces FeedbackState.
    Used by Robot._control_cycle().
    """

    def __init__(self, imu: IMUSensor, contact: ContactSensor):
        self._imu     = imu
        self._contact = contact
        self._last_t  = time.monotonic()

    def read(self) -> FeedbackState:
        now = time.monotonic()
        dt  = now - self._last_t
        self._last_t = now
        return FeedbackState(
            imu      = self._imu.read(),
            contact  = self._contact.read(),
            cycle_dt = dt,
        )

    def calibrate(self) -> None:
        self._imu.calibrate()
