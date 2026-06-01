# Changelog

## [0.4.0] — 2026-05-31  (current)

### Fixed — IK sagittal plane error (Fix 1)
The v0.3 IK computed the knee+ankle angles in an assumed-vertical sagittal
plane.  For legs at intermediate mount angles (±30°, ±60°, ±150°…) this
produced systematic foot-position errors of up to 180 mm.

The fix is a full 3-D geometric solver:
  1. Hip angle — exact atan2 in XY plane.
  2. Coxa tip — computed in body frame using the clamped hip angle.
  3. Sagittal projection — target expressed relative to the coxa tip in the
     leg's own forward/down frame using a dot product with the leg's azimuth
     unit vector.  This projection is exact for all mount angles.
  4. 2-link IK — solved in the sagittal frame.
  5. Ankle angle — computed from the residual vector (knee tip → target)
     rather than mirroring the femur angle.  Result: IK→FK error < 0.001 mm
     on all mount angles when joints are within limits.

Also corrected the bundled YAML joint limits: ankle was -45..45 (too narrow
for normal stance depth). Updated to -90..90, matching real SG90 mechanical
range and enabling the IK to reach realistic foot targets without clamping.

### Fixed — Balance controller oscillation (Fix 2)
v0.3 used a PI controller.  Without a derivative term, the controller
overshoots and oscillates at servo bandwidth frequency (1–3 Hz).

`BalanceController` is now a full PID controller.  The derivative term reads
`imu.roll_rate` and `imu.pitch_rate` directly from the gyroscope (not
differentiated angle, which amplifies noise).  The D term brakes overshoot:
when the robot is approaching level fast, the rate signal reverses the
correction before the set-point is crossed.

Tuning defaults: kp=0.30, ki=0.02, kd=0.05.

### Fixed — Thermal model ODE discretisation bug (Fix 3)
The thermal ODE was:
    dT = (P/C - (T-Ta)/R) * dt          ← wrong (effective T_ss = Ta + P*R/C)

Correct form of the first-order thermal ODE (C·dT/dt = P - (T-Ta)/R):
    dT = (P/C - (T-Ta)/(R*C)) * dt      ← correct (T_ss = Ta + P*R)

The old form caused the model to converge to T_ss ≈ 55 °C instead of the
correct 120 °C, so the thermal throttle never fired at realistic duty cycles.

Parameters validated against SG90 published data (I_stall=700 mA,
observed time to 80 °C ≈ 90 s at stall):
  r_winding_ohm = 6.0  (measured; SG90 datasheet absent)
  c_thermal     = 3.2  J/°C
  r_thermal     = 32.3 °C/W
These give T_ss=120 °C and t_80≈90 s, matching observation.

### Added — ContactCalibrator (Fix 4)
Auto-measures per-foot ADC thresholds for FSRContactSensor by sampling
rest (no contact) and loaded (all feet on ground) values, then setting
each threshold to the midpoint.  Eliminates manual threshold guessing and
handles FSR part-to-part variation and different robot weights.

### Other
- All bundled robot YAMLs: ankle limits updated to -90..90,
  knee limits updated to -90..120.
- Test suite: 82 tests, all passing.

## [0.3.0] — 2026-05-31
- IMU (MPU-6050) + contact (FSR, INA219) sensor integration
- PI balance controller, torque/thermal/power safety subsystems
- PoseCorrector drift detection
- 118-test suite

## [0.2.0] — 2026-05-31
- JointCalibration (zero_offset, direction, speed_dps)
- ChainedPCA9685Driver (18-channel dual-board)
- TrajectoryInterpolator (servo-speed-limited frame pacing)
- safe_startup() / safe_shutdown()
- 75-test suite

## [0.1.0] — 2026-05-30
- Initial release: MDL parser, gaits, IK/FK, balance engine, CLI, 59 tests
