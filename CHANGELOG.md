# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — unreleased

### Added
- `Robot.load()` — universal entry point accepting YAML file path, YAML string, or dict
- Morphology Description Language (MDL) parser — converts YAML robot definitions into
  structured `RobotModel` objects with limbs, joints, topology, and mass
- Kinematics engine — analytical 2-link IK and FK per limb
- Gait engine — `TripodGait`, `WaveGait`, `RippleGait`, `TrotGait`
- Automatic gait selection via `select_gait(family, behavior)`
- Balance engine — convex-hull support polygon, stability margin, tip-risk scoring
- Motor abstraction layer — `SimulatedDriver` + swappable hardware driver interface
- Bundled robot definitions: hexapod, quadruped, spider (octopod), centipede
- `rlmf.robots` sub-package for accessing bundled YAML definitions by Wafula Ian Elmer
- `rlmf` CLI with `describe`, `walk`, `balance`, and `robots` commands
- `py.typed` marker for PEP 561 type-checking support
- Full pytest test suite (55+ tests)
