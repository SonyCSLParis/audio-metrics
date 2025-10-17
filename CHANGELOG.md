# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.3] - 2025-10-17

### Added
- Version bumb for workflow triggers

## [1.0.2] - 2025-10-17

### Added
- GitHub Actions workflow for automated releases and PyPI publishing
- `__version__` attribute available in package

### Changed
- Replace problematic `cylimiter` dependency by `numpy-audio-limiter`; 
- Fixed `torchvision` installation issue for Python 3.13

### Fixed
- Dependency issues and build system configuration
- Missing rich dependency

## [1.0.1] - 2025-05-16

### Added
- Missing VGGish module

### Changed
- Project definition (pyproject.toml)

## [1.0.0] - 2024

### Changed
- Full rewrite of audio metrics package
