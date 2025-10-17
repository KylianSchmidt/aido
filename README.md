# aido - AI Detector Optimization

The goal of this framework is to optimize the geometry of detectors simulated with geant4.

## Installation

Installation with 'setup.py' requires pip>=24.0 and python>=3.12:
1. Git clone this repository (or download the code)
2. `cd` to the directory
3. Install the `aido` package with

```
pip install .
```

The use of a virtual environment is highly recommended.

## Dependencies

For the scheduler:

 - b2luigi
 - numpy
 - pandas
 - awkward (from the examples)
 - pyarrow
 - fastparquet

For the ML models:
 - torch

## Documentation

https://aido.readthedocs.io/en/latest/index.html