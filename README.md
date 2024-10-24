# aido - AI Detector Optimization

The goal of this framework is to optimize the geometry of detectors simulated with geant4.

## Installation

Installation with 'setup.py' requires pip>=24.0 and python>=3.12:
```
python3 -m pip install aido
```

## Requirements

For the scheduler:

 - b2luigi
 - numpy
 - pandas
 - awkward
 - pyarrow
 - fastparquet
 - torch

## Usage

TBI

## Structure

The pipeline for the optimization algorithm will be handled by b2luigi. 

 - Wrapper:
    - Accept geometry parameters with starting value, type, min, max, cost scaling coefficient
    - Write parameters to file for each iteration of the training loop
    - Call the class that generates new parameters in a given region (for example normally distributed)
    - Call the detector simulation and reconstruction for each set of parameters

 - Detector simulation: 
    - Start several containers of the geant4 simulation using the executable provided by the user.
    - The start parameters are written to a .json file.
    - This .json file must be used by the user to initialize the relevant parameters of the geometry in the simulation. The implementation of this step is left to the user, for example a script that converts this parameter dict to a geant4 macro file, function parameters or similar.
    - The simulation finally outputs a file which must be saved to the directory specified by b2luigi (using LocalTarget).

 - Reconstruction:
    - An analysis program provided by the user performs a physics analysis that indicates the performance of the geometry. The input are the output files of each simulation (using "requires") and the output is a file containing a Metric that quantifies the performance of each detector (commonly refered to as a Loss). 

  - Optimization: 
    - Surrogate model trained on the detector parameters of each geometry
    - Learns the expected physics performance in a given region of parameter space.
    - After finding a local minimum, the Optimizer model will propose a new region to explore in the detector geometry parameter space.
    - These parameters are passed back to the Wrapper handling the detector simulation, which will in turn start new jobs with the new parameters.
    - Through iteration, a set of optimized parameters are found, which are the final output of the program.
