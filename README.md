# aido - AI Detector Optimization

The goal of this framework is to optimize the geometry of detectors simulated with geant4.

## Installation

Requires pip>=24.0 and python>=3.12. 

1. Installation with 'setup.py':
```
python3 -m pip install .
```
2. Or with 'requirements.txt'
```
python3 -m pip install -r requirements.txt
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

  ## Milestones

  - Detector simulation:
    - [x] Start an empty container from python (using singularity, to avoid problems with root rights)
    - [x] Spawn a container with b2luigi
    - [x] Provide a set of parameters to the container as json. 
        - The list of all parameters is handled by the SimulationParameterDictionary class, which holds SimulationParameter instances (floats, string for materials, integers should all be inherited class of the base class)
        - The writting of these parameters to a file that the geant4 simulation can read is left to the end user (write to macro file, parse in CLI, etc.)
    - [x] Discrete parameters are learned using probabilities associated with each possible value that represent the confidence of the Optimizer.
    - [x] Set the output directory of the simulation handled by b2luigi
    - [x] Spawn multiple containers from b2luigi, each with different parameters
    - [x] Open the API to the user for spawning the containers (e.g. CLI commands)

 - Reconstruction
    - [x] Start a Task with GPU support
    - [x] Start a Reconstruction Task for each simulated geometry (link using requires()?)
    - [x] Read the corresponding output parameters of the simulation
    - [x] Write to file the output of the reconstruction (same location as simulation)
    - [x] Merge the output of the simulation into pd.DataFrame 
    - [x] API using pd.DataFrame (user has to provide individual keys)
    - [x] Normalize once at the first iteration and continue using those normalizations later (for better convergence of the Surrogate model)

 - Optimization
    - [x] Read the outputs of the reconstruction and build an array for the training
    - [x] Start a GPU training Task that produces the surrogate model
    - [x] Use Gradient descent to find local minimum
    - [x] Write optimal parameters to file for this iteration
    - [x] Call the class responsable for generating new parameter sets. Now in SimulationParameterDictionary

 - Others
    - [x] Pip package or venv list of all packages used in the main b2luigi scheduler file
    - [x] Read the outputs of the reconstruction and build an array for the training
    - [x] Start a GPU training Task that produces the surrogate model
    - [x] Use Gradient descent to find local minimum
    - [x] Write optimal parameters to file for this iteration
    - [x] Call the class responsable for generating new parameter sets

