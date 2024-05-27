# aido - AI Detector Optimization

The goal of this framework is to optimize the geometry of detectors simulated with geant4.

## Requires

 - b2luigi
 - numpy
 - singularity
 - pytorch (motivates the use of another container?)

## Structure

The pipeline for the optimization algorithm will be handled by b2luigi. 

 - Wrapper:
    - Accept geometry parameters with starting value, type, min, max, cost scaling coefficient
    - Write parameters to file for each iteration of the training loop
    - Call the class that generates new parameters in a given region (for example normally distributed)
    - Call the detector simulation and reconstruction for set of parameters

 - Detector simulation: 
    - Start several containers of the geant4 simulation executable provided by the user.
    - The start parameters should be written to a json file or provided to b2luigi in some dictionary format.
    - Subsequently, b2luigi passes the relevant parameters of the geometry to the simulation (as geant4 macro file, function parameters or similar).
    - The simulation finally outputs a root file saved to a directory specified by b2luigi (using LocalTarget).

 - Reconstruction: an analysis program provided by the user performs a physics analysis that indicates the performance of the geometry. The input is the root file of each simulation (using "requires") and the output is a file containing the physics metrics. 

  - Optimization: 
    - Surrogate ML model trained on the detector parameters of each geometry
    - Learns the expected physics performance in a given region of parameter space.
    - After finding a local minimum, the model will propose a new region to explore in the detector geometry parameter space.
    - These parameters are passed back to the Wrapper handling the detector simulation, which will in turn start new jobs with the new parameters.
    - Through iteration, a set of optimized parameters are found, which are the final output of the program.

  ## Milestones

  - Detector simulation:
    - [x] Start an empty container from python (using singularity, to avoid problems with root rights)
    - [x] Spawn a container with b2luigi
    - [ ] Provide a set of parameters to the container (json or python parameters? depends on the geant4 simulation later on).
        - Note: if all parameters of two tasks are the same, b2luigi uses the same instance (useful to avoid repetitions). 
        - Use b2luigi parameters for each simulation parameter? Advantage is that it enables unique Tasks and is easy to track throughtout the pipeline. Drawback is that it has to written to json file for the simulation and be specified by the user (not so tragic).
    - [ ] Set the output directory of the simulation
    - [ ] Spawn multiple containers from b2luigi, each with different parameters
    - [ ] Open the API to the user for spawning the containers (e.g. CLI commands)

 - Reconstruction
    - [ ] Start a Task with GPU support
    - [ ] Start a Reconstruction Task for each simulated geometry (link using requires()?)
    - [ ] Read the corresponding output parameters of the simulation
    - [ ] Write to file the output of the reconstruction (same location as simiulation)

 - Optimization
    - [ ] Read the outputs of the reconstruction and build an array for the training
    - [ ] Start a GPU training Task that produces the surrogate model
    - [ ] Use Gradient descent to find local minimum
    - [ ] Write optimal parameters to file for this iteration
    - [ ] Call the class responsable for generating new parameter sets
