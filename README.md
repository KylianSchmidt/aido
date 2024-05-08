# aido - AI Detector Optimization

The goal of this framework is to optimize the geometry of detectors simulated with geant4.

## Requires

 - b2luigi
 - numpy
 - docker
 - pytorch (motivates the use of a container?)

## Structure

The pipeline for the optimization algorithm will be handled by b2luigi. 

 - Detector simulation: start several containers of the geant4 simulation executable provided by the user. The start parameters should be written to a json file or provided to b2luigi in some dictionary format. Subsequently, b2luigi passes the relevant parameters of the geometry to the simulation (as geant4 macro file, function parameters or similar). The simulation finally outputs a root file saved to a directory specified by b2luigi (using LocalTarget).

 - Reconstruction: an analysis program provided by the user performs a physics analysis that indicates the performance of the geometry. The input is the root file of each simulation (using "requires") and the output is a file containing the physics metrics. 

  - Optimization: a surrogate ML model is trained on the detector parameters of each geometry and learns the expected physics performance in a region of parameter space. After finding a local minimum, the model will propose a set of new starting parameters for the detector geometry. These parameters are passed back to the Wrapper handling the detector simulation which will in turn start new jobs with the new parameters. Through iteration, a set of optimized parameters are found which are the final output of the program.

  ## Milestones

  - Detector simulation:
    - Start an empty container from python (using singularity, to avoid problems with root rights)
      - Have root rights on the machine you want the container to be
    - Spawn a container with b2luigi
    - Provide a set of parameters to the container (json or python parameters? depends on the geant4 simulation later on)
    - Set the output directory of the simulation
    - Spawn multiple containers from b2luigi, each with different parameters

 - Reconstruction
 - Optimization
