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