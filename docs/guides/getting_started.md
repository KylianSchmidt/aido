# Getting started

## What is AIDO meant to do?

AIDO is a machine learning tool that transcribes a design task, e.g. where to place materials, in what quantity and with what dimensions, into an optimization problem. This means that in the first step, the goodness of a detector must be quantified into an appropriate loss function. This loss function is then improved iteratively by the AIDO Optimizer. The AIDO framework provides you with the tools to optimize your detector, based on your requirements. However, there are some pieces that you must have yourself. Here is a graph of the workflow with the elements that you need to implement:

| Element           | **AIDO**  | **User**   | Description                                                   |
|-------------------|-----------|------------|---------------------------------------------------------------|
| Orchestration     | x         |            | Task scheduling and function execution                        |
| Simulation        |           | x          | Running the Geant4 simulation software                        |
| Merging           |           | x          | Combining the simulated data into a format of your choosing   |
| Reconstruction    |           | x          | Computing the loss for each event                             |
| Smoothing         | x         |            | Local interpolation of the loss                               |
| Optimization      | x         | (penalties)| ML-based gradient descent                                     |
| Plotting          | x         | (optional) | Plots for model evaluation                                    |

The communication between your programs and AIDO is handled by three classes:

 - [SimulationParameter](/api/aido.simulation_helpers)
 - [SimulationParameterDictionary](/api/aido.simulation_helpers)
 - [UserInterface](/api/aido.interface)

## What do you need?

These items must be provided to the AIDO framework in order for it to run:

 1. **Simulation Software**

    A geant4 simulation software with configurable parameters. For example, if you want to optimize the dimensions of some block in the detector, that parameter has to be dynamic in some way. Whether you implement this with Geant4 to python bindings, reading a JSON file or with Geant4 macro functions is up to you.
   
 2. **Reconstruction Algorithm**

    A piece of code that takes the files of your simulation as input and returns the goodness of the detector according to your metrics. The input and output formats are both not relevant for AIDO. 
    
    ```{admonition} Metrics
    :class: hint
    The goodness of a single event should be implemented as a convex loss function, so a smaller value is better. This loss function will be interpolated by AIDO and used to optimize your detector setup.
    ```

 3. **Interface**
   This is an implementation of the `aido.UserInterface` class. More information is found in [usage](usage)