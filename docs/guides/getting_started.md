# Getting started

## What is AIDO meant to do?

AIDO is a machine learning tool that transcribes a design task, e.g. where to place materials, in what quantity and with what dimensions, into an optimization problem. This means that in the first step, the goodness of a detector must be quantified into an appropriate loss function. This loss function is then improved iteratively by the AIDO Optimizer. The AIDO framework provides you with the tools to optimize your detector, based on your requirements. However, there are some pieces that you must have yourself. Here is a graph of the workflow with the elements that you need to implement:

| Element           | **AIDO**  | **User**  | Description                                                   |
|-------------------|-----------|-----------|---------------------------------------------------------------|
| Orchestration     | x         |           | Task scheduling and function execution                        |
| Simulation        |           | x         | Running the Geant4 simulation software                        |
| Merging           |           | x         | Combining the simulated data into a format of your choosing   |
| Reconstruction    |           | x         | Computing the loss for each event                             |
| Smoothing         | x         | x         | Local interpolation of the loss                               |
| Optimization      | x         | x         | ML-based gradient descent                                     |
| Plotting          | x         | (optional)| Plots for model evaluation                                    |

The communication between your programs and AIDO is handled by three classes:

 - SimulationParameter
 - SimulationParameterDictionary
 - UserInterface

## What do you need?

These items must be provided to the AIDO framework in order for it to run:

 1. Simulation software

    A geant4 simulation software, that takes as inputs the values

