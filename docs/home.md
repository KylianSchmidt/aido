# Introduction

The AI Detector Optimization framework (AIDO) is a tool for learning the optimal
design of particle physics detectors. By interpolating the results of simulations
with slightly different geometries, it can iteratively find the best set of detector
parameters.

This framework fragments the workflow into [b2luigi](https://b2luigi.belle2.org/index.html)
Tasks for parallel simulations and the training of ML models on GPUs.

In order to use this framework, you need:

1. A simulation software. Any tool that can produce relevant information with which to
   gauge the performance of your detector. Explicitly, AIDO was developed with Geant4
   simulations in mind, but there is no hard constraint on this. The details about
   the requirements for you simulation software are explained further
2. A reconstruction algorithm. This can be any piece of code that computes a loss function
   based on expected versus true Monte Carlo information. In essence, AIDO works by
   optimizing the loss you provide with this algorithm.

A parameter is defined as any value that can be adjusted in your simulation software. It
is the goal of AIDO to perform a hyperparameter optimization on this parameter to improve
the loss calculated by the reconstruction algorithm. The [SimulationParameter](source/aido#aido.simulation_helpers.SimulationParameter)
object is the basic building block for a parameter. It keeps track of the current value
during the optimization process as well as other useful information.

A set of parameters are combined into a single [SimulationParameterDictionary](source/aido#aido.simulation_helpers.SimulationParameterDictionary)
which has extra tools. Most relevant is the way we interface the AIDO framework with your
simulation and reconstruction. For this, the dictionary is stored as a json file which
you can easily access in any programming language (for example C++ when using Geant4).
By inputting these values in your simulation, AIDO is able to optimize the parameters
automatically.

For more details, see the API Reference section.

# Documentation

- Getting Started getting_started
- Usage
  - [SimulationParameter](source/aido#aido.simulation_helpers.SimulationParameter): single parameter with many options
  - [SimulationParameterDictionary](source/aido#aido.simulation_helpers.SimulationParameterDictionary): collection of parameters for your simulations
  - [UserInterfaceBase](source/aido#aido.interface.UserInterfaceBase): connection of your code to the aido framework
- Examples

# Indices and Tables

* General Index
* [Module Index]()
* Search Page
