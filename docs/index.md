# AIDO Documentation

## Introduction

The **AI** **D**etector **O**ptimization framework (AIDO) is designed to approximate the optimal
design of particle physics detectors. By interpolating the results of simulations
with slightly different geometries, it can estimate the best set of detector
parameters with the help of gradient descent.

This framework fragments the workflow into [b2luigi](https://b2luigi.belle2.org/index.html)
Tasks for parallel simulations and training of ML models on GPUs.

In order to use this framework, you need:

1. A simulation software. Any tool that can produce relevant information with which to
   gauge the performance of your detector. Explicitly, AIDO was developed with Geant4
   simulations in mind, but there is no hard constraint on this. The details about
   the requirements for you simulation software are explained further

2. A reconstruction algorithm. This can be any piece of code that computes a loss function
   based on expected versus true Monte Carlo information. In essence, AIDO works by
   optimizing the loss you provide with this algorithm. It is important that the loss you
   provide meaningfully represents the goal you want to achieve.

A parameter is defined as any value that can be adjusted in your simulation software. It
is the goal of AIDO to perform a hyperparameter optimization on this parameter to improve
the loss calculated by the reconstruction algorithm. The [SimulationParameter](api/aido.simulation_helpers)
object is the basic building block for a parameter. It keeps track of the current value
during the optimization process as well as other useful information.

A set of parameters are combined into a single [SimulationParameterDictionary](api/aido.simulation_helpers)
which has extra tools. Most relevant is the way we interface the AIDO framework with your
simulation and reconstruction. For this, the dictionary is stored as a json file which
you can easily access in any programming language (for example C++ when using Geant4).
By inputting these values in your simulation, AIDO is able to optimize the parameters
automatically.

For more details, see the API section.

## References

```{toctree}
:maxdepth: 2
:caption: Guides

guides/getting_started
guides/usage
guides/example
```

```{toctree}
:maxdepth: 1
:caption: API

api/toc
```