# AIDO Codebase Overview

Table of content

Orchestration:

 * [Main](aido.main)
 * [Scheduler](aido.scheduler)
 * [Config](aido.config)
 * [Task](aido.task)

Frontend Interfaces:
 * [Simulation](aido.simulation_helpers)
    - SimulationParameter (single parameter)
    - SimulationParameterDictionary (container with IO features)
 * [User Interface](aido.interface)

Machine Learning Models

 * [Optimization Parameters](aido.optimization_helpers)
    - Continuous Parameters
    - Discrete Parameters
    - `ParameterModule` for dynamic Parameter building
 * [Surrogate](aido.surrogate)
 * [Training](aido.training)

Other:

 * [Logging](aido.logger)
 * [Plotting](aido.plotting)
