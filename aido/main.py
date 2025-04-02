"""
The AI Detector Optimization framework (AIDO) is a tool for learning the optimal
design of particle physics detectors. By interpolating the results of simulations
with slightly different geometries, it can iteratively find the best set of detector
parameters.

This framework fragments the workflow into [b2luigi](https://b2luigi.readthedocs.io/en/stable/index.html)
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
the loss calculated by the reconstruction algorithm. The :class: `aido.SimulationParameter`
object is the basic building block for a parameter. It keeps track of the current value
during the optimization process as well as other useful information.

A set of parameters are combined into a single :class: `aido.SimulationParameterDictionary`
which has extra tools. Most relevant is the way we interface the AIDO framework with your
simulation and reconstruction. For this, the dictionary is stored as a `json` file which
you can easily access in any programming language (for example C++ when using Geant4).
By inputting these values in your simulation, AIDO is able to optimize the parameters
automatically.
"""

import os
from typing import Any, List

from aido.config import AIDOConfig
from aido.interface import UserInterfaceBase
from aido.scheduler import start_scheduler
from aido.simulation_helpers import SimulationParameter, SimulationParameterDictionary


def optimize(
        parameters: List[SimulationParameter] | SimulationParameterDictionary,
        user_interface: UserInterfaceBase,
        simulation_tasks: int = 1,
        max_iterations: int = 50,
        threads: int = 1,
        results_dir: str | os.PathLike = "./results/",
        description: str = "",
        validation_tasks: int = 0,
        **kwargs,
        ):
    """
    Args:
        parameters (List[AIDO.parameter] | SimulationParameterDictionary): Instance of a
            SimulationParameterDictionary with all the desired parameters. These are the starting parameters
            for the optimization loop and the outcome can depend on their starting values. Can also be a
            simple list of SimulationParameter / AIDO.parameters (the latter is a proxy method).
        user_interface (class or instance inherited from AIDOUserInterface): Regulates the interaction
            between user-defined code (simulation, reconstruction, merging of output files) and the
            AIDO workflow manager.
        simulation_tasks (int): Number of simulations started during each iteration.
        max_iterations (int): Maximum amount of iterations of the optimization loop
        threads (int): Allowed number of threads to allocate the simulation tasks.
            NOTE There is no benefit in having 'threads' > 'simulation_tasks' per se, but in some cases,
            errors involving missing dependencies after the simulation step can be fixed by setting:
            'threads' = 'simulation_tasks' + 1.
        results_dir (str): Indicates where to save the results. Useful when differentiating runs from
            each other.
        description (str, optional): Additional text associated with the run. Is saved in the parameter
            json files under 'metadata.description"
        validation_tasks (int): Control the number of simulation tasks dedicated only for validation
            purposes on top of the regular simulation tasks 'simulation_tasks'. Defaults to 'None' which
            is no validation tasks. This will also disable the call of 'interface.reconstruct' with
            'is_validation=True'.
        kwargs (key-word arguments, optional): Arguments to pass to 'b2luigi.process', such as

                - show_output: bool = False
                - dry_run: bool = False
                - test: bool = False
                - batch: bool = False
                - ignore_additional_command_line_args: bool = False

            See the corresponding documentation at https://b2luigi.readthedocs.io/en/stable/documentation/api.html
    """
    if isinstance(parameters, list):
        parameters = SimulationParameterDictionary(parameters)

    parameters.description += description

    start_scheduler(
        parameters=parameters,
        user_interface=user_interface,
        simulation_tasks=simulation_tasks,
        max_iterations=max_iterations,
        threads=threads,
        results_dir=results_dir,
        validation_tasks=validation_tasks,
        **kwargs,
    )


def check_results_folder_format(directory: str | os.PathLike) -> bool:
    """
    Checks if the specified directory is of the 'results' format specified by AIDO.optimize().

    Args:
        directory (str | os.PathLike): The path to the directory to check.

    Returns:
        bool
            - True if the directory contains all the required folders

                ("loss", "models", "parameters", "plots", "task_outputs"),

            - False otherwise.
    """
    existing_folders = set(os.listdir(directory))
    required_folders = set(["loss", "models", "parameters", "plots", "task_outputs"])
    return True if required_folders.issubset(existing_folders) else False


def set_config(key: str, value: Any):
    config = AIDOConfig.from_json("config.json")
    config.set_value(key, value)
    config.to_json("config.json")


def get_config(key: str) -> Any:
    return AIDOConfig.from_json("config.json").get_value(key)
