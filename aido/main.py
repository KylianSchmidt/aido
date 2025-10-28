import os
from typing import Any, List, Type

from aido.config import AIDOConfig
from aido.interface import UserInterfaceBase
from aido.scheduler import start_scheduler
from aido.simulation_helpers import SimulationParameter, SimulationParameterDictionary


def optimize(
        parameters: List[SimulationParameter] | SimulationParameterDictionary,
        user_interface: UserInterfaceBase | Type[UserInterfaceBase],
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
