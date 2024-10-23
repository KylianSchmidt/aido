import argparse
import importlib
import importlib.util
import sys

from modules.scheduler import AIDO
from modules.simulation_helpers import SimulationParameterDictionary


def interface_loader(interface_path: str):
    """
    Load and return the specified class from the given file path.

    Args:
        interface_path (str): The path to the file and the name of the class in the format 'file_path:ClassName'.

    Returns:
        class: The specified class from the file.

    Raises:
        ValueError: If the interface path is not in the correct format.
    """
    if ":" not in interface_path:
        raise ValueError("Interface path must be of format 'file_path:ClassName'")

    file_path, class_name = interface_path.split(":")
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(description="Detector Optimization framework for Geant4 Simulations")

    parser.add_argument(
        "--parameter_file_path",
        type=str,
        help="Path to a .json file containing the Simulation Parameters. Format must be consistent with "
        "SimulationParameterDictionary."
    )
    parser.add_argument(
        "--interface",
        type=str,
        help="Interface class based on 'interface.py:AIDOUserInterface'. Must be in the format 'file_path:ClassName'"
    )
    parser.add_argument(
        "--simulation_tasks",
        type=int,
        default=1,
        help="Number of parallel Simulation Tasks. Defaults to 1."
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Maximum number of total iterations. Defaults to 50."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads allowed for the Simulation Tasks. Defaults to 1."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results/",
        help="Directory for the results. Defaults to './results/'"
    )
    args, b2luigi_args = parser.parse_known_args()
    param_dict = SimulationParameterDictionary.from_json(args.parameter_file_path)
    AIDO.optimize(
        param_dict,
        user_interface=interface_loader(args.interface),
        simulation_tasks=args.simulation_tasks,
        max_iterations=args.max_iterations,
        threads=args.threads,
        results_dir=args.results_dir
    )
    sys.argv[1:] = b2luigi_args

    print("DEBUG", sys.argv)


if __name__ == "__main__":
    main()
