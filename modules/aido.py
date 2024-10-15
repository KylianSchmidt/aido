import inspect
import json
import os
from functools import wraps
from typing import List

import b2luigi

from interface import AIDOUserInterface
from modules.plotting import AIDOPlotting
from modules.simulation_helpers import SimulationParameter, SimulationParameterDictionary


class StartSimulationTask(b2luigi.Task):
    iteration = b2luigi.IntParameter()
    simulation_task_id = b2luigi.IntParameter()
    iter_start_param_dict_file_path = b2luigi.PathParameter(hashed=True, significant=False)

    def output(self):
        yield self.add_to_output("simulation_output")
        yield self.add_to_output("param_dict.json")

    def run(self):
        """ Workflow:
         1. Generate a new set of parameters based on the previous iteration
         2. Start geant4 simulations using the 'interface.simulate' method provided by the user
        """
        output_path = self.get_output_file_name("simulation_output")
        output_parameter_dict_path = self.get_output_file_name("param_dict.json")

        start_parameters = SimulationParameterDictionary.from_json(self.iter_start_param_dict_file_path)
        parameters = start_parameters.generate_new()
        parameters.to_json(output_parameter_dict_path)
        interface.simulate(output_parameter_dict_path, output_path)


class IteratorTask(b2luigi.Task):
    """ This Task requires n='num_simulation_tasks' of StartSimulationTask before running. If the output of
    this Task exists, then it will be completely skipped.
    When running, it calls the user-provided 'interface.merge()' and 'interface.reconstruct' methods. The
    output of the later is passed to the Surrogate/Optimizer.
    """
    iteration = b2luigi.IntParameter()
    num_simulation_tasks = b2luigi.IntParameter(significant=False)
    iter_start_param_dict_file_path = b2luigi.PathParameter(hashed=True, significant=False)
    results_dir = b2luigi.PathParameter(hashed=True, significant=False)

    def output(self):
        """
        'reco_output_df': store the output of the reconstruction model
        'reconstruction_input_file_path': the simulation output files are kept
            in this file to be passed to the reconstruction model
        'param_dict.json': parameter dictionary file path
        """
        yield self.add_to_output("reco_output_df")
        yield self.add_to_output("reco_input_df")  # Not an output file
        yield self.add_to_output("param_dict.json")
        yield self.add_to_output("reco_paths_dict")

    def requires(self):
        """ Create Tasks for each set of simulation parameters. The seed 'num_simulation_tasks' ensures that
        b2luigi does not skip any Task due to duplicates.

        TODO Have the parameters from the previous iteration and pass them to each sub-task
        TODO Check that the param_dict of is of the same shape as the previous one (in case the user changes
        something in the SPD and then continues training)
        """
        
        self.next_param_dict_file = f"{self.results_dir}/parameters/param_dict_iter_{self.iteration + 1}.json"

        if not os.path.isfile(self.next_param_dict_file):

            for i in range(self.num_simulation_tasks):
                yield self.clone(
                    StartSimulationTask,
                    iteration=self.iteration,
                    iter_start_param_dict_file_path=self.iter_start_param_dict_file_path,
                    simulation_task_id=i,
                )

    def run(self):
        """ For each root file produced by the simulation Task, start a container with the reconstruction algorithm.
        Afterwards, the parameter dictionary used to generate these results are also passed as output

        Alternative container:
            /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest

        Current parameter dict is the main parameter dict of this iteration that was used to generate the
            simulations. It is fed to the Reconstruction and Surrogate/Optimizer models as input
        Updated parameter dict is the output of the optimizer and is saved as the parameter dict of the
            next iteration (becoming its current parameter)
        Next parameter dict is the location of the next iteration's parameter dict, if already exists, the
            whole Tasks is skipped. Otherwise, the updated parameter dict is saved in this location
        """
        parameter_dict_file_paths = self.get_input_file_names("param_dict.json")
        simulation_file_paths = self.get_input_file_names("simulation_output")
        self.reco_paths_dict = {
            "own_path": str(self.get_output_file_name("reco_paths_dict")),
            "surrogate_model_previous_path": f"{self.results_dir}/models/surrogate_{self.iteration - 1}.pt",
            "optimizer_model_previous_path": f"{self.results_dir}/models/optimizer_{self.iteration - 1}.pt",
            "surrogate_model_save_path": f"{self.results_dir}/models/surrogate_{self.iteration}.pt",
            "optimizer_model_save_path": f"{self.results_dir}/models/optimizer_{self.iteration}.pt",
            "current_parameter_dict": str(self.iter_start_param_dict_file_path),
            "updated_parameter_dict": str(self.get_output_file_name("param_dict.json")),
            "next_parameter_dict": f"{self.results_dir}/parameters/param_dict_iter_{self.iteration + 1}.json",
            "reco_input_df": str(self.get_output_file_name("reco_input_df")),
            "reco_output_df": str(self.get_output_file_name("reco_output_df")),
            "optimizer_loss_save_path": f"{self.results_dir}/loss/optimizer/optimizer_loss_{self.iteration}",
            "constraints_loss_save_path": f"{self.results_dir}/loss/constraints/contraints_loss_{self.iteration}"
        }
        if os.path.isfile(self.next_param_dict_file):
            print(f"Iteration {self.iteration} has an updated parameter dict already and will be skipped")
            return None

        with open(self.reco_paths_dict["own_path"], "w") as file:
            json.dump(self.reco_paths_dict, file)

        # Run the reconstruction algorithm
        interface.merge(parameter_dict_file_paths, simulation_file_paths, self.reco_paths_dict["reco_input_df"])
        interface.reconstruct(self.reco_paths_dict["reco_input_df"], self.reco_paths_dict["reco_output_df"])

        # Run surrogate and optimizer model
        os.system(f"python3 modules/training_script.py {self.reco_paths_dict["own_path"]}")

        new_param_dict = SimulationParameterDictionary.from_json(self.reco_paths_dict["updated_parameter_dict"])
        new_param_dict.iteration = self.iteration + 1
        new_param_dict.to_json(self.reco_paths_dict["next_parameter_dict"])

        # Plot the evolution
        # TODO Make it accessible to the end user to add plotting scripts
        if True:
            AIDOPlotting.plot(results_dir=self.results_dir)


class AIDOMainTask(b2luigi.Task):
    """ Trigger recursive calls for each Iteration
    TODO Fix exit condition in 'run' method
    TODO parameter results dir
    TODO Unable to resume optimization because only condition is that iteration 0 worked
    """
    num_max_iterations = b2luigi.IntParameter(significant=False)
    num_simulation_tasks = b2luigi.IntParameter(significant=False)
    start_param_dict_file_path = b2luigi.PathParameter(hashed=True)
    results_dir = b2luigi.PathParameter(hashed=True, significant=False)

    def run(self):
        for iteration in range(0, self.num_max_iterations):
            yield IteratorTask(
                iteration=iteration,
                num_simulation_tasks=self.num_simulation_tasks,
                iter_start_param_dict_file_path=f"{self.results_dir}/parameters/param_dict_iter_{iteration}.json",
                results_dir=self.results_dir
            )


class AIDO:
    """
    AIDO
    ----
    
    The AI Detector Optimization framework (AIDO) is a tool for finding the optimal
    design of particle physics detectors. By interpolating the results of simulations
    with slightly different geometries, it can learn the best set of detector parameters.

    Using b2luigi Tasks, this framework can run parallel simulation Tasks and
    reconstruction and optimization ML models on GPUs.

    Remarks
    -------
    For geant4 simulations with multi-threading capabilities, it is advisable to work in single-threaded mode.

    Workflow
    --------
    The internal optimization loop is structured as follows:

        1) Start a total of 'simulation_tasks' simulations using 'threads'.

        2) Merge and write an input file for the Reconstruction Task

        3) Run the Reconstruction Task

        4) Convert to pandas.DataFrame for the Optimizer model

        5) Run the Optimizer, which predicts the best set of parameters for this iteration

    Repeat for a number of 'max_iterations'.

    Results
    -------
    The 'results' directory will contain:

        - 'models': pytorch files with the states and weights of the surrogate and optimizer models

        - 'parameters': a list of parameter dictionaries (.json) with the parameters of each iteration

        - 'plots': evolution plots of the parameters TODO

        - 'task_outputs': Task outputs generated by the b2luigi scheduler. Can be safely removed to
            save disc space.
    """

    def optimize(
            parameters: List[SimulationParameter] | SimulationParameterDictionary,
            user_interface: AIDOUserInterface,
            simulation_tasks: int = 1,
            max_iterations: int = 50,
            threads: int = 1,
            results_dir: str | os.PathLike = "./results/"
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
        """
        b2luigi.set_setting("result_dir", f"{results_dir}/task_outputs")
        os.makedirs(f"{results_dir}/parameters", exist_ok=True)
        os.makedirs(f"{results_dir}/models", exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        os.makedirs(f"{results_dir}/loss/optimizer", exist_ok=True)
        os.makedirs(f"{results_dir}/loss/constraints", exist_ok=True)
        start_param_dict_file_path = f"{results_dir}/parameters/param_dict_iter_0.json"

        if isinstance(parameters, list):
            parameters = SimulationParameterDictionary(parameters)
        parameters.to_json(start_param_dict_file_path)

        assert (
            parameters.get_current_values("list") != []
        ), "Simulation Parameter Dictionary requires at least one optimizable Parameter."

        if inspect.isclass(user_interface):
            user_interface = user_interface()
        assert (
            issubclass(type(user_interface), AIDOUserInterface)
        ), f"The class {user_interface} must inherit from {AIDOUserInterface}."

        global interface  # Fix for b2luigi, as passing b2luigi.Parameter of non-serializable classes is not possible
        interface = user_interface

        b2luigi.process(
            AIDOMainTask(
                start_param_dict_file_path=start_param_dict_file_path,
                num_simulation_tasks=simulation_tasks,
                num_max_iterations=max_iterations,
                results_dir=results_dir
            ),
            workers=threads,
        )

    def parameter(*args, **kwargs):
        """ Create a new Simulation Parameter

        Args
        ----
                name (str): The name of the parameter.
                starting_value (Any): The starting value of the parameter.
                current_value (Any, optional): The current value of the parameter. Defaults to None.
                units (str, optional): The units of the parameter. Defaults to None.
                optimizable (bool, optional): Whether the parameter is optimizable. Defaults to True.
                min_value (float, optional): The minimum value of the parameter. Defaults to None.
                max_value (float, optional): The maximum value of the parameter. Defaults to None.
                sigma (float, optional): The standard deviation of the parameter. Defaults to None.
                discrete_values (Iterable, optional): The allowed discrete values of the parameter. Defaults to None.
                cost (float, optional): A float that quantifies the cost per unit of this Parameter. Defaults to None.
        """

        @wraps(SimulationParameter.__init__)
        def wrapper(*args, **kwargs):
            return SimulationParameter(*args, **kwargs)

        return wrapper(*args, **kwargs)
    
    def parameter_dict(*args, **kwargs):
        """
        Decorator function that wraps the initialization of the SimulationParameterDictionary class.
        """

        @wraps(SimulationParameterDictionary.__init__)
        def wrapper(*args, **kwargs):
            return SimulationParameterDictionary(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def check_results_folder_format(directory: str | os.PathLike):
        """
        Checks if the specified directory is of the 'results' format specified by AIDO.optimize().

        Args:
            directory (str | os.PathLike): The path to the directory to check.

        Returns:
            bool: True if the directory contains all the required folders
                  ("loss", "models", "parameters", "plots", "task_outputs"),
                  False otherwise.
        """
        existing_folders = set(os.listdir(directory))
        required_folders = set(["loss", "models", "parameters", "plots", "task_outputs"])
        return True if required_folders.issubset(existing_folders) else False
