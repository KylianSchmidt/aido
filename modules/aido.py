import b2luigi
import os
import json
import inspect
from typing import List, Any, Iterable
from modules.simulation_helpers import SimulationParameterDictionary, SimulationParameter
from interface import AIDOUserInterface


class StartSimulationTask(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    iter_start_param_dict_file_path = b2luigi.PathParameter(hashed=True)

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
        parameters = start_parameters.generate_new(rng_seed=self.simulation_task_rng_seed)
        parameters.to_json(output_parameter_dict_path)

        interface.simulate(output_parameter_dict_path, output_path)


class IteratorTask(b2luigi.Task):
    """ This Task requires n='num_simulation_tasks' of StartSimulationTask before running. If the output of
    this Task exists, then it will be completely skipped.
    When running, it calls the user-provided 'interface.merge()' and 'interface.reconstruct' methods. The
    output of the later is passed to the Surrogate/Optimizer.
    """
    iteration_counter = b2luigi.IntParameter()
    num_simulation_tasks = b2luigi.IntParameter()
    iter_start_param_dict_file_path = b2luigi.PathParameter(hashed=True)

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
        """
        
        self.next_param_dict_file = f"./results/parameters/param_dict_iter_{self.iteration_counter + 1}.json"

        if not os.path.isfile(self.next_param_dict_file):

            for i in range(self.num_simulation_tasks):
                yield self.clone(
                    StartSimulationTask,
                    iter_start_param_dict_file_path=self.iter_start_param_dict_file_path,
                    simulation_task_rng_seed=i,
                )

    def run(self):
        """ For each root file produced by the simulation Task, start a container with the reconstruction algorithm.
        Afterwards, the parameter dictionary used to generate these results are also passed as output

        Alternative container:
            /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
        """
        parameter_dict_file_paths = self.get_input_file_names("param_dict.json")
        simulation_file_paths = self.get_input_file_names("simulation_output")
        self.reco_paths_dict = {
            "own_path": str(self.get_output_file_name("reco_paths_dict")),
            "surrogate_model_previous_path": f"./results/models/surrogate_{self.iteration_counter - 1}.pt",
            "optimizer_model_previous_path": f"./results/models/optimizer_{self.iteration_counter - 1}.pt",
            "surrogate_model_save_path": f"./results/models/surrogate_{self.iteration_counter}.pt",
            "optimizer_model_save_path": f"./results/models/optimizer_{self.iteration_counter}.pt",
            "current_parameter_dict": str(self.iter_start_param_dict_file_path),
            "updated_parameter_dict": str(self.get_output_file_name("param_dict.json")),
            "next_parameter_dict": f"./results/parameters/param_dict_iter_{self.iteration_counter + 1}.json",
            "reco_input_df": str(self.get_output_file_name("reco_input_df")),
            "reco_output_df": str(self.get_output_file_name("reco_output_df"))
        }
        if os.path.isfile(self.next_param_dict_file):
            print(f"Iteration {self.iteration_counter} has an updated parameter dict already and will be skipped")
            return None

        with open(self.reco_paths_dict["own_path"], "w") as file:
            json.dump(self.reco_paths_dict, file)

        # Run the reconstruction algorithm
        interface.merge(parameter_dict_file_paths, simulation_file_paths, self.reco_paths_dict["reco_input_df"])
        interface.reconstruct(self.reco_paths_dict["reco_input_df"], self.reco_paths_dict["reco_output_df"])

        # Run surrogate and optimizer model
        os.system(f"python3 modules/training_script.py {self.reco_paths_dict["own_path"]}")

        # Update parameter dict if not exist
        if os.path.isfile(self.next_param_dict_file):
            # Dont change anything, just propagate the values for b2luigi
            updated_param_dict = SimulationParameterDictionary.from_json(self.reco_paths_dict["next_parameter_dict"])
            updated_param_dict = updated_param_dict.get_current_values(format="dict")
        else:
            with open(self.reco_paths_dict["updated_parameter_dict"], "r") as file:
                updated_param_dict = json.load(file)

        initial_param_dict = SimulationParameterDictionary.from_json(self.reco_paths_dict["current_parameter_dict"])

        new_param_dict = initial_param_dict.update_current_values(updated_param_dict)
        new_param_dict.to_json(self.reco_paths_dict["next_parameter_dict"])
        new_param_dict.to_json(self.reco_paths_dict["updated_parameter_dict"])


class AIDOMainWrapperTask(b2luigi.WrapperTask):
    """ Trigger recursive calls for each Iteration
    TODO Fix exit condition in 'run' method
    TODO parameter results dir
    """
    num_max_iterations = b2luigi.IntParameter()
    num_simulation_tasks = b2luigi.IntParameter()
    start_param_dict_file_path = b2luigi.PathParameter(hashed=True)

    def requires(self):
        yield IteratorTask(
            iteration_counter=0,
            num_simulation_tasks=self.num_simulation_tasks,
            iter_start_param_dict_file_path="./results/parameters/param_dict_iter_0.json"
        )

    def run(self):
        for iteration in range(1, self.num_max_iterations):
            yield IteratorTask(
                iteration_counter=iteration,
                num_simulation_tasks=self.num_simulation_tasks,
                iter_start_param_dict_file_path=f"./results/parameters/param_dict_iter_{iteration}.json"
            )


class AIDO:
    def __init__(
            self,
            parameters: List[SimulationParameter] | SimulationParameterDictionary,
            user_interface: AIDOUserInterface,
            simulation_tasks: int = 1,
            max_iterations: int = 50,
            threads: int = 1
            ):
        """
        AIDO
        ----
        
        The AI Detector Optimization framework (AIDO) is a tool for learning optimal
        design for particle physics detectors. By interpolating the results of simulations
        with slightly different geometries, it can learn the best set of detector parameters.

        Using b2luigi Tasks, this framework can run parallel simulation Tasks and running
        reconstruction and optimization ML models on GPUs.

        Args:
            sim_param_dict (SimulationParameterDictionary): Instance of a SimulationParameterDictionary
                with all the desired parameters. These are the starting parameters for the optimization
                loop and the outcome can depend on their starting values.
            user_interface (class or instance inherited from AIDOUserInterface): Regulates the interaction
                between user-defined code (simulation, reconstruction, merging of output files) and the
                AIDO workflow manager.
            simulation_tasks (int): Number of simulations started during each iteration.
            max_iterations (int): Maximum amount of iterations of the optimization loop
            threads (int): Allowed number of threads to allocate the simulation tasks. There is no benefit
                in having 'threads' > 'simulation_tasks'.

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

        b2luigi.set_setting("result_dir", "results/task_outputs")
        os.makedirs("./results/parameters", exist_ok=True)
        os.makedirs("./results/models", exist_ok=True)
        os.makedirs("./results/plots", exist_ok=True)
        start_param_dict_file_path = "./results/parameters/param_dict_iter_0.json"

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
            AIDOMainWrapperTask(
                start_param_dict_file_path=start_param_dict_file_path,
                num_simulation_tasks=simulation_tasks,
                num_max_iterations=max_iterations,
            ),
            workers=threads,
        )
        os.system("rm *.pkl")

    @classmethod
    def parameter(**kwargs):
        return SimulationParameter(**kwargs)
