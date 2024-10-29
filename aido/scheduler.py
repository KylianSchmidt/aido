import inspect
import json
import os

import b2luigi

from aido.interface import AIDOBaseUserInterface
from aido.plotting import Plotting
from aido.simulation_helpers import SimulationParameterDictionary
from aido.training import training_loop


class SimulationTask(b2luigi.Task):
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


class OptimizationTask(b2luigi.Task):
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
                    SimulationTask,
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
            "constraints_loss_save_path": f"{self.results_dir}/loss/constraints/constraints_loss_{self.iteration}",
            "surrogate_loss_save_path": f"{self.results_dir}/loss/surrogate/surrogate_loss_{self.iteration}"
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
        training_loop(self.reco_paths_dict["own_path"], interface.constraints)

        new_param_dict = SimulationParameterDictionary.from_json(self.reco_paths_dict["updated_parameter_dict"])
        new_param_dict.iteration = self.iteration + 1
        new_param_dict.to_json(self.reco_paths_dict["next_parameter_dict"])

        # Plot the evolution
        # TODO Make it accessible to the end user to add plotting scripts
        if True:
            Plotting.plot(results_dir=self.results_dir)


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
            yield OptimizationTask(
                iteration=iteration,
                num_simulation_tasks=self.num_simulation_tasks,
                iter_start_param_dict_file_path=f"{self.results_dir}/parameters/param_dict_iter_{iteration}.json",
                results_dir=self.results_dir
            )


def start_scheduler(
        parameters: SimulationParameterDictionary,
        user_interface: AIDOBaseUserInterface,
        simulation_tasks: int,
        max_iterations: int,
        threads: int,
        results_dir: str | os.PathLike
        ):
    b2luigi.set_setting("result_dir", f"{results_dir}/task_outputs")
    os.makedirs(f"{results_dir}", exist_ok=True)
    os.makedirs(f"{results_dir}/parameters", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    os.makedirs(f"{results_dir}/loss/optimizer", exist_ok=True)
    os.makedirs(f"{results_dir}/loss/constraints", exist_ok=True)
    os.makedirs(f"{results_dir}/loss/surrogate", exist_ok=True)
    start_param_dict_file_path = f"{results_dir}/parameters/param_dict_iter_0.json"

    parameters.to_json(start_param_dict_file_path)

    assert (
        parameters.get_current_values("list") != []
    ), "Simulation Parameter Dictionary requires at least one optimizable Parameter."

    if inspect.isclass(user_interface):
        user_interface = user_interface()

    assert (
        issubclass(type(user_interface), AIDOBaseUserInterface)
    ), f"The class {user_interface} must inherit from {AIDOBaseUserInterface}."

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
