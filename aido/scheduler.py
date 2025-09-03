import inspect
import json
import os
import time
from typing import Dict, Generator

import b2luigi
import torch

from aido.config import AIDOConfig
from aido.interface import UserInterfaceBase
from aido.logger import logger
from aido.plotting import Plotting
from aido.simulation_helpers import SimulationParameterDictionary
from aido.task import AIDOTask, torch_safe_wrapper
from aido.training import training_loop


class SimulationTask(AIDOTask):
    iteration = b2luigi.IntParameter()
    validation = b2luigi.BoolParameter()
    simulation_task_id = b2luigi.IntParameter()
    num_simulation_tasks = b2luigi.IntParameter(significant=False)
    num_validation_tasks = b2luigi.IntParameter(significant=False)
    start_param_dict_filepath = b2luigi.PathParameter(hashed=True, significant=False)
    results_dir = b2luigi.PathParameter(hashed=True, significant=False)

    def requires(self):
        if self.iteration > 0:
            return OptimizationTask(
                iteration=self.iteration - 1,
                num_simulation_tasks=self.num_simulation_tasks,
                num_validation_tasks=self.num_validation_tasks,
                results_dir=self.results_dir,
            )

    def output(self) -> Generator:
        yield self.add_to_output("param_dict.json")
        yield self.add_to_output("simulation_output")

    def run(self) -> None:
        """ Workflow:
         1. Generate a new set of parameters based on the previous iteration
         2. Start geant4 simulations using the 'interface.simulate' method provided by the user
        """
        output_path = self.get_output_file_name("simulation_output")
        output_parameter_dict_path = self.get_output_file_name("param_dict.json")

        start_parameters = SimulationParameterDictionary.from_json(self.start_param_dict_filepath)

        if self.simulation_task_id == 0:
            parameters = start_parameters
            parameters.rng_seed = start_parameters.generate_new().rng_seed
        else:
            parameters = start_parameters.generate_new()

        parameters.to_json(output_parameter_dict_path)
        interface.simulate(output_parameter_dict_path, output_path)


class ReconstructionTask(AIDOTask):
    iteration = b2luigi.IntParameter()
    validation = b2luigi.BoolParameter()
    num_simulation_tasks = b2luigi.IntParameter(significant=False)
    num_validation_tasks = b2luigi.IntParameter(significant=False)
    start_param_dict_filepath = b2luigi.PathParameter(hashed=True, significant=False)
    results_dir = b2luigi.PathParameter(hashed=True, significant=False)

    def requires(self) -> Generator:
        assert isinstance(self.validation, bool), "'validation' parameter must be of type bool."

        if self.validation:
            num_simulations = self.num_validation_tasks
        else:
            num_simulations = self.num_simulation_tasks

        for i in range(num_simulations):
            yield self.clone(
                SimulationTask,
                iteration=self.iteration,
                validation=self.validation,
                simulation_task_id=i,
                num_simulation_tasks=self.num_simulation_tasks,
                num_validation_tasks=self.num_validation_tasks,
                start_param_dict_filepath=self.start_param_dict_filepath,
                results_dir=self.results_dir
            )

    def output(self) -> Generator:
        """
        Define the output files for the task based on the validation parameter.
        """
        if self.validation:
            yield self.add_to_output("validation_input_df")
            yield self.add_to_output("validation_output_df")
        else:
            yield self.add_to_output("reco_input_df")
            yield self.add_to_output("reco_output_df")

    def run(self) -> None:
        """
        Run the reconstruction process. The type of processing depends on the validation flag.
        """
        output_type = "reco" if not self.validation else "validation"
        
        interface.merge(
            parameter_dict_file_paths=self.get_input_file_names("param_dict.json"),
            simulation_file_paths=self.get_input_file_names("simulation_output"),
            reco_input_path=self.get_output_file_name(f"{output_type}_input_df")
        )
        
        interface.reconstruct(
            reco_input_path=self.get_output_file_name(f"{output_type}_input_df"),
            reco_output_path=self.get_output_file_name(f"{output_type}_output_df"),
            is_validation=self.validation
        )


class OptimizationTask(AIDOTask):
    """ This Task requires n='num_simulation_tasks' of StartSimulationTask before running. If the output of
    this Task exists, then it will be completely skipped.
    When running, it calls the user-provided 'interface.merge()' and 'interface.reconstruct()' methods. The
    output of the later is passed to the Surrogate/Optimizer.
    """
    iteration = b2luigi.IntParameter()
    num_simulation_tasks = b2luigi.IntParameter(significant=False)
    num_validation_tasks = b2luigi.IntParameter(significant=False)
    results_dir = b2luigi.PathParameter(hashed=True, significant=False)

    def output(self) -> Generator:
        if self.iteration >= 0:
            yield self.add_to_output("reco_paths_dict")
            yield self.add_to_output("param_dict.json")

    def requires(self) -> Generator:
        """ Starts the Reconstruction Tasks for regular reconstruction and for validation (the latter only
        if 'num_validation_tasks' > 0).
        """
        def start_reconstruction_task(num_simulations: int, validation: bool = False) -> Generator:
            yield ReconstructionTask(
                iteration=self.iteration,
                validation=validation,
                num_simulation_tasks=num_simulations,
                num_validation_tasks=self.num_validation_tasks,
                start_param_dict_filepath=f"{self.results_dir}/parameters/param_dict_iter_{self.iteration}.json",
                results_dir=self.results_dir,
            )

        if self.iteration < 0:
            return None

        if self.num_validation_tasks:
            yield from start_reconstruction_task(self.num_validation_tasks, validation=True)
        yield from start_reconstruction_task(self.num_simulation_tasks)

    def create_reco_path_dict(self) -> Dict:
        return {
            "results_dir": str(self.results_dir),
            "own_path": str(self.get_output_file_name("reco_paths_dict")),
            "config_path": f"{self.results_dir}/config.json",
            "surrogate_model_previous_path": f"{self.results_dir}/models/surrogate_{self.iteration - 1}.pt",
            "optimizer_model_previous_path": f"{self.results_dir}/models/optimizer_{self.iteration - 1}.pt",
            "surrogate_model_save_path": f"{self.results_dir}/models/surrogate_{self.iteration}.pt",
            "optimizer_model_save_path": f"{self.results_dir}/models/optimizer_{self.iteration}.pt",
            "current_parameter_dict": f"{self.results_dir}/parameters/param_dict_iter_{self.iteration}.json",
            "next_parameter_dict": f"{self.results_dir}/parameters/param_dict_iter_{self.iteration + 1}.json",
            "reco_output_df": str(self.get_input_file_names("reco_output_df")[0]),
            "validation_output_df": None,
            "optimizer_loss_save_path": f"{self.results_dir}/loss/optimizer/optimizer_loss_{self.iteration}",
            "constraints_loss_save_path": f"{self.results_dir}/loss/constraints/constraints_loss_{self.iteration}",
            "surrogate_loss_save_path": f"{self.results_dir}/loss/surrogate/surrogate_loss_{self.iteration}"
        }

    def run(self) -> None:
        """ For each root file produced by the simulation Task, start a container with the reconstruction algorithm.
        Afterwards, the parameter dictionary used to generate these results are also passed as output

        Current parameter dict is the main parameter dict of this iteration that was used to generate the
            simulations. It is fed to the Reconstruction and Surrogate/Optimizer models as input
        Updated parameter dict is the output of the optimizer and is saved as the parameter dict of the
            next iteration (becoming its current parameter)
        Next parameter dict is the location of the next iteration's parameter dict, if already exists, the
            whole Tasks is skipped. Otherwise, the updated parameter dict is saved in this location
        """
        if self.iteration == -1:
            return None

        self.reco_paths_dict = self.create_reco_path_dict()
        config = AIDOConfig.from_json(os.path.join(self.results_dir, "config.json"))

        with open(self.reco_paths_dict["own_path"], "w") as file:
            json.dump(self.reco_paths_dict, file)

        # Run surrogate and optimizer models
        num_training_loop_tries: int = 0
        training_loop_out_of_memory: bool = True

        while training_loop_out_of_memory:
            try:
                training_loop_out_of_memory = False
                new_param_dict = torch_safe_wrapper(
                    training_loop,
                    reco_file_paths_dict=self.reco_paths_dict["own_path"],
                    reconstruction_loss_function=interface.loss,
                    constraints=interface.constraints,
                )
            except torch.cuda.OutOfMemoryError as e:
                training_loop_out_of_memory = True
                num_training_loop_tries += 1
                torch.cuda.empty_cache()
                time.sleep(config.scheduler.training_delay_between_retries)

                if num_training_loop_tries > config.scheduler.training_num_retries:
                    raise e

        new_param_dict.iteration = self.iteration + 1
        # TODO Change datetime too
        new_param_dict.to_json(self.reco_paths_dict["next_parameter_dict"])
        new_param_dict.to_json(self.get_output_file_name("param_dict.json"))

        # Plot results
        Plotting.plot(results_dir=self.results_dir)
        try:
            interface.plot(parameter_dict=new_param_dict)
        except Exception as e:
            logger.warning(f"The following Exception was raised during user-defined plotting:\n{e}")


def start_scheduler(
    parameters: SimulationParameterDictionary,
    user_interface: UserInterfaceBase,
    simulation_tasks: int,
    max_iterations: int,
    threads: int,
    results_dir: str | os.PathLike,
    validation_tasks: int = 0,
    **kwargs,
):
    b2luigi.set_setting("result_dir", f"{results_dir}/task_outputs")
    os.makedirs(f"{results_dir}", exist_ok=True)
    assert os.path.isdir(results_dir), f"Provided results directory '{results_dir}' is not valid."
    os.makedirs(f"{results_dir}/parameters", exist_ok=True)
    os.makedirs(f"{results_dir}/models", exist_ok=True)
    os.makedirs(f"{results_dir}/plots/validation/reco_model/on_trainingData", exist_ok=True)
    os.makedirs(f"{results_dir}/plots/validation/reco_model/on_validationData", exist_ok=True)
    os.makedirs(f"{results_dir}/plots/validation/surrogate/on_trainingData", exist_ok=True)
    os.makedirs(f"{results_dir}/plots/validation/surrogate/on_validationData", exist_ok=True)
    os.makedirs(f"{results_dir}/loss/optimizer", exist_ok=True)
    os.makedirs(f"{results_dir}/loss/constraints", exist_ok=True)
    os.makedirs(f"{results_dir}/loss/surrogate", exist_ok=True)
    parameters.to_json(f"{results_dir}/parameters/param_dict_iter_0.json")

    assert (
        parameters.get_current_values("list") != []
    ), "Simulation Parameter Dictionary requires at least one optimizable Parameter."

    if inspect.isclass(user_interface):
        user_interface = user_interface()

    assert (
        issubclass(type(user_interface), UserInterfaceBase)
    ), f"The class {user_interface} must inherit from {UserInterfaceBase}."

    global config
    config = AIDOConfig.from_json("config.json")
    config.to_json(os.path.join(results_dir, "config.json"))

    global interface  # Fix for b2luigi, as passing b2luigi.Parameter of non-serializable classes is not possible
    interface = user_interface
    interface.results_dir = results_dir

    b2luigi.process(
        OptimizationTask(
            num_simulation_tasks=simulation_tasks,
            num_validation_tasks=validation_tasks,
            iteration=max_iterations - 1,
            results_dir=results_dir
        ),
        workers=threads,
        **kwargs,
    )
