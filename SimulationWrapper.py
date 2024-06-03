import b2luigi
import os
from simulation.SimulationHelpers import SimulationParameterDictionary, SimulationParameter
from simulation.generator import GenerateNewParameters


class StartSimulationTask(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    simulation_container_file_path = b2luigi.PathParameter()
    parameter_dict_file_path = b2luigi.PathParameter()

    def output(self):
        return b2luigi.LocalTarget(f"results/simulation/output_{self.simulation_task_rng_seed}")

    def run(self):
        """ Workflow:
         1. Generate a new set of parameters based on the previous iteration

         2. Execute the container with the geant4 simulation software
            TODO the container should be executed by a script provided by the end user
        """
        generator_new_parameters = GenerateNewParameters(self.parameter_dict_file_path)
        param_dict = generator_new_parameters.increase_by_random_number(self.simulation_task_rng_seed)
        parameter_of_interest = param_dict.parameter_list[0].current_value
        output_file_path = self.output().path

        os.system(f"singularity exec --home /work/kschmidt docker://python python3 {self.simulation_container_file_path} {output_file_path} {parameter_of_interest}")


class Reconstruction(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    parameter_dict_file_path = b2luigi.PathParameter()
    simulation_container_file_path = b2luigi.PathParameter()
    reconstruction_container_file_path = b2luigi.PathParameter()

    def output(self):
        return b2luigi.LocalTarget(f"results/reconstruction/output_{self.simulation_task_rng_seed}")

    def requires(self):
        StartSimulationTask(
            parameter_dict_file_path=self.parameter_dict_file_path,
            simulation_task_rng_seed=self.simulation_task_rng_seed,
            simulation_container_file_path=self.simulation_container_file_path
            )

    def run(self):
        output_file_path = self.output().path
        parameter_of_interest = "test"
        os.system(f"singularity exec --home /work/kschmidt/ docker://python python3 {self.reconstruction_container_file_path} {output_file_path} {parameter_of_interest}")


class SimulationWrapperTask(b2luigi.WrapperTask):
    num_simulation_tasks = b2luigi.IntParameter()

    def requires(self):
        """ Create Tasks for each set of simulation parameters

        TODO Have the parameters from the previous iteration and pass them to each sub-task

        TODO Pipeline will be:
        WrapperTask (this) requires -> Reconstruction Task requires -> Simulation Task
        """
        for i in range(self.num_simulation_tasks):
            yield self.clone(
                Reconstruction,
                parameter_dict_file_path="./sim_param_dict.json",
                simulation_task_rng_seed=i,
                simulation_container_file_path="container_examples/simulation_test.py",
                reconstruction_container_file_path="container_examples/reconstruction_test.py"
                )

    def run(self):
        ...


if __name__ == "__main__":
    num_simulation_threads = 5
    os.system("rm ./results -rf")
    b2luigi.set_setting("result_dir", "results")

    param_foo = SimulationParameter("foo", 1.0)
    sim_param_dict = SimulationParameterDictionary()

    sim_param_dict.add_parameter(param_foo)
    sim_param_dict.to_json("./sim_param_dict.json")

    b2luigi.process(
        SimulationWrapperTask(num_simulation_tasks=5),
        workers=num_simulation_threads
        )
