import b2luigi
import os
from simulation.SimulationHelpers import SimulationParameterDictionary, SimulationParameter
from simulation.generator import GenerateNewParameters


class StartSimulationTask(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    simulation_container_file_path = b2luigi.PathParameter(hashed=True)
    initial_parameter_dict_file_path = b2luigi.PathParameter(hashed=True)

    def output(self):
        yield self.add_to_output("simulation_output")
        yield self.add_to_output("param_dict.json")

    def run(self):
        """ Workflow:
         1. Generate a new set of parameters based on the previous iteration

         2. Execute the container with the geant4 simulation software
            TODO the container should be executed by a script provided by the end user
        """
        new_parameter_generator = GenerateNewParameters(self.initial_parameter_dict_file_path)
        new_parameter_dict_file_path = self.get_output_file_name("param_dict.json")
        new_parameter_generator.decrease_by_half().to_json(new_parameter_dict_file_path)

        param_dict = SimulationParameterDictionary.from_json(new_parameter_dict_file_path)
        parameter_of_interest = param_dict.parameter_list[0].current_value
        output_file_path = self.get_output_file_name("simulation_output")

        os.system(
            f"singularity exec --home /work/kschmidt docker://python python3 \
            {self.simulation_container_file_path} {output_file_path} {parameter_of_interest}")


class Reconstruction(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    initial_parameter_dict_file_path = b2luigi.PathParameter(hashed=True)
    simulation_container_file_path = b2luigi.PathParameter(hashed=True)
    reconstruction_container_file_path = b2luigi.PathParameter(hashed=True)

    def output(self):
        yield self.add_to_output("reconstruction_output")
        yield self.add_to_output("param_dict.json")

    def requires(self):
        yield StartSimulationTask(
            initial_parameter_dict_file_path=initial_parameter_dict_file_path,
            simulation_task_rng_seed=self.simulation_task_rng_seed,
            simulation_container_file_path=self.simulation_container_file_path
            )

    def run(self):
        """
        For each root file produced by the simulation Task, start a container with the reconstruction algorithm.
        Afterwards, the parameter dictionary used to generate these results are also passed as output

        TODO For now, only the latest file is the output of this Task. Try to merge the output if it is split
        into several files
        """
        output_file_path = self.get_output_file_name("reconstruction_output")
        for input_file_path in self.get_input_file_names("simulation_output"):
            os.system(
                    f"singularity exec --home /work/kschmidt/ docker://python python3 \
                    {self.reconstruction_container_file_path} {input_file_path} {output_file_path}"
                )

        param_dict = SimulationParameterDictionary.from_json(self.get_input_file_names("param_dict.json")[0])
        param_dict.to_json(self.get_output_file_name("param_dict.json"))


class SimulationWrapperTask(b2luigi.WrapperTask):
    num_simulation_tasks = b2luigi.IntParameter()
    initial_parameter_dict_file_path = b2luigi.PathParameter()

    def requires(self):
        """ Create Tasks for each set of simulation parameters

        TODO Have the parameters from the previous iteration and pass them to each sub-task
        """
        for i in range(self.num_simulation_tasks):
            yield self.clone(
                Reconstruction,
                parameter_dict_file_path=self.initial_parameter_dict_file_path,
                simulation_task_rng_seed=i,
                simulation_container_file_path="container_examples/simulation_test.py",
                reconstruction_container_file_path="container_examples/reconstruction_test.py"
                )

    def run(self):
        """ TODO Start the reconstruction Tasks, which in turn each start a simulation Task. Once the
        requirements are met (all Tasks completed), gather the results into a file that is passed
        to the preprocessing Task for the optimization model.
        map: json file -> reco output
        """
        ...


if __name__ == "__main__":
    num_simulation_threads = 5
    os.system("rm ./results -rf")
    b2luigi.set_setting("result_dir", "results")

    sim_param_dict = SimulationParameterDictionary(
        [
            SimulationParameter("foo", 16.0),
            SimulationParameter("energy", 1000, optimizable=False)
        ]
    )

    os.makedirs("./parameters", exist_ok=True)  # make /parameters a variable name
    initial_parameter_dict_file_path = "./parameters/initial_param_dict.json"
    sim_param_dict.to_json(initial_parameter_dict_file_path)

    b2luigi.process(
        SimulationWrapperTask(
            num_simulation_tasks=5,
            initial_parameter_dict_file_path=initial_parameter_dict_file_path),
        workers=num_simulation_threads
        )
