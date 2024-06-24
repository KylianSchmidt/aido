import b2luigi
import os
import json
from simulation.SimulationHelpers import SimulationParameterDictionary, SimulationParameter
from simulation.generator import GenerateNewParameters
from simulation.SimulationHelpers import GatherResults


class StartSimulationTask(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    initial_parameter_dict_file_path = b2luigi.PathParameter(hashed=True)

    def output(self):
        yield self.add_to_output("simulation_output")
        yield self.add_to_output("param_dict.json")

    def run(self):
        """ Workflow:
         1. Generate a new set of parameters based on the previous iteration
         TODO Do not generate new parameters itself but instead get them from WrapperTask

         2. Execute the container with the geant4 simulation software
            TODO the container should be executed by a script provided by the end user
        """
        output_path = self.get_output_file_name("simulation_output")
        parameter_dict_path = self.get_output_file_name("param_dict.json")

        parameters = SimulationParameterDictionary.from_json(initial_parameter_dict_file_path)
        parameters.to_json(parameter_dict_path)

        os.system(
            f"singularity exec -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/calo_opt_sim.py {parameter_dict_path} {output_path}"
        )


class Reconstruction(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    initial_parameter_dict_file_path = b2luigi.PathParameter(hashed=True)

    def output(self):
        yield self.add_to_output("reconstruction_output")
        yield self.add_to_output("param_dict.json")

    def requires(self):
        yield StartSimulationTask(
            initial_parameter_dict_file_path=initial_parameter_dict_file_path,
            simulation_task_rng_seed=self.simulation_task_rng_seed,
            )

    def run(self):
        """
        For each root file produced by the simulation Task, start a container with the reconstruction algorithm.
        Afterwards, the parameter dictionary used to generate these results are also passed as output

        TODO For now, only the latest file is the output of this Task. Try to merge the output if it is split
        into several files

        Alternative container: /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
        """
        output_file_path = self.get_output_file_name("reconstruction_output")
        parameter_dict_file_path = self.get_output_file_name("param_dict.json")

        param_dict = SimulationParameterDictionary.from_json(self.get_input_file_names("param_dict.json")[0])
        param_dict.to_json(parameter_dict_file_path)

        for input_file_path in self.get_input_file_names("simulation_output"):
            os.system(
                f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
                container_examples/calo_opt/reconstruction.py {input_file_path} {parameter_dict_file_path} {output_file_path}"
            )


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
                )

    def run(self):
        """ Gather the results into a file that is passed to the preprocessing Task for the optimization model.
        map: json file -> reco output
        """
        reconstruction_array = GatherResults.from_numpy_files(
            self.get_input_file_names("reconstruction_output"), delimiter=",", dtype=float
            )
        parameter_list = GatherResults.from_parameter_dicts(
            self.get_input_file_names("param_dict.json")
        )
        assert reconstruction_array.shape[0] == len(parameter_list), "Mismatched lengths."
        print("RECO ARRY", reconstruction_array)
        print("parameter list", parameter_list)
        # TODO combine them into one file
        # TODO write to file
        # TODO Run the surrogate model here?


if __name__ == "__main__":
    num_simulation_threads = 2
    os.system("rm ./results -rf")
    b2luigi.set_setting("result_dir", "results")

    sim_param_dict = SimulationParameterDictionary(
        [
            SimulationParameter('thickness_absorber_0', 0.7642903, min_value=1E-3),
            SimulationParameter('thickness_absorber_1', 10.469371, min_value=1E-3),
            SimulationParameter('thickness_scintillator_0', 30.585306, min_value=1E-3),
            SimulationParameter('thickness_scintillator_1', 22.256506, min_value=1E-3)
        ]
    )

    os.makedirs("./parameters", exist_ok=True)  # make /parameters a variable name
    initial_parameter_dict_file_path = "./parameters/initial_param_dict.json"
    sim_param_dict.to_json(initial_parameter_dict_file_path)

    b2luigi.process(
        SimulationWrapperTask(
            num_simulation_tasks=2,
            initial_parameter_dict_file_path=initial_parameter_dict_file_path
            ),
        workers=num_simulation_threads
        )
    
    os.system("rm *.pkl")
