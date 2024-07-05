import b2luigi
import os
import pandas as pd
from typing import List
from simulation.SimulationHelpers import SimulationParameterDictionary, SimulationParameter, GatherResults
from simulation.conversion import convert_sim_to_reco


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
            container_examples/calo_opt/simulation.py {parameter_dict_path} {output_path}"
        )


class Reconstruction(b2luigi.Task):
    num_simulation_tasks = b2luigi.IntParameter()
    initial_parameter_dict_file_path = b2luigi.PathParameter(hashed=True)

    def output(self):
        """
        'reconstruction_output': store the output of the reconstruction model
        'reconstruction_input_file_path': the simulation output files are kept
            in this file to be passed to the reconstruction model
        'param_dict.json': parameter dictionary file path
        """
        yield self.add_to_output("reconstruction_output")
        yield self.add_to_output("reconstruction_input_file_path")  # Not an output file

    def requires(self):

        for i in range(self.num_simulation_tasks):
            yield self.clone(
                StartSimulationTask,
                initial_parameter_dict_file_path=initial_parameter_dict_file_path,
                simulation_task_rng_seed=i,
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
        parameter_dict_file_path = self.get_input_file_names("param_dict.json")
        simulation_file_paths = self.get_input_file_names("simulation_output")
        reconstruction_input_file_path = self.get_output_file_name("reconstruction_input_file_path")

        df_list: List[pd.DataFrame] = []
        
        for simulation_output_path in list(zip(parameter_dict_file_path, simulation_file_paths)):
            df_list.append(
                convert_sim_to_reco(
                    *simulation_output_path,
                    input_keys=[
                        'sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z',
                        'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer'
                    ],
                    target_keys=["true_energy"],
                    context_keys=["true_pid"]
                )
            )

        df: pd.DataFrame = pd.concat(df_list, axis=0, ignore_index=True)
        df.to_parquet(reconstruction_input_file_path, index=range(len(df)))

        os.system(
            f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/reconstruction.py {reconstruction_input_file_path} {output_file_path}"
        )


class SimulationWrapperTask(b2luigi.WrapperTask):
    num_simulation_tasks = b2luigi.IntParameter()
    initial_parameter_dict_file_path = b2luigi.PathParameter()

    def requires(self):
        """ Create Tasks for each set of simulation parameters

        TODO Have the parameters from the previous iteration and pass them to each sub-task
        """
        yield Reconstruction(
            initial_parameter_dict_file_path=self.initial_parameter_dict_file_path,
            num_simulation_tasks=self.num_simulation_tasks
        )
        
    def run(self):
        """ Read reconstruction output 
        """
        reco_output = self.get_input_file_names("reconstruction_output")[0]
        df = pd.read_parquet(reco_output)
        print("OUTPUT df\n", df)


if __name__ == "__main__":
    num_simulation_threads = 2
    os.system("rm ./results -rf")
    b2luigi.set_setting("result_dir", "results")

    sim_param_dict = SimulationParameterDictionary(
        [
            SimulationParameter('thickness_absorber_0', 0.7642903, min_value=1E-3),
            SimulationParameter('thickness_absorber_1', 10.469371, min_value=1E-3),
            SimulationParameter('thickness_scintillator_0', 30.585306, min_value=1E-3),
            SimulationParameter('thickness_scintillator_1', 22.256506, min_value=1E-3),
            SimulationParameter("num_events", 100, optimizable=False)
        ]
    )

    os.makedirs("./parameters", exist_ok=True)  # make /parameters a variable name
    initial_parameter_dict_file_path = "./parameters/initial_param_dict.json"
    sim_param_dict.to_json(initial_parameter_dict_file_path)

    b2luigi.process(
        SimulationWrapperTask(
            num_simulation_tasks=50,
            initial_parameter_dict_file_path=initial_parameter_dict_file_path
            ),
        workers=num_simulation_threads
        )
    
    os.system("rm *.pkl")
