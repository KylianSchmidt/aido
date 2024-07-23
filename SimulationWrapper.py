import b2luigi
import os
import json
from simulation.SimulationHelpers import SimulationParameterDictionary, SimulationParameter
from modules import ReconstructionExample


class StartSimulationTask(b2luigi.Task):
    simulation_task_rng_seed = b2luigi.IntParameter()
    iter_start_param_dict_file_path = b2luigi.PathParameter(hashed=True)

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
        output_parameter_dict_path = self.get_output_file_name("param_dict.json")

        start_parameters = SimulationParameterDictionary.from_json(self.iter_start_param_dict_file_path)
        parameters = start_parameters.generate_new(rng_seed=self.simulation_task_rng_seed)
        parameters.to_json(output_parameter_dict_path)

        os.system(
            f"singularity exec -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/simulation.py {output_parameter_dict_path} {output_path}"
        )


class IteratorTask(b2luigi.Task):
    """ This Task wraps around ReconstructionTask and might become redundant in the future
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
        yield self.add_to_output("reco_file_paths_dict")

    def requires(self):
        """ Create Tasks for each set of simulation parameters

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

        TODO For now, only the latest file is the output of this Task. Try to merge the output if it is split
        into several files

        Alternative container: /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
        """ 
        parameter_dict_file_paths = self.get_input_file_names("param_dict.json")
        simulation_file_paths = self.get_input_file_names("simulation_output")
        reconstruction_input_df_path = self.get_output_file_name("reco_input_df")
        self.reco_file_paths_dict = {
            "own_path": str(self.get_output_file_name("reco_file_paths_dict")),
            "surrogate_model_previous_path": f"./results/models/surrogate_{self.iteration_counter - 1}.pt",
            "optimizer_model_previous_path": f"./results/models/optimizer_{self.iteration_counter - 1}.pt",
            "surrogate_model_save_path": f"./results/models/surrogate_{self.iteration_counter}.pt",
            "optimizer_model_save_path": f"./results/models/optimizer_{self.iteration_counter}.pt",
            "current_parameter_dict": str(self.iter_start_param_dict_file_path),
            "updated_parameter_dict": str(self.get_output_file_name("param_dict.json")),
            "next_parameter_dict": f"./results/parameters/param_dict_iter_{self.iteration_counter + 1}.json",
            "reco_input_df": str(reconstruction_input_df_path),
            "reco_output_df": str(self.get_output_file_name("reco_output_df"))
        }
        if os.path.isfile(self.next_param_dict_file):
            print(f"Iteration {self.iteration_counter} has an updated parameter dict already and will be skipped")
            return None

        with open(self.reco_file_paths_dict["own_path"], "w") as file:
            json.dump(self.reco_file_paths_dict, file)

        # Run the reconstruction algorithm
        reco = ReconstructionExample()
        reco.merge(parameter_dict_file_paths, simulation_file_paths, self.reco_file_paths_dict["reco_input_df"])
        reco.run(self.reco_file_paths_dict["reco_output_df"])

        # Run surrogate and optimizer model

        # Update parameter dict if not exist
        self.next_param_dict_file = self.reco_file_paths_dict["next_parameter_dict"]

        if os.path.isfile(self.next_param_dict_file):
            # Dont change anything, just propagate the values for b2luigi
            updated_parameter_dict_file_path = self.next_param_dict_file
            updated_param_dict = SimulationParameterDictionary.from_json(updated_parameter_dict_file_path)
            updated_param_dict = updated_param_dict.get_current_values(format="dict")
        else:
            with open(self.reco_file_paths_dict["updated_parameter_dict"], "r") as file:
                updated_param_dict: dict = json.load(file)

        initial_param_dict = SimulationParameterDictionary.from_json(self.iter_start_param_dict_file_path)

        print("DEBUG updated_param_dict\n", updated_param_dict)
        param_dict = initial_param_dict.update_current_values(updated_param_dict)
        param_dict.to_json(self.next_param_dict_file)
        param_dict.to_json(self.reco_file_paths_dict["updated_parameter_dict"])

        os.system("rm *.pkl")


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


if __name__ == "__main__":
    os.system("rm ./results -rf")
    b2luigi.set_setting("result_dir", "results/task_outputs")

    sim_param_dict = SimulationParameterDictionary([
        SimulationParameter('thickness_absorber_0', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
        SimulationParameter('thickness_absorber_1', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
        SimulationParameter('thickness_scintillator_0', 0.5, min_value=1E-3, max_value=1.0, sigma=0.2),
        SimulationParameter('thickness_scintillator_1', 0.1, min_value=1E-3, max_value=1.0, sigma=0.2),
        SimulationParameter("num_events", 300, optimizable=False)
    ])

    os.makedirs("./results/parameters", exist_ok=True)
    os.makedirs("./results/models", exist_ok=True)
    start_param_dict_file_path = "./results/parameters/param_dict_iter_0.json"
    sim_param_dict.to_json(start_param_dict_file_path)

    b2luigi.process(
        AIDOMainWrapperTask(
            start_param_dict_file_path=start_param_dict_file_path,
            num_simulation_tasks=5,
            num_max_iterations=50,
        ),
        workers=5,
    )
    os.system("rm *.pkl")
