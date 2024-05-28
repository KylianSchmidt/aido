import b2luigi
import os
from utils.SimulationHelpers import SimulationParameterDictionary, SimulationParameter
import numpy as np  # just for testing purposes


class GenerateNewParameters(b2luigi.Task):
    seed = b2luigi.IntParameter()

    def output(self):
        return b2luigi.LocalTarget("parameter_dict.json")
    
    def run(self):
        param_dict = SimulationParameterDictionary.from_json("sim_param_dict.json")
        for parameter in param_dict.parameter_list:
            parameter.current_value = parameter._starting_value + np.random.randint(0, 1000)

        param_dict.to_json(self.output().path)


class StartSimulationTask(b2luigi.Task):
    parameter = b2luigi.IntParameter()

    def requires(self):
        return GenerateNewParameters(seed=self.parameter)

    def output(self):
        return b2luigi.LocalTarget("output.txt")

    def run(self):
        
        param_dict = SimulationParameterDictionary.from_json(
            self.get_input_file_names("parameter_dict.json")[0]
            )
        parameter_of_interest = param_dict.parameter_list[0].current_value

        os.system(f"singularity exec docker://python python3 ./test.py {parameter_of_interest}")

        with self.output().open("w") as file:
            file.write("")


class SimulationWrapperTask(b2luigi.WrapperTask):
    num_simulation_tasks = b2luigi.IntParameter()

    def requires(self):
        for i in range(self.num_simulation_tasks):
            yield self.clone(StartSimulationTask, parameter=i)

    def run(self):
        print("Ran SimulationWrapperTask")


if __name__ == "__main__":
    num_simulation_threads = 10

    param_foo = SimulationParameter("foo", 1.0)
    sim_param_dict = SimulationParameterDictionary()

    sim_param_dict.add_parameter(param_foo)
    sim_param_dict.to_json("./sim_param_dict.json")

    print(sim_param_dict.parameter_list[0].to_dict())

    b2luigi.process(
        SimulationWrapperTask(num_simulation_tasks=5),
        workers=num_simulation_threads
        )

    os.system("rm ./output.txt")
    os.system("rm ./parameter_dict.json")
