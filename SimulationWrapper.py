import b2luigi
import os
from utils.SimulationHelpers import SimulationParameterDictionary, SimulationParameter
from utils.generator import GenerateNewParameters


class StartSimulationTask(b2luigi.Task):
    parameter = b2luigi.IntParameter()

    def output(self):
        return b2luigi.LocalTarget("output.txt")

    def run(self):
        generator_new_parameters = GenerateNewParameters("./sim_param_dict.json")
        param_dict = generator_new_parameters.increase_by_random_number()
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

    b2luigi.process(
        SimulationWrapperTask(num_simulation_tasks=5),
        workers=num_simulation_threads
        )

    os.system("rm ./output.txt")
