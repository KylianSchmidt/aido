import b2luigi
import os


class StartSimulationTask(b2luigi.Task):
    parameter = b2luigi.IntParameter()

    def output(self):
        return b2luigi.LocalTarget("output.txt")

    def run(self):
        print(f"Running the Simulation Task with Parameter {self.parameter}")
        os.system(f"sleep {self.parameter}")
        os.system("singularity run docker://godlovedc/lolcow")

        with self.output().open("w") as file:
            file.write("")


class SimulationWrapperTask(b2luigi.WrapperTask):

    def requires(self):
        for i in range(3):
            yield self.clone(StartSimulationTask, parameter=i)


if __name__ == "__main__":
    b2luigi.process(SimulationWrapperTask(), workers=3)
    os.system("rm ./output.txt")
