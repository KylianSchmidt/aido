import b2luigi
import os
from typing import Dict
from utils.htcondor_settings import local_condor_settings, global_htcondor_settings


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
    batch_system = "htcondor"
    _resources 

    @property
    def htcondor_settings(self) -> Dict[str, str]:
        base_settings = global_htcondor_settings
        base_settings.update({
            f"request_{resource}": str(value) for resource, value in self._resources.items() 
        })
        base_settings.update({
            "transfer_executable": "true",
            "transfer_input_files": "",
            "when_to_transfer_output": "ON_SUCCESS",
            "transfer_output_files": "remote_setup.sh",
            "JobBatchName": self.stage,
        })
        return base_settings
    
    @property
    def resources(self) -> Dict[str, int]:
        if self.batch_system == "local":
            res = {key: value for key, value in self._resources.items() if key in ["cpus", "memory", "gpus"]}
            res.update({
                "local_workers": 1})
            return res
        else:
            return {"htcondor": 1}

    def requires(self):
        for i in range(3):
            yield self.clone(StartSimulationTask, parameter=i)


if __name__ == "__main__":
    b2luigi.process(SimulationWrapperTask(), workers=3)
    os.system("rm ./output.txt")
