# Code from Jonas Eppelt for the use of b2luigi and htcondor

local_condor_settings = {
    "requirements": '(TARGET.ProvidesCPU == True) && (TARGET.ProvidesEKPResources == True) && (Target.CloudSite =?= "schnepf")',
    "accounting_group": "belle", 
    "universe": "docker",
    "docker_image": "mschnepf/slc7-condocker",
    "stream_output": "true",
    "stream_error": "true",
    "when_to_transfer_output": "ON_SUCCESS",
    "transfer_executable": "true",
    "getenv": "false",
}

global_htcondor_settings = {
    # "requirements": '(Machine != "f03-001-140-e.gridka.de")',
    "+remotejob": "true",
    "request_cpus": "1",
    "accounting_group": "belle",
    "universe": "docker",
    "docker_image": "mschnepf/slc7-condocker",
    "stream_output": "true",
    "stream_error": "true",
    "transfer_executable": "true",
    "when_to_transfer_output": "ON_SUCCESS",
    "ShouldTransferFiles": "True",
    "getenv": "false",
    "+evictable": "True",
}

# Code to add to the Wrapping Task
"""
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
"""