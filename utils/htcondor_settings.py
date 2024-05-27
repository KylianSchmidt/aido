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