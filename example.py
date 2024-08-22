import os
from modules.aido import AIDO  # required
from container_examples.calo_opt.calo_opt_interface import AIDOUserInterfaceExample  # Import your derived class

global_htcondor_settings = {
    # "requirements": '(Machine != "f03-001-140-e.gridka.de")',
    "+remotejob": "true",
    "request_cpus": "1",
    "universe": "docker",
    "docker_image": "mschnepf/slc7-condocker",
    "stream_output": "true",
    "stream_error": "true",
    "transfer_executable": "true",
    "when_to_transfer_output": "ON_SUCCESS",
    "ShouldTransferFiles": "True",
    "getenv": "True",
    "+evictable": "True",
}


if __name__ == "__main__":
    os.system("rm ./results -rf")  # remove everything from results

    AIDO(
        parameters=[
            AIDO.parameter('thickness_absorber_0', 10.0, min_value=1E-3, max_value=50.0, sigma=0.1),
            AIDO.parameter('thickness_scintillator_0', 5.0, min_value=1E-3, max_value=10.0, sigma=0.1),
            AIDO.parameter('thickness_absorber_1', 10.0, min_value=1E-3, max_value=50.0, sigma=0.1),
            AIDO.parameter('thickness_scintillator_1', 1.0, min_value=1E-3, max_value=10.0, sigma=0.1),
            AIDO.parameter('thickness_absorber_2', 10.0, min_value=1.0, max_value=50.0, sigma=0.1),
            AIDO.parameter('thickness_scintillator_2', 1.0, min_value=0.05, max_value=10.0, sigma=0.1),
            AIDO.parameter("num_events", 100, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=10,
        threads=11
    )

    os.system("rm *.root")
