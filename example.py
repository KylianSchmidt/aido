import os
from modules.aido import AIDO  # required
from modules.simulation_helpers import SimulationParameter  # required
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
            SimulationParameter('thickness_absorber_0', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
            SimulationParameter('thickness_absorber_1', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
            SimulationParameter('thickness_scintillator_0', 0.5, min_value=1E-3, max_value=1.0, sigma=0.2),
            SimulationParameter('thickness_scintillator_1', 0.1, min_value=1E-3, max_value=1.0, sigma=0.2),
            SimulationParameter("num_events", 100, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=4,
        threads=5
    )
