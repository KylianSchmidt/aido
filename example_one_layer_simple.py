import os

from container_examples.calo_opt.interface_calo_opt import AIDOUserInterfaceExample  # Import your derived class
from modules.aido import AIDO  # required

if __name__ == "__main__":
    AIDO.optimize(
        parameters=[
            AIDO.parameter('thickness_absorber_0', 1.0, min_value=0.0, max_value=50.0, sigma=0.5),
            AIDO.parameter('thickness_scintillator_0', 1.0, min_value=1.0, max_value=25.0, sigma=0.5),
            AIDO.parameter("num_events", 400, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=10,
        max_iterations=55,
        threads=11,
        results_dir="/results_nikhil/results_20241017"
    )
    os.system("rm *.root")
