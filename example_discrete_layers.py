import os
import numpy as np
from modules.aido import AIDO
from container_examples.calo_opt.interface import AIDOUserInterfaceExample


if __name__ == "__main__":
    os.system("rm ./results_discrete -rf")  # remove everything from results

    AIDO.optimize(
        parameters=[
            AIDO.parameter("thickness_absorber", 10.0, min_value=0.01, max_value=50.0, sigma=0.1, cost=1.1),
            AIDO.parameter("thickness_scintillator", 3.14, min_value=0.05, max_value=10.0, sigma=0.1, cost=13.0),
            AIDO.parameter("num_blocks", 5, discrete_values=list(range(1, 20)), cost=np.concat([np.arange(1, 10, 1), 5 * np.arange(10, 20)]).tolist()),
            AIDO.parameter("num_events", 400, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=10,
        threads=11,
        results_dir="./results_discrete/",
        max_iterations=50
    )
