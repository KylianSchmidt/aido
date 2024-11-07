import os

import numpy as np

from container_examples.calo_opt.interface_calo_opt import AIDOUserInterfaceExample
from modules.aido import AIDO

if __name__ == "__main__":
    results_dir = "results_discrete_20241014"
    os.system(f"rm ./results_old/{results_dir} -rf")  # remove everything from results

    AIDO.optimize(
        parameters=[
            AIDO.parameter(
                "thickness_absorber",
                5.6,
                min_value=0.01,
                max_value=20.0,
                sigma=0.2,
                cost=1.1,
            ),
            AIDO.parameter(
                "thickness_scintillator",
                0.1,
                min_value=0.05,
                max_value=1.0,
                sigma=0.2,
                cost=5.0,
            ),
            AIDO.parameter(
                "absorber_material",
                "G4_Pb",
                discrete_values=["G4_Pb", "G4_W", "G4_Fe"],
                cost=[1.3, 0.26, 0.092],
            ),
            AIDO.parameter(
                "num_blocks",
                3,
                discrete_values=list(range(1, 10)),
                cost=(0.1 * np.arange(1, 10)).tolist(),
            ),
            AIDO.parameter("num_events", 200, optimizable=False),
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=10,
        threads=11,
        results_dir=f"./results_old/{results_dir}/",
        max_iterations=20,
    )
