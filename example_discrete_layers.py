import os
from modules.aido import AIDO
from container_examples.calo_opt.interface import AIDOUserInterfaceExample


if __name__ == "__main__":
    os.system("rm ./results_discrete -rf")  # remove everything from results

    AIDO.optimize(
        parameters=[
            AIDO.parameter("thickness_absorber", 10.0, optimizable=False),
            AIDO.parameter("thickness_scintillator", 1.0, optimizable=False),
            AIDO.parameter("num_blocks", 1, discrete_values=list(range(0, 20))),
            AIDO.parameter("num_events", 200, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=4,
        threads=5,
        results_dir="./results_discrete/"
    )
