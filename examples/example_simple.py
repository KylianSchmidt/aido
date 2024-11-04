import os

import aido
from examples.calo_opt.interface_calo_opt import AIDOUserInterfaceExample  # Import your derived class

if __name__ == "__main__":
    os.system("rm ./results -rf")  # remove everything from results

    aido.optimize(
        parameters=[
            aido.SimulationParameter('thickness_absorber_0', 10.0, min_value=1E-3, max_value=50.0, sigma=0.1),
            aido.SimulationParameter('thickness_scintillator_0', 5.0, min_value=1E-3, max_value=10.0, sigma=0.1),
            aido.SimulationParameter('thickness_absorber_1', 10.0, min_value=1E-3, max_value=50.0, sigma=0.1),
            aido.SimulationParameter('thickness_scintillator_1', 1.0, min_value=1E-3, max_value=10.0, sigma=0.1),
            aido.SimulationParameter('thickness_absorber_2', 10.0, min_value=1.0, max_value=50.0, sigma=0.1),
            aido.SimulationParameter('thickness_scintillator_2', 1.0, min_value=0.05, max_value=10.0, sigma=0.1),
            aido.SimulationParameter("num_events", 10, optimizable=False),
            aido.SimulationParameter("simple_setup", True, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=2,
        threads=2,
        results_dir="./results_simple_test_994124",
        max_iterations=1
    )

    os.system("rm *.root")
    os.system("rm ./results_simple_test_994124 -rf")
