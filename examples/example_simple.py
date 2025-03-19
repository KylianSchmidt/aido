import os

from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class

import aido

if __name__ == "__main__":
    aido.optimize(
        parameters=[
            aido.SimulationParameter("thickness_scintillator_0", 5.0, min_value=1E-3, max_value=15.0, sigma=1.5),
            aido.SimulationParameter(
                "material_scintillator_0",
                "G4_POLYSTYRENE",
                discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
                cost=[2500.0, 0.01],
            ),
            aido.SimulationParameter("num_events", 400, optimizable=False),
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=40,
        threads=20,
        results_dir="./results",
        max_iterations=50,
    )
    os.system("rm *.root")
