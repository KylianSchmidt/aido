from container_examples.calo_opt.interface import AIDOUserInterfaceExample
from modules.aido import AIDO

if __name__ == "__main__":
    AIDO.optimize([
        AIDO.parameter("thickness_absorber", 20.0, max_value=50.0, min_value=0.1, sigma=0.5),
        AIDO.parameter("thickness_scintillator", 25.0, max_value=35.0, min_value=1.0, sigma=0.5),
        AIDO.parameter("absorber_material", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[1.3, 0.092]),
        AIDO.parameter("scintillator_material", "G4_PbWO4", discrete_values=["G4_PbWO4", "G4_Fe"], cost=[1.5, 1.0]),
        AIDO.parameter("num_blocks", 3, optimizable=False),
        AIDO.parameter("num_events", 200, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=19,
        threads=20,
        max_iterations=50,
        results_dir="./results_comparison/results_20241014_2/"
    )
