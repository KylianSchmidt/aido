import aido
from examples.calo_opt.interface_calo_opt import AIDOUserInterfaceExample

if __name__ == "__main__":
    aido.optimize([
        aido.SimulationParameter("thickness_absorber", 1.0, units="cm", max_value=50.0, min_value=0.1, sigma=0.5),
        aido.SimulationParameter("thickness_scintillator", 0.5, units="cm", max_value=10.0, min_value=0.01, sigma=0.5),
        aido.SimulationParameter("absorber_material", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[1.3, 0.092]),
        aido.SimulationParameter(
            "scintillator_material", "G4_PbWO4", discrete_values=["G4_PbWO4", "G4_Fe"], cost=[1.5, 1.0]
        ),
        aido.SimulationParameter("num_blocks", 10, optimizable=False),
        aido.SimulationParameter("num_events", 400, optimizable=False)
        ],
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=11,
        threads=10,
        max_iterations=50,
        results_dir="./results_comparison/results_20241017_2/"
    )
