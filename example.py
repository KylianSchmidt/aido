import os
from modules.aido import AIDO  # required
from modules.simulation_helpers import SimulationParameter, SimulationParameterDictionary  # required
from container_examples.calo_opt.calo_opt_interface import AIDOUserInterfaceExample  # Import your derived class


if __name__ == "__main__":
    os.system("rm ./results -rf")  # remove everything from results

    sim_param_dict = SimulationParameterDictionary([
        SimulationParameter('thickness_absorber_0', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
        SimulationParameter('thickness_absorber_1', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
        SimulationParameter('thickness_scintillator_0', 0.5, min_value=1E-3, max_value=1.0, sigma=0.2),
        SimulationParameter('thickness_scintillator_1', 0.1, min_value=1E-3, max_value=1.0, sigma=0.2),
        SimulationParameter("num_events", 100, optimizable=False)
    ])

    AIDO(
        sim_param_dict,
        user_interface=AIDOUserInterfaceExample,
        simulation_tasks=4,
        threads=5
    )
