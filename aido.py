import os
from SimulationWrapper import AIDO
from simulation.SimulationHelpers import SimulationParameter, SimulationParameterDictionary
from interface import AIDOUserInterfaceExample


if __name__ == "__main__":
    os.system("rm ./results -rf")

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
