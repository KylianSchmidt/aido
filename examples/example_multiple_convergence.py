"""
This example is based on the 'full calorimeter' example, but we run multiple trials
sequentially to demonstrate generalized convergence from different starting configurations.
"""
import os

import numpy as np
from example_full_calorimeter import UIFullCalorimeter

import aido

sigma: float = 2.5
min_value: float = 0.0
starting_seed = np.random.randint(0, 1000)
results_dir: str = f"/work/kschmidt/aido/results_convergence/seed_{starting_seed}"

rng = np.random.default_rng(seed=starting_seed)
parameters = aido.SimulationParameterDictionary([
    aido.SimulationParameter("thickness_absorber_0", rng.uniform(1, 30), min_value=min_value, sigma=sigma),
    aido.SimulationParameter("thickness_scintillator_0", rng.uniform(1, 30), min_value=min_value, sigma=sigma),
    aido.SimulationParameter(
        "material_absorber_0",
        rng.choice(["G4_Pb", "G4_Fe"]),
        discrete_values=["G4_Pb", "G4_Fe"],
        cost=[25, 4.166],
    ),
    aido.SimulationParameter(
        "material_scintillator_0",
        rng.choice(["G4_PbWO4", "G4_POLYSTYRENE"]),
        discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
        cost=[2500.0, 0.01],
    ),
    aido.SimulationParameter("thickness_absorber_1", rng.uniform(1, 30), min_value=min_value, sigma=sigma),
    aido.SimulationParameter("thickness_scintillator_1", rng.uniform(1, 30), min_value=min_value, sigma=sigma),
    aido.SimulationParameter(
        "material_absorber_1",
        rng.choice(["G4_Pb", "G4_Fe"]),
        discrete_values=["G4_Pb", "G4_Fe"],
        cost=[25, 4.166],
    ),
    aido.SimulationParameter(
        "material_scintillator_1",
        rng.choice(["G4_PbWO4", "G4_POLYSTYRENE"]),
        discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
        cost=[2500.0, 0.01],
    ),
    aido.SimulationParameter("thickness_absorber_2", rng.uniform(1, 30), min_value=min_value, sigma=sigma),
    aido.SimulationParameter("thickness_scintillator_2", rng.uniform(1, 30), min_value=min_value, sigma=sigma),
    aido.SimulationParameter(
        "material_absorber_2",
        rng.choice(["G4_Pb", "G4_Fe"]),
        discrete_values=["G4_Pb", "G4_Fe"],
        cost=[25, 4.166],
    ),
    aido.SimulationParameter(
        "material_scintillator_2",
        rng.choice(["G4_PbWO4", "G4_POLYSTYRENE"]),
        discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
        cost=[2500.0, 0.01],
    ),
    aido.SimulationParameter("num_events", 400, optimizable=False),
    aido.SimulationParameter("max_length", 200, optimizable=False),
    aido.SimulationParameter("max_cost", 200_000, optimizable=False),
])


if __name__ == "__main__":
    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=20,
        max_iterations=100,
        threads=20,
        results_dir=results_dir,
        description=f"Starting seed: {starting_seed}"
    )
    os.system("rm *.root")
