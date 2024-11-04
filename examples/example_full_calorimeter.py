import os

import numpy as np
import torch
from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class

import aido


class UIFullCalorimeter(AIDOUserInterfaceExample):

    @classmethod
    def constraints(self, parameter_dict: aido.SimulationParameterDictionary) -> torch.Tensor:

        detector_length = 0.0
        cost = 0.0

        for i in range(3):
            for name in ["absorber", "scintillator"]:
                cost += (
                    parameter_dict[f"thickness_{name}_{i}"].current_value
                    * np.array(parameter_dict[f"material_{name}_{i}"].weighted_cost)
                )
                detector_length += parameter_dict[f"thickness_{name}_{i}"].current_value

        detector_length_loss = torch.mean(
            10.0 * torch.nn.ReLU()(torch.tensor(detector_length - parameter_dict["max_length"].current_value)) ** 2
        )
        max_cost = parameter_dict["max_cost"].current_value
        max_cost_penalty = torch.mean(2.0 / max_cost * torch.nn.ReLU()(torch.tensor(cost) - max_cost) ** 2)
        return detector_length_loss + max_cost_penalty


if __name__ == "__main__":

    sigma = 5.0
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 5.0, min_value=0.1, max_value=50.0, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_0", 5.0, min_value=1.0, max_value=25.0, sigma=sigma),
        aido.SimulationParameter("material_absorber_0", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[25, 4.166]),
        aido.SimulationParameter(
            "material_scintillator_0",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01]
        ),
        aido.SimulationParameter("thickness_absorber_1", 5.0, min_value=0.1, max_value=50.0, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_1", 5.0, min_value=1.0, max_value=25.0, sigma=sigma),
        aido.SimulationParameter("material_absorber_1", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[25, 4.166]),
        aido.SimulationParameter(
            "material_scintillator_1",
            "G4_PbWO4",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01]
        ),
        aido.SimulationParameter("thickness_absorber_2", 5.0, min_value=0.1, max_value=50.0, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_2", 5.0, min_value=1.0, max_value=25.0, sigma=sigma),
        aido.SimulationParameter("material_absorber_2", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[25, 4.166]),
        aido.SimulationParameter(
            "material_scintillator_2",
            "G4_PbWO4",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01]
        ),
        aido.SimulationParameter("num_events", 500, optimizable=False),
        aido.SimulationParameter("max_length", 200, optimizable=False),
        aido.SimulationParameter("max_cost", 50_000, optimizable=False),
        aido.SimulationParameter("full_calorimeter", True, optimizable=False)
    ])

    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=30,
        max_iterations=200,
        threads=15,
        results_dir="/work/kschmidt/aido/results_full_calorimeter/results_20241104",
        description="""
            Full Calorimeter with cost and length constraints.
            Improved normalization of reconstructed array in Surrogate Model
            Using boosted parameter dict output by optimizer
        """
    )
    os.system("rm *.root")
