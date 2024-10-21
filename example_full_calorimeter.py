import os

import torch

from container_examples.calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class
from modules.aido import AIDO  # required
from modules.simulation_helpers import SimulationParameterDictionary

if __name__ == "__main__":

    class UIFullCalorimeter(AIDOUserInterfaceExample):
        def constraints(self, parameter_dict: SimulationParameterDictionary) -> torch.Tensor:
            detector_length = (
                parameter_dict["thickness_absorber_0"].current_value
                + parameter_dict["thickness_scintillator_0"].current_value
                + parameter_dict["thickness_absorber_1"].current_value
                + parameter_dict["thickness_scintillator_1"].current_value
                + parameter_dict["thickness_absorber_2"].current_value
                + parameter_dict["thickness_scintillator_2"].current_value
            )
            detector_length_loss = torch.mean(
                10.0 * torch.nn.ReLU()(torch.tensor(detector_length - parameter_dict["max_length"].current_value))
            ) ** 2
            return detector_length_loss

    sigma = 3.0
    parameters = [
        AIDO.parameter("thickness_absorber_0", 40.0, min_value=0.1, max_value=50.0, sigma=sigma),
        AIDO.parameter("thickness_scintillator_0", 10.0, min_value=1.0, max_value=25.0, sigma=sigma),
        AIDO.parameter("material_absorber_0", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"]),
        AIDO.parameter("material_scintillator_0", "G4_Si", discrete_values=["G4_PbWO4", "G4_Si"]),
        AIDO.parameter("thickness_absorber_1", 40.0, min_value=0.1, max_value=50.0, sigma=sigma),
        AIDO.parameter("thickness_scintillator_1", 10.0, min_value=1.0, max_value=25.0, sigma=sigma),
        AIDO.parameter("material_absorber_1", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"]),
        AIDO.parameter("material_scintillator_1", "G4_Si", discrete_values=["G4_PbWO4", "G4_Si"]),
        AIDO.parameter("thickness_absorber_2", 40.0, min_value=0.1, max_value=50.0, sigma=sigma),
        AIDO.parameter("thickness_scintillator_2", 10.0, min_value=1.0, max_value=25.0, sigma=sigma),
        AIDO.parameter("material_absorber_2", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"]),
        AIDO.parameter("material_scintillator_2", "G4_Si", discrete_values=["G4_PbWO4", "G4_Si"]),
        AIDO.parameter("num_events", 200, optimizable=False),
        AIDO.parameter("max_length", 250, optimizable=False),
        AIDO.parameter("full_calorimeter", True, optimizable=False)
    ]
    AIDO.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=10,
        max_iterations=200,
        threads=11,
        results_dir="./results_full_calorimeter/results_20241021"
    )
    os.system("rm *.root")
