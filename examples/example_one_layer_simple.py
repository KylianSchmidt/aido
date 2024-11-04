import os

import torch

import aido
from examples.calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class

if __name__ == "__main__":
    class UISimple(AIDOUserInterfaceExample):
        def constraints(self, parameter_dict: aido.SimulationParameterDictionary) -> torch.Tensor:
            detector_length = (
                parameter_dict["thickness_absorber"].current_value
                + parameter_dict["thickness_scintillator"].current_value
            )
            detector_length_loss = torch.mean(
                10.0 * torch.nn.ReLU()(torch.tensor(detector_length - parameter_dict["max_length"].current_value))
            ) ** 2
            return detector_length_loss

    aido.optimize(
        parameters=[
            aido.SimulationParameter('thickness_absorber', 1.0, min_value=0.0, max_value=50.0, sigma=0.5),
            aido.SimulationParameter('thickness_scintillator', 1.0, min_value=0.05, max_value=25.0, sigma=0.5),
            aido.SimulationParameter("num_events", 400, optimizable=False),
            aido.SimulationParameter("simple_setup", True, optimizable=False),
            aido.SimulationParameter("max_length", 75.0, optimizable=False)
        ],
        user_interface=UISimple,
        simulation_tasks=10,
        max_iterations=55,
        threads=11,
        results_dir="./results_nikhil/results_20241018"
    )
    os.system("rm *.root")
