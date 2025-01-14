import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class
from calo_opt.plotting import CaloOptPlotting

import aido


class UIFullCalorimeter(AIDOUserInterfaceExample):

    material_scaling_factor = 1.0

    @classmethod
    def constraints(
            self,
            parameter_dict: aido.SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> torch.Tensor:

        detector_length = 0.0
        cost = 0.0
        materials = {
            "absorber": {"costly": 25.0, "cheap": 4.166},
            "scintillator": {"costly": 2500.0, "cheap": 0.01}
        }

        for i in range(3):
            for name in ["absorber", "scintillator"]:
                layer_thickness = parameter_dict_as_tensor[f"thickness_{name}_{i}"]
                layer_choice = parameter_dict_as_tensor[f"material_{name}_{i}"]
                layer_composition = torch.sigmoid(self.material_scaling_factor * layer_choice)
                layer_cost_per_unit = (
                    layer_composition * materials[name]["costly"]
                    + (1 - layer_composition) * materials[name]["cheap"]
                )

                cost += layer_thickness * layer_cost_per_unit
                detector_length += layer_thickness

        self.material_scaling_factor += 0.08
        max_length = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length_penalty = torch.mean(10.0 * torch.nn.ReLU()(detector_length - max_length)**2)
        max_cost_penalty = torch.mean(2.0 / max_cost * torch.nn.ReLU()(cost - max_cost)**2)
        return detector_length_penalty + max_cost_penalty
    
    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:
        calo_opt_plotter = CaloOptPlotting(self.results_dir)
        calo_opt_plotter.plot_energy_resolution_all()
        calo_opt_plotter.plot_energy_resolution_first_and_last()
        calo_opt_plotter.plot_energy_resolution_evolution()
        calo_opt_plotter.plot_calorimeter_sideview()
        plt.close("all")
        return None


if __name__ == "__main__":
    min_value = 0.001
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", np.random.uniform(0.1, 5), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_0", np.random.uniform(1, 5), min_value=min_value),
        aido.SimulationParameter("material_absorber_0", 0.0, min_value=-1.0, max_value=1.0),
        aido.SimulationParameter("material_scintillator_0", 0.0, min_value=-1.0, max_value=1.0),
        aido.SimulationParameter("thickness_absorber_1", np.random.uniform(0.1, 5), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_1", np.random.uniform(0.1, 10), min_value=min_value),
        aido.SimulationParameter("material_absorber_1", 0.0, min_value=-1.0, max_value=1.0),
        aido.SimulationParameter("material_scintillator_1", 0.0, min_value=-1.0, max_value=1.0),
        aido.SimulationParameter("thickness_absorber_2", np.random.uniform(0.1, 5), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_2", np.random.uniform(0.1, 10), min_value=min_value),
        aido.SimulationParameter("material_absorber_2", 0.0, min_value=-1.0, max_value=1.0),
        aido.SimulationParameter("material_scintillator_2", 0.0, min_value=-1.0, max_value=1.0),
        aido.SimulationParameter("num_events", 400, optimizable=False),
        aido.SimulationParameter("max_length", 200, optimizable=False),
        aido.SimulationParameter("max_cost", 50_000, optimizable=False),
        aido.SimulationParameter("nikhil_material_choice", True, optimizable=False)
    ])

    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=20,
        max_iterations=200,
        threads=20,
        results_dir="/work/kschmidt/aido/results_material_choice/results_20250109",
        description="""
            With constraints, pions and photons, boundaries and adjustements to the covariance matrix.
            With same material composition loss as Nikhil
        """
    )
    os.system("rm *.root")
