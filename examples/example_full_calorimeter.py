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
                layer_choice = parameter_dict[f"material_{name}_{i}"].current_value
                layer_cost_per_unit = materials[name]["costly"] if layer_choice >= 0 else materials[name]["cheap"]

                cost += layer_thickness * layer_cost_per_unit
                detector_length += layer_thickness

        self.material_scaling_factor += 0.08
        max_loss = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length_penalty = torch.mean(10.0 * torch.nn.ReLU()(detector_length - max_loss)**2)
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

    aido.SimulationParameter.set_config("sigma", 1.5)
    min_value = 0.001
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", np.random.uniform(0.1, 50), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_0", np.random.uniform(20, 35), min_value=min_value),
        aido.SimulationParameter("material_absorber_0", -1, optimizable=False),
        aido.SimulationParameter("material_scintillator_0", 1, optimizable=False),
        aido.SimulationParameter("thickness_absorber_1", np.random.uniform(0.1, 50), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_1", np.random.uniform(0.1, 35), min_value=min_value),
        aido.SimulationParameter("material_absorber_1", 1, optimizable=False),
        aido.SimulationParameter("material_scintillator_1", -1, optimizable=False),
        aido.SimulationParameter("thickness_absorber_2", np.random.uniform(0.1, 50), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_2", np.random.uniform(0.1, 10), min_value=min_value),
        aido.SimulationParameter("material_absorber_2", 1, optimizable=False),
        aido.SimulationParameter("material_scintillator_2", -1, optimizable=False),
        aido.SimulationParameter("num_events", 400, optimizable=False),
        aido.SimulationParameter("max_length", 200, optimizable=False),
        aido.SimulationParameter("max_cost", 50_000, optimizable=False),
        aido.SimulationParameter("nikhil_material_choice", True, optimizable=False)
    ])

    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=40,
        max_iterations=200,
        threads=10,
        results_dir="/work/kschmidt/aido/results_material_choice/results_20241209",
        description="""
            Full Calorimeter with cost and length constraints.
            Improved normalization of reconstructed array in Surrogate Model
            Using boosted parameter dict output by optimizer
            Reduced sigma
            One-Hot parameters in Reco and Surrogate
            Made reco results 1d (temporary!)
            Normalized reco loss in surrogate
            Separetely decrease the learning of discrete parameters
            Set discrete learning rate a bit higher (1e-4)
            With correct gradients for the constraints
            Replaced empty events with bad reco loss (fixed)
            Penalties for empty sensors
            Longer Surrogate training
            Add true energy to context and removed penalties from loss
            Add deposited energy to Context
            Increased sigma
            Add validation Tasks
            Changed Optimizer to compute reco loss itself
            Improvements to the Surrogate model training
            Actually implemented covariance box correctly
            Save reco model between iterations
            Discrete LR = 0.001, gradients clamped at 0.01
            Fixed material choice
        """
    )
    os.system("rm *.root")
