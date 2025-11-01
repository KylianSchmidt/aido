import os
from typing import Dict

import torch
from calo_opt.interface import CaloOptInterface  # Import your derived class
from calo_opt.plotting import CaloOptPlotting

import aido


class UIFullCalorimeter(CaloOptInterface):

    @classmethod
    def constraints(
            self,
            parameter_dict: aido.SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> torch.Tensor:
        """ Additional constraints to add to the Loss.

        Use the Tensors found in 'parameter_dict_as_tensor' (Dict) to compute the constraints
        and return a 1-dimensional Tensor. Note that missing gradients at this stage will
        negatively impact the training of the optimizer.

        Use the usual 'parameter_dict' instance to access additional information such as
        boundaries, costs per item and all other stored values.

        In this example, we add the cost per layer for all six layers by looping over the
        index of the layer (0, 1, 2) and their type (absorber / scintillator). Using the

        """

        detector_length_list = []
        cost_list = []

        for i in range(3):
            for name in ["absorber", "scintillator"]:
                material_probabilities = parameter_dict_as_tensor[f"material_{name}_{i}"]
                material_cost = torch.tensor(
                    parameter_dict[f"material_{name}_{i}"].cost,
                    device=material_probabilities.device
                )
                layer_weighted_cost = material_probabilities * material_cost
                layer_thickness = parameter_dict_as_tensor[f"thickness_{name}_{i}"]

                cost_list.append(layer_thickness * layer_weighted_cost)
                detector_length_list.append(layer_thickness)

        max_length = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length = torch.stack(detector_length_list).sum()
        cost = torch.stack(cost_list).sum()
        detector_length_penalty = torch.mean(torch.nn.functional.relu((detector_length - max_length) / max_length)**2)
        max_cost_penalty = torch.mean(torch.nn.functional.relu((cost - max_cost) / max_cost)**2)
        return detector_length_penalty + max_cost_penalty

    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:
        calo_opt_plotter = CaloOptPlotting(self.results_dir)
        calo_opt_plotter.mplstyle()
        calo_opt_plotter.plot()
        return None


if __name__ == "__main__":
    sigma: float = 2.5
    min_value: float = 0.0

    ui_interface = UIFullCalorimeter()
    ui_interface.container_path = "/ceph/kschmidt/singularity_cache/minicalosim_latest.sif"
    ui_interface.container_extra_flags = "-B /work,/ceph"
    ui_interface.verbose = True
    results_dir: str = "/work/kschmidt/aido/results_example"

    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 9.030052185058594, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_0", 37.155208587646484, min_value=min_value, sigma=sigma),
        aido.SimulationParameter(
            "material_absorber_0",
            "G4_Fe",
            discrete_values=["G4_Pb", "G4_Fe"],
            cost=[25, 4.166],
            probabilities=[0.01, 0.99]
        ),
        aido.SimulationParameter(
            "material_scintillator_0",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01],
            probabilities=[0.99, 0.01]
        ),
        aido.SimulationParameter("thickness_absorber_1", 10.446663856506348, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_1", 29.665525436401367, min_value=min_value, sigma=sigma),
        aido.SimulationParameter(
            "material_absorber_1",
            "G4_Fe",
            discrete_values=["G4_Pb", "G4_Fe"],
            cost=[25, 4.166],
            probabilities=[0.99, 0.01]
        ),
        aido.SimulationParameter(
            "material_scintillator_1",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01],
            probabilities=[0.01, 0.99]
        ),
        aido.SimulationParameter("thickness_absorber_2", 36.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_2", 27.5, min_value=min_value, sigma=sigma),
        aido.SimulationParameter(
            "material_absorber_2",
            "G4_Fe",
            discrete_values=["G4_Pb", "G4_Fe"],
            cost=[25, 4.166],
            probabilities=[0.01, 0.99]
        ),
        aido.SimulationParameter(
            "material_scintillator_2",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01],
            probabilities=[0.01, 0.99]
        ),
        aido.SimulationParameter("num_events", 400, optimizable=False),
        aido.SimulationParameter("max_length", 200, optimizable=False),
        aido.SimulationParameter("max_cost", 200_000, optimizable=False),
    ])
    aido.optimize(
        parameters=parameters,
        user_interface=ui_interface,
        simulation_tasks=20,
        max_iterations=220,
        threads=20,
        results_dir=results_dir,
        description="""
Optimization of a sampling calorimeter with cost and length constraints.
Includes the optimization of discrete parameters and specific plotting functions
"""
    )
    os.system("rm *.root")
