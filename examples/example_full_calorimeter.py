import glob
import os
import re
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class

import aido


class UIFullCalorimeter(AIDOUserInterfaceExample):

    @classmethod
    def constraints(
            self,
            parameter_dict: aido.SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> torch.Tensor:

        detector_length = 0.0
        cost = 0.0
        device = parameter_dict_as_tensor["thickness_absorber_0"].device

        for i in range(3):
            for name in ["absorber", "scintillator"]:
                layer_weighted_cost = torch.Tensor(parameter_dict[f"material_{name}_{i}"].cost)
                layer_thickness = parameter_dict_as_tensor[f"thickness_{name}_{i}"]
                layer_material = parameter_dict_as_tensor[f"material_{name}_{i}"]

                cost += layer_thickness * layer_material.dot(layer_weighted_cost.to(device))
                detector_length += layer_thickness

        max_loss = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length_penalty = torch.mean(10.0 * torch.nn.ReLU()(detector_length - max_loss)**2)
        max_cost_penalty = torch.mean(2.0 / max_cost * torch.nn.ReLU()(cost - max_cost)**2) / 10
        return detector_length_penalty + max_cost_penalty

    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:

        def add_plot_header(ax: plt.Axes) -> plt.Axes:
            plt.text(
                0.01, 0.99,
                "Detector Optimization",
                transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left'
            )
            plt.text(
                0.01, 0.93,
                "Stacked Calorimeter, Run 1\nParticles: " + r"$\pi^+\gamma, E=[1, 20]$" + " GeV",
                transform=ax.transAxes, fontsize=12, va='top', ha='left'
            )
            return ax

        def plot_energy_resolution_single(
                iteration: int,
                file_name: str | os.PathLike,
                bins: np.ndarray,
                color=None,
                label: str | None = None,
                ):
            df = pd.read_parquet(file_name)
            e_rec: pd.Series = df["Reconstructed"]["true_energy"] - df["Targets"]["true_energy"]
            plt.hist(
                e_rec,
                bins=bins,
                color=color,
                histtype="step",
                label=label,
                density=False
            )
            plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
            plt.xlabel("Energy resolution [GeV]")
        
        def plot_energy_resolution_all() -> None:
            dirs = glob.glob(f"{self.results_dir}/task_outputs/iteration=*/validation=False/reco_output_df")
            cmap = plt.get_cmap('coolwarm', len(dirs))
            fig, ax = plt.subplots(figsize=(8, 6))
            bins = np.linspace(-20, 20, 80 + 1)

            for file_name in dirs:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
                plot_energy_resolution_single(
                    iteration=iteration,
                    file_name=file_name,
                    bins=bins,
                    color=cmap(iteration),
                    label=(f"Iteration {iteration:3d}" if iteration % 20 == 0 or iteration == len(dirs) - 1 else None))

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles)))
            ax.legend(handles, labels)
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_all"))
            plt.close()

        def plot_energy_resolution_first_and_last() -> None:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = add_plot_header(ax)
            dirs = glob.glob(f"{self.results_dir}/task_outputs/iteration=*/validation=False/reco_output_df")
            cmap = plt.get_cmap('coolwarm', len(dirs))
            bins = np.linspace(-20, 20, 80 + 1)

            for file_name in dirs:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))

                if iteration == 0 or iteration == len(dirs) - 1:
                    df = pd.read_parquet(file_name)
                    e_rec = (df["Targets"] - df["Reconstructed"])
                    e_rec_binned, *_ = plt.hist(
                        e_rec,
                        bins=bins,
                        color=cmap(iteration),
                        histtype="step",
                        label=f"Iteration {iteration:3d}",
                    )
                    ax = aido.Plotting.fwhm(bins, e_rec_binned, ax=ax)
            plt.legend()
            plt.xlabel(r"Energy Resolution $E_{\text{true}} - E_{\text{rec}}$ [GeV]")
            plt.ylabel(f"Counts {(bins[1] - bins[0]):.2f}")
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_first_and_last"))
            plt.close()

        def plot_calorimeter_sideview() -> None:
            df_list = []
            df_materials_list = []
            parameter_dir = os.path.join(self.results_dir, "parameters/")

            for file_name in os.listdir(parameter_dir):
                param_dict = aido.SimulationParameterDictionary.from_json(parameter_dir + file_name)
                df_list.append(pd.DataFrame(
                    param_dict.get_current_values(format="dict", types="continuous"),
                    index=[param_dict.iteration],
                ))
                df_materials = pd.DataFrame(param_dict.get_probabilities()).drop(index=0)
                df_materials.index = [param_dict.iteration]
                df_materials_list.append(df_materials)

            df: pd.DataFrame = pd.concat(df_list, axis=0).sort_index()
            df_materials: pd.DataFrame = pd.concat(df_materials_list, axis=0).sort_index()
            df_materials.columns = df.columns

            fig, ax = plt.subplots(figsize=(10, 7))
            ax = add_plot_header(ax)
            absorber_cmap = plt.get_cmap("copper")
            scintillator_cmap = plt.get_cmap("spring")

            def get_color(label: str, prob: Iterable):
                if "absorber" in label:
                    return absorber_cmap(prob)
                if "scintillator" in label:
                    return scintillator_cmap(prob)
                else:
                    return "white"

            for i in df.index:
                bottom = 0
                plt.gca().set_prop_cycle(None)

                for column in df.columns:
                    plt.bar(
                        i,
                        df[column][i],
                        bottom=bottom,
                        color=get_color(column, df_materials[column][i]),
                        width=1,
                        align="edge",
                        label=column,
                    )
                    bottom += df[column][i]

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
            plt.ylabel("Longitudinal Calorimeter Composition [cm]")
            plt.xlabel("Iteration")
            plt.xlim(0, len(df))
            plt.text(
                0.01, 0.99,
                "Detector Optimization",
                transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left'
            )
            plt.text(
                0.01, 0.93,
                "Stacked Calorimeter, Run 1\nParticles: " + r"$\pi^+\gamma, E=[1, 20]$" + " GeV",
                transform=ax.transAxes, fontsize=12, va='top', ha='left'
            )
            cbar_absorber = plt.cm.ScalarMappable(cmap=absorber_cmap)
            cbar_absorber.set_array([])
            cbar1 = plt.colorbar(cbar_absorber, ax=ax, fraction=0.05)
            cbar1.ax.invert_yaxis()  # Invert so Fe is high and Pb is low
            cbar1.ax.set_yticks([0, 1])
            cbar1.ax.set_yticklabels(['Pb', 'Fe'], rotation=90, va='center')  # Custom ticks

            # Add colormap for "scintillator" with labels "Polystyrene" and "PbWO4"
            cbar_scintillator = plt.cm.ScalarMappable(cmap=scintillator_cmap)
            cbar_scintillator.set_array([])
            cbar2 = plt.colorbar(cbar_scintillator, ax=ax, fraction=0.05, location='right')
            cbar2.ax.invert_yaxis()
            cbar2.ax.set_yticks([0, 1])
            cbar2.ax.set_yticklabels(['PbWO4', 'Polystyrene'], rotation=90, va='center')  # Custom ticks

            plt.savefig(os.path.join(self.results_dir, "plots/calorimeter_sideview"))
            plt.close()

        def plot_energy_resolution_evolution() -> None:
            dirs = glob.glob(f"{self.results_dir}/task_outputs/iteration=*/validation=False/reco_output_df")
            e_rec_array = np.full(len(dirs), 0.0)
            bins = np.linspace(-20, 20, 80 + 1)

            for file_name in dirs:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
                df = pd.read_parquet(file_name)
                e_rec: pd.Series = df["Reconstructed"]["true_energy"] - df["Targets"]["true_energy"]
                e_rec_binned, *_ = plt.hist(
                    e_rec,
                    bins=bins
                )
                e_rec_array[iteration] = aido.Plotting.fwhm(bins, e_rec_binned)[0]

            plt.close()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = add_plot_header(ax)
            plt.plot(e_rec_array)
            plt.xlabel("Iteration")
            plt.ylabel("Energy Resolution [GeV]")
            plt.xlim(0, len(e_rec_array))
            plt.ylim(bottom=0)
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_evolution"))
            plt.close()

        plot_energy_resolution_all()
        plot_energy_resolution_first_and_last()
        plot_energy_resolution_evolution()
        plot_calorimeter_sideview()
        plt.close("all")
        return None


if __name__ == "__main__":

    sigma = 1.5
    min_value = 0.001
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 35.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_0", 5.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("material_absorber_0", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[25, 4.166]),
        aido.SimulationParameter(
            "material_scintillator_0",
            "G4_POLYSTYRENE",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01]
        ),
        aido.SimulationParameter("thickness_absorber_1", 20.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_1", 10.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("material_absorber_1", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[25, 4.166]),
        aido.SimulationParameter(
            "material_scintillator_1",
            "G4_PbWO4",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01]
        ),
        aido.SimulationParameter("thickness_absorber_2", 20.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("thickness_scintillator_2", 2.0, min_value=min_value, sigma=sigma),
        aido.SimulationParameter("material_absorber_2", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[25, 4.166]),
        aido.SimulationParameter(
            "material_scintillator_2",
            "G4_PbWO4",
            discrete_values=["G4_PbWO4", "G4_POLYSTYRENE"],
            cost=[2500.0, 0.01]
        ),
        aido.SimulationParameter("num_events", 400, optimizable=False),
        aido.SimulationParameter("max_length", 200, optimizable=False),
        aido.SimulationParameter("max_cost", 120_000, optimizable=False),
        aido.SimulationParameter("full_calorimeter", True, optimizable=False)
    ])

    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=20,
        max_iterations=100,
        threads=20,
        results_dir="/work/kschmidt/aido/results_full_calorimeter/results_20250116",
        description="""
            Full Calorimeter with cost and length constraints.
            With discrete parameters
        """
    )
    os.system("rm *.root")
