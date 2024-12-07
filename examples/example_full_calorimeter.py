import glob
import os
import re
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class

import aido


class UIFullCalorimeter(AIDOUserInterfaceExample):

    material_scaling_factor = 1.0

    @classmethod
    def constraints(
            self,
            parameter_dict: aido.SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> torch.Tensor:

        def sigmoid(x: float):
            return 1.0 / (1.0 + torch.exp(-x))

        detector_length = 0.0
        cost = 0.0
        materials = {
            "absorber": {"costly": 25.0, "cheap": 4.166},
            "scintillator": {"costly": 2500.0, "cheap": 0.01}
        }

        for i in range(3):
            for name in ["absorber", "scintillator"]:
                layer_thickness = parameter_dict_as_tensor[f"thickness_{name}_{i}"]
                layer_material = parameter_dict_as_tensor[f"material_{name}_{i}"]

                sigmoid_scintillator = sigmoid(self.material_scaling_factor * layer_material)
                layer_cost_per_unit = (
                    sigmoid_scintillator * materials[name]["costly"]
                    + (1.0 - sigmoid_scintillator) * materials[name]["cheap"]
                )
                cost += layer_thickness * layer_cost_per_unit
                detector_length += layer_thickness

        self.material_scaling_factor += 0.08
        max_loss = parameter_dict["max_length"].current_value
        max_cost = parameter_dict["max_cost"].current_value
        detector_length_penalty = torch.mean(10.0 * torch.nn.ReLU()(detector_length - max_loss)**2)
        max_cost_penalty = torch.mean(2.0 / max_cost * torch.nn.ReLU()(cost - max_cost)**2)
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

            def get_dfs():
                df_list = []
                parameter_dir = os.path.join(self.results_dir, "parameters/")

                for file_name in os.listdir(parameter_dir):
                    param_dict = aido.SimulationParameterDictionary.from_json(parameter_dir + file_name)
                    df_list.append(pd.DataFrame(
                        param_dict.get_current_values(format="dict", types="continuous"),
                        index=[param_dict.iteration],
                    ))

                df_combined: pd.DataFrame = pd.concat(df_list, axis=0).sort_index()
                
                df_thickness = df_combined.loc[:, df_combined.columns.str.startswith("thickness")]
                df_materials = df_combined.loc[:, df_combined.columns.str.startswith("material")]
                return (df_thickness, df_materials)

            fig, ax = plt.subplots(figsize=(10, 7))

            def get_color(label: str, value: float):
                if "absorber" in label:
                    return "black" if value >= 0 else "grey"
                if "scintillator" in label:
                    return "pink" if value >= 0 else "yellow"
                else:
                    return "white"
            
            df_thickness, df_materials = get_dfs()

            for i in df_thickness.index:
                bottom = 0
                plt.gca().set_prop_cycle(None)

                for j, column in enumerate(df_thickness):
                    plt.bar(
                        i,
                        df_thickness[column][i],
                        bottom=bottom,
                        color=get_color(column, df_materials[df_materials.columns[j]][i]),
                        width=1,
                        align="edge",
                        label=column,
                    )
                    bottom += df_thickness[column][i]

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=10)
            plt.ylabel("Longitudinal Calorimeter Composition [cm]")
            plt.xlabel("Iteration")
            plt.xlim(0, len(df_thickness))
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

    aido.SimulationParameter.set_config("sigma", 1.5)
    min_value = 0.001
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", np.random.uniform(0.1, 50), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_0", np.random.uniform(20, 35), min_value=min_value),
        aido.SimulationParameter("material_absorber_0", np.random.uniform(-1, 1)),
        aido.SimulationParameter("material_scintillator_0", np.random.uniform(-1, 1)),
        aido.SimulationParameter("thickness_absorber_1", np.random.uniform(0.1, 5), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_1", np.random.uniform(0.1, 35), min_value=min_value),
        aido.SimulationParameter("material_absorber_1", np.random.uniform(-1, 1)),
        aido.SimulationParameter("material_scintillator_1", np.random.uniform(-1, 1)),
        aido.SimulationParameter("thickness_absorber_2", np.random.uniform(0.1, 50), min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_2", np.random.uniform(0.1, 10), min_value=min_value),
        aido.SimulationParameter("material_absorber_2", np.random.uniform(-1, 1)),
        aido.SimulationParameter("material_scintillator_2", np.random.uniform(-1, 1)),
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
        threads=10,
        results_dir="/work/kschmidt/aido/results_material_choice/results_20241206_1",
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
        """
    )
    os.system("rm *.root")
