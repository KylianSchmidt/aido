import glob
import os
import pathlib
import re
from typing import Iterable

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aido

matplotlib.use("agg")


class CaloOptPlotting:

    def __init__(self, results_dir: str | os.PathLike) -> None:
        self.results_dir = results_dir
        self.reco_output_paths: str = glob.glob(
            f"{results_dir}/task_outputs/iteration=*/validation=False/reco_output_df"
        )

    @staticmethod
    def mplstyle() -> None:
        plt.style.use(pathlib.Path(__file__).parent / "aido.mplstyle")

    @classmethod
    def add_plot_header(cls, ax: plt.Axes) -> plt.Axes:
        plt.text(
            0.0, 1.06,
            "AIDO",
            transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left'
        )
        plt.text(
            0.125, 1.06,
            "Detector Optimization",
            transform=ax.transAxes, fontsize=14, style='italic', va='top', ha='left'
        )
        plt.text(
            0.015, 0.98,
            "Sampling Calorimeter\n"
            "50% photons and 50% pions\n"
            r"$20 \times 400$" + " MC Events / Iteration\n"
            r"$E_\text{true}=[1, 20]$" + " GeV",
            transform=ax.transAxes, va='top', ha='left'
        )  # Adjust the text as fitting
        return ax

    def plot(self, parameter_dict: aido.SimulationParameterDictionary | None = None) -> None:

        def plot_reco_loss(
                iteration: int,
                file_name: str | os.PathLike,
                bins: np.ndarray,
                color=None,
                label: str | None = None,
                ):
            df = pd.read_parquet(file_name)[:400]
            e_rec: pd.Series = df["Loss"]
            plt.hist(
                e_rec,
                bins=bins,
                color=color,
                histtype="step",
                label=label,
                linewidth=1,
                zorder=iteration
            )

        def plot_reco_loss_all() -> None:
            sampled_iterations = [0, 10, 20, 200]
            cmap = plt.get_cmap('coolwarm', len(sampled_iterations))
            fig, ax = plt.subplots()
            bins = np.linspace(0, 10, 100 + 1)

            for file_name in self.reco_output_paths:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
                if iteration in sampled_iterations:
                    plot_reco_loss(
                        iteration=iteration,
                        file_name=file_name,
                        bins=bins,
                        color=cmap(iteration),
                        label=(f"Iteration {iteration:3d}"),
                    )

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles)))
            ax = self.add_plot_header(ax)
            ax.legend(handles, labels)
            plt.yscale("log")
            plt.xlim(bins[0], bins[-1])
            plt.ylim(1, 5000)
            plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
            plt.xlabel("Reconstruction Loss [GeV]")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/reco_loss_all"))
            plt.close()

        def plot_energy_resolution_single(
                iteration: int,
                file_name: str | os.PathLike,
                bins: np.ndarray,
                color=None,
                label: str | None = None,
                ):
            df = pd.read_parquet(file_name)[:400]
            e_rec: pd.Series = df["Reconstructed"]["true_energy"] - df["Targets"]["true_energy"]
            e_rec = e_rec / np.sqrt(df["Targets"]["true_energy"])
            plt.hist(
                e_rec,
                bins=bins,
                color=color,
                histtype="step",
                label=label,
                linewidth=1,
                zorder=iteration
            )

        def plot_energy_resolution_all() -> None:
            sampled_iterations = [0, 10, 20, 200]
            cmap = plt.get_cmap('coolwarm', len(sampled_iterations))
            fig, ax = plt.subplots()
            bins = np.linspace(-5, 5, 100 + 1)

            for file_name in self.reco_output_paths:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
                if iteration in sampled_iterations:
                    plot_energy_resolution_single(
                        iteration=iteration,
                        file_name=file_name,
                        bins=bins,
                        color=cmap(iteration),
                        label=(f"Iteration {iteration:3d}"),
                    )

            handles, labels = ax.get_legend_handles_labels()
            labels, handles = zip(*sorted(zip(labels, handles)))
            ax = self.add_plot_header(ax)
            ax.legend(handles, labels)
            plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV" + r"$^{1/2}$" + ")")
            plt.xlabel(r"$(E_\text{rec} - E_\text{true}) / E_\text{true}^{1/2}\, \left[ \text{GeV}^{1/2} \right]$")
            plt.xlim(bins[0], bins[-1])
            plt.ylim(1, 175)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_all"))
            plt.close()

        def plot_energy_resolution_first_and_last() -> None:
            fig, ax = plt.subplots()
            ax = self.add_plot_header(ax)
            cmap = plt.get_cmap('coolwarm', len(self.reco_output_paths))
            bins = np.linspace(-20, 20, 80 + 1)

            for file_name in self.reco_output_paths:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))

                if iteration == 0 or iteration == len(self.reco_output_paths) - 1:
                    df = pd.read_parquet(file_name)
                    e_rec = (df["Targets"] - df["Reconstructed"])
                    e_rec_binned, *_ = plt.hist(
                        e_rec,
                        bins=bins,
                        color=cmap(iteration),
                        histtype="step",
                        label=f"Iteration {iteration:3d}",
                    )
                    ax = aido.Plotting.FWHM(bins, e_rec_binned).add_to_axis(ax)

            plt.legend()
            plt.xlim(-10, 10)
            plt.xlabel(r"Energy Resolution $E_{\text{true}} - E_{\text{rec}}$ [GeV]")
            plt.ylabel(f"Counts {(bins[1] - bins[0]):.2f}")
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_first_and_last"))
            plt.close()

        def plot_calorimeter_sideview(
            add_legend: bool = None,
        ) -> None:
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

            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            ax = self.add_plot_header(ax)
            absorber_cmap = mcolors.LinearSegmentedColormap.from_list("blue_grey", ["blue", "grey"])
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
                        label=column.replace("_", " ").removeprefix("thickness ").capitalize(),
                    )
                    bottom += df[column][i]

            if add_legend:
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper right")

            plt.ylabel("Longitudinal Calorimeter Composition [cm]")
            plt.xlabel("Iteration")
            plt.xlim(0, len(df))
            plt.ylim(0, 220)
            ax = self.add_plot_header(ax)
            cbar_absorber = plt.cm.ScalarMappable(cmap=absorber_cmap)
            cbar_absorber.set_array([])
            cbar1 = plt.colorbar(
                cbar_absorber,
                ax=ax,
                fraction=0.04,
                location="right",
            )
            cbar1.ax.invert_yaxis()  # Invert so Fe is high and Pb is low
            cbar1.ax.set_yticks([0, 1], labels=['Pb', 'Fe'], rotation=90, va='center')  # Custom ticks

            # Add colormap for "scintillator" with labels "Polystyrene" and "PbWO4"
            cbar_scintillator = plt.cm.ScalarMappable(cmap=scintillator_cmap)
            cbar_scintillator.set_array([])
            cbar2 = plt.colorbar(
                cbar_scintillator,
                ax=ax,
                fraction=0.04,
                location="right",
            )
            cbar2.ax.invert_yaxis()
            cbar2.ax.set_yticks([0, 1], labels=['PbWO4', 'Polystyrene'], rotation=90, va='center')  # Custom ticks
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/calorimeter_sideview"), dpi=500)
            plt.close()

        def plot_energy_resolution_evolution(use_checkpoint: bool = False) -> None:
            e_rec_array = np.full(len(self.reco_output_paths), 0.0)
            e_loss_best_array = np.full(len(self.reco_output_paths), 0.0)

            for file_name in self.reco_output_paths:
                iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
                df = pd.read_parquet(file_name)[0:400]
                e_rec: pd.Series = df["Reconstructed"]["true_energy"] - df["Targets"]["true_energy"]
                e_rec = e_rec**2 / (df["Targets"]["true_energy"] + 1)
                e_rec_array[iteration] = np.mean(e_rec)
                e_loss_best_array[iteration] = np.mean(df["Loss"])

            plt.close()

            df_loss: pd.DataFrame = aido.Plotting.optimizer_loss(results_dir=self.results_dir)
            df_loss = df_loss[["Scaled Epoch", "Loss"]]
            df_loss = df_loss.set_index("Scaled Epoch")

            fig, ax = plt.subplots(figsize=(7, 5))
            ax = self.add_plot_header(ax)
            plt.plot(
                e_loss_best_array,
                label="Mean Reconstruction Loss " + r"($\mathcal{L}_\text{reco}$)",
            )
            plt.plot(
                df_loss.rolling(window=30).mean(),
                label="Optimizer Loss " + r"($\mathcal{L}'$)"
            )
            plt.legend()
            plt.xlabel("Iteration")
            plt.ylabel("Energy Resolution [GeV]")
            plt.xlim(0, len(e_rec_array))
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_evolution.pdf"))
            plt.close()
        
        def plot_constraints() -> None:
            def cost(parameter_dict: aido.SimulationParameterDictionary) -> float:
                cost = 0.0
                for i in range(3):
                    for name in ["absorber", "scintillator"]:
                        cost += (
                            parameter_dict[f"thickness_{name}_{i}"].current_value
                            * np.array(parameter_dict[f"material_{name}_{i}"].weighted_cost)
                        )
                return cost

            cost_list = []
            for i in range(len(self.reco_output_paths)):
                sim_param_dict = aido.SimulationParameterDictionary.from_json(
                    f"{self.results_dir}/parameters/param_dict_iter_{i}.json"
                )
                cost_item = cost(sim_param_dict)
                cost_list.append(cost_item)

            fig, ax = plt.subplots(figsize=(None, 2.5))
            plt.plot(cost_list)
            plt.xlabel("Iteration")
            plt.ylabel("Cost [EUR]")
            plt.xlim(0, len(cost_list) + 1)
            plt.ylim(0,)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, "plots/cost_constraints"))
            plt.close()

        if len(self.reco_output_paths) <= 1:
            print(f"No task outputs found in '{self.results_dir}/task_outputs/'. Skipping plotting.")
            return None

        plot_energy_resolution_all()
        plot_reco_loss_all()
        plot_energy_resolution_first_and_last()
        plot_energy_resolution_evolution()
        plot_calorimeter_sideview()
        plt.close("all")
        return None


if __name__ == "__main__":
    results_dir: str = ...

    plotter = CaloOptPlotting(results_dir)
    plotter.mplstyle()
    plotter.plot()
