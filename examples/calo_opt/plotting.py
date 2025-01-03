import glob
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aido

matplotlib.use("agg")


class CaloOptPlotting:

    def __init__(self, results_dir: str | os.PathLike) -> None:
        self.results_dir = results_dir

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
            ) -> None:
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
    
    def plot_energy_resolution_all(self) -> None:
        dirs = glob.glob(f"{self.results_dir}/task_outputs/iteration=*/validation=False/reco_output_df")
        cmap = plt.get_cmap('coolwarm', len(dirs))
        fig, ax = plt.subplots(figsize=(8, 6))
        bins = np.linspace(-20, 20, 80 + 1)

        for file_name in dirs:
            iteration = int(re.search(r"iteration=(\d+)", file_name).group(1))
            self.plot_energy_resolution_single(
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

    def plot_energy_resolution_first_and_last(self) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax = self.add_plot_header(ax)
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

    def plot_calorimeter_sideview(self) -> None:

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

    def plot_energy_resolution_evolution(self) -> None:
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
        ax = self.add_plot_header(ax)
        plt.plot(e_rec_array)
        plt.xlabel("Iteration")
        plt.ylabel("Energy Resolution [GeV]")
        plt.xlim(0, len(e_rec_array))
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(self.results_dir, "plots/energy_resolution_evolution"))
        plt.close()
