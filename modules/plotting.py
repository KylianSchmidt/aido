import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .simulation_helpers import SimulationParameterDictionary


class AIDOPlotting:

    def parameter_evolution(
            fig_savepath: str | os.PathLike | None = "./results/plots/parameter_evolution",
            parameter_dir: str | os.PathLike = "./results/parameters"
            ):
        """ Plots the evolution of all simulation parameters along with their respective "sigma".

        Args:
            fig_savepath (str | os.PathLike, optional): The file path to save the figure.
                Defaults to "./results/plots/parameter_evolution".
            parameter_dir (str | os.PathLike, optional): The directory path where the SimulationParameterDictionaries
                are stored (.json files). Defaults to "./results/parameters".
        """
        df_list = []
        sigma_df_list = []

        for file_name in sorted(os.listdir(parameter_dir)):
            param_dict = SimulationParameterDictionary.from_json("./results/parameters/" + file_name)
            index = int(file_name.removeprefix("param_dict_iter_").removesuffix(".json"))

            df_list.append(pd.DataFrame(param_dict.get_current_values(format="dict"), index=[index]))

            sigma_df_list.append([np.diag(param_dict.covariance)])

        df: pd.DataFrame = pd.concat(df_list, axis=0).sort_index()
        sigma = np.concatenate(sigma_df_list, axis=0)

        plt.figure(figsize=(8, 6), dpi=400)
        plt.plot(df, label=df.columns)

        for i, col in enumerate(df.columns):
            plt.fill_between(df[col].index, df[col] - sigma[:, i], df[col] + sigma[:, i], alpha=0.5)

        plt.legend()
        plt.xlabel("Iteration", loc="right")
        plt.ylabel("Parameter Value", loc="top")

        if fig_savepath is not None:
            plt.savefig(fig_savepath)

    def optimizer_loss(
            fig_savepath: str | os.PathLike | None = "./results/plots/optimizer_loss",
            optimizer_loss_dir: str | os.PathLike = "./results/loss/optimizer"
            ) -> None:
        """
        Plot the optimizer loss over epochs and save the figure if `fig_savepath` is provided.
        Parameters:
        - fig_savepath (str | os.PathLike | None): Path to save the figure. If None, the figure will not be saved.
        - optimizer_loss_dir (str | os.PathLike): Directory containing the optimizer loss files.
        Returns:
        - None
        """
        df_loss_list = []

        for file_name in sorted(os.listdir(optimizer_loss_dir)):
            df_loss_list.append(pd.read_csv(optimizer_loss_dir + "/" + file_name, index_col=0))

        df_loss: pd.DataFrame = pd.concat(df_loss_list, axis=1)

        plt.figure(figsize=(8, 6), dpi=400)
        plt.plot(
            np.linspace(0, df_loss.shape[1], df_loss.shape[0] * df_loss.shape[1]),
            df_loss.to_numpy().flatten("F"),
            c="k",
            label="optimizer_loss"
        )
        plt.xlim(0, df_loss.shape[1])
        plt.xticks(np.arange(0, df_loss.shape[1] + 1))
        plt.xlabel("Epoch", loc="right")
        plt.ylabel("Loss", loc="top")
        plt.yscale("log")
        plt.legend()

        if fig_savepath is not None:
            plt.savefig(fig_savepath)
