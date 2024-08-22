import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from .simulation_helpers import SimulationParameterDictionary


class AIDOPlotting:

    def parameter_evolution(
            fig_savepath: str | os.PathLike | None = "./results/plots/parameter_evolution",
            parameter_dir: str | os.PathLike = "./results/parameters"
            ) -> Tuple[pd.DataFrame, np.ndarray]:
        """ Plots the evolution of all simulation parameters along with their respective "sigma".

        Args:
            fig_savepath (str | os.PathLike, optional): The file path to save the figure.
                Defaults to "./results/plots/parameter_evolution".
            parameter_dir (str | os.PathLike, optional): The directory path where the SimulationParameterDictionaries
                are stored (.json files). Defaults to "./results/parameters".
        Returns:
            Tuple(pd.DataFrame, np.ndarray): A Tuple containing the DataFrame with all parameters provided by the
                optimizer after each iteration, and the simulation sampling standard deviation (2D array).
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

        if fig_savepath is not None:
            plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(df, label=df.columns)

            for i, col in enumerate(df.columns):
                plt.fill_between(df[col].index, df[col] - sigma[:, i], df[col] + sigma[:, i], alpha=0.5)

            plt.legend()
            plt.xlabel("Iteration", loc="right")
            plt.ylabel("Parameter Value", loc="top")
            plt.savefig(fig_savepath)

        return df, sigma

    def optimizer_loss(
            fig_savepath: str | os.PathLike | None = "./results/plots/optimizer_loss",
            optimizer_loss_dir: str | os.PathLike = "./results/loss/optimizer"
            ) -> None:
        """
        Plot the optimizer loss over epochs and save the figure if `fig_savepath` is provided.
        Args:
            fig_savepath (str | os.PathLike | None): Path to save the figure. If None, the figure will not be saved.
            optimizer_loss_dir (str | os.PathLike): Directory containing the optimizer loss files.
        Returns:
            df_loss (pd.DataFrame): DataFrame with the optimizer loss at each iteration
        """
        df_loss_list = []

        for file_name in sorted(os.listdir(optimizer_loss_dir)):
            df_loss_list.append(pd.read_csv(optimizer_loss_dir + "/" + file_name, index_col=0))

        df_loss: pd.DataFrame = pd.concat(df_loss_list, axis=1)

        if fig_savepath is not None:
            plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(
                np.linspace(0, df_loss.shape[1], df_loss.shape[0] * df_loss.shape[1]),
                df_loss.to_numpy().flatten("F"),
                c="k",
                label="optimizer_loss"
            )
            plt.xlim(0, df_loss.shape[1])
            plt.xlabel("Epoch", loc="right")
            plt.ylabel("Loss", loc="top")
            plt.yscale("linear")
            plt.legend()
            plt.savefig(fig_savepath)

        return df_loss

    def simulation_samples(
            fig_savepath: str | os.PathLike | None = "./results/plots/simulation_samples",
            sampled_param_dict_filepath: str | os.PathLike = "./results/task_outputs/iteration=*/"
            ) -> Tuple[pd.DataFrame, np.ndarray]:
        """ Generate a DataFrame of simulation parameters and their values for each iteration and task.
        Args:
            fig_savepath (str | os.PathLike | None, optional): Path to save the generated plot.
                Defaults to "./results/plots/simulation_samples".
            sampled_param_dict_filepath (str | os.PathLike, optional): Path to the sampled parameter dictionary files.
                Defaults to "./results/task_outputs/simulation_task*".
        Returns:
            Tuple(pd.DataFrame, np.ndarray): A tuple containing the DataFrame of simulation parameters and a
                numpy array of sigma values.
        
        TODO Check for the files in a dynamic way in case b2luigi changes the names of the directories
        due to changes in the b2luigi.Parameters of the SimulationTasks.
        """
        df_list = []

        for iteration_dir in glob.glob(sampled_param_dict_filepath):
            
            for simulation_dir in glob.glob(iteration_dir + "/simulation_task_id=*"):

                df = SimulationParameterDictionary.from_json(simulation_dir + "/param_dict.json").to_df(1)
                df["Iteration"] = int(re.search(r"/iteration=(\d+)/", iteration_dir).group(1))
                df["Task_ID"] = int(re.search(r"task_id=(\d+)", simulation_dir).group(1))
                df_list.append(df)

        df_params = pd.concat(df_list)
        df_params = df_params.sort_values(["Iteration", "Task_ID"]).reset_index(drop=True)

        if fig_savepath is not None:
            df_optim, sigma = AIDOPlotting.parameter_evolution(None)

            plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(df_optim, label=df_optim.columns)

            for i, col in enumerate(df_optim.columns):
                plt.fill_between(df_optim[col].index, df_optim[col] - sigma[:, i], df_optim[col] + sigma[:, i], alpha=0.5)

            plt.gca().set_prop_cycle(None)

            for i, col in enumerate(df_params.columns.drop(["Iteration", "Task_ID"])):
                plt.scatter(df_params["Iteration"], df_params[col].values, marker="+", s=100)

            plt.xlabel("Iteration", loc="right")
            plt.ylabel("Parameter Value", loc="top")
            plt.legend()
            plt.savefig(fig_savepath)

        return df_params, sigma

