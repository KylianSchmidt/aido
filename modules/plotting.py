import glob
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.simulation_helpers import SimulationParameterDictionary


class AIDOPlotting:

    @classmethod
    def plot(cls, plot_types: str | List[str] = "all", results_dir: str | os.PathLike = "./results/"):
        """
        Plot the evolution of variables of interest over the Optimization process.

        Args:
            plot_types (str | List[str], optional): The types of plots to be generated.
                It can be a string or a list of strings. If "all" is specified, it will
                generate all available plots. Available methods:

                    ["parameter_evolution", "optimizer_loss", "simulation_samples"]

        Returns:
            None

        TODO Clean up this class and do not repeat the reading of files all the time
        """
        if plot_types == "all":
            plot_types = ["optimizer_loss", "probability_evolution", "parameter_evolution", "simulation_samples"]

        if isinstance(plot_types, str):
            plot_types = [plot_types]

        for plot_type in plot_types:
            getattr(cls, plot_type)(results_dir=results_dir)

        print(f"AIDOPlotting: Saved all figures to {results_dir}")

    def parameter_evolution(
            fig_savepath: str | os.PathLike | None = "/plots/parameter_evolution",
            results_dir: str = "./results/",
            parameter_dir: str | os.PathLike = "/parameters/"
            ) -> Tuple[pd.DataFrame, np.ndarray]:
        """ Plots the evolution of all simulation parameters along with their respective "sigma".

        Args:
            fig_savepath (str | os.PathLike, optional): The file path to save the figure.
                Defaults to "<results_dir>/plots/parameter_evolution". If None, the figure will not be saved.
            results_dir (str | os.PathLike, optional): Results directory. Defaults to "./results/"
            parameter_dir (str | os.PathLike, optional): The directory path where the SimulationParameterDictionaries
                are stored (.json files). Defaults to "<results_dir>/parameters".
        Returns:
            Tuple(pd.DataFrame, np.ndarray): A Tuple containing the DataFrame with all parameters provided by the
                optimizer after each iteration, and the simulation sampling standard deviation (2D array).
        """
        fig_savepath = f"{results_dir}/{fig_savepath}"
        parameter_dir = f"{results_dir}/{parameter_dir}"

        df_list = []
        sigma_df_list = []

        for file_name in os.listdir(parameter_dir):
            param_dict = SimulationParameterDictionary.from_json(parameter_dir + file_name)
            df_list.append(pd.DataFrame(
                param_dict.get_current_values(format="dict", types="continuous"),
                index=[param_dict.iteration],
            ))
            sigma_df_list.append(np.diag(param_dict.covariance))

        df: pd.DataFrame = pd.concat(df_list, axis=0).sort_index()
        sigma = np.concatenate(sigma_df_list, axis=0)

        if fig_savepath is not None:
            plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(df, label=df.columns)

            for i, col in enumerate(df.columns):
                if np.any(sigma[i]):
                    plt.fill_between(df[col].index, df[col] - sigma[i], df[col] + sigma[i], alpha=0.5)

            plt.legend()
            plt.xlabel("Iteration", loc="right")
            plt.ylabel("Parameter Value", loc="top")
            plt.savefig(fig_savepath)
            plt.close()

        return df, sigma

    def optimizer_loss(
            fig_savepath: str | os.PathLike | None = "/plots/optimizer_loss",
            results_dir: str = "./results/",
            optimizer_loss_dir: str | os.PathLike = "/loss/optimizer"
            ) -> pd.DataFrame:
        """
        Plot the optimizer loss over epochs and save the figure if `fig_savepath` is provided.
        Args:
            fig_savepath (str | os.PathLike | None): Path to save the figure. If None, the figure will not be saved.
            results_dir (str | os.PathLike, optional): Results directory. Defaults to "./results/"
            optimizer_loss_dir (str | os.PathLike): Directory containing the optimizer loss files.
        Returns:
            df_loss (pd.DataFrame): DataFrame with the optimizer loss at each iteration
        """
        fig_savepath = f"{results_dir}/{fig_savepath}"
        optimizer_loss_dir = f"{results_dir}/{optimizer_loss_dir}"

        df_loss_list = []

        for file_name in glob.glob(f"{optimizer_loss_dir}/*"):
            df_i = pd.read_csv(file_name, names=["Epoch", "Loss"])
            df_i["Iteration"] = int(re.search(r"optimizer_loss_(\d+)", file_name).group(1))
            df_loss_list.append(df_i)

        df_loss: pd.DataFrame = pd.concat(df_loss_list).sort_values(["Iteration", "Epoch"])

        if fig_savepath is not None:
            plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(
                np.linspace(0, df_loss["Iteration"].to_numpy()[-1], len(df_loss)),
                df_loss["Loss"].to_numpy().flatten("F"),
                c="k",
                label="optimizer_loss"
            )
            plt.xlim(0, df_loss["Iteration"].to_numpy()[-1])
            plt.xlabel("Epoch", loc="right")
            plt.ylabel("Loss", loc="top")
            plt.yscale("linear")
            plt.legend()
            plt.savefig(fig_savepath)
            plt.close()

        return df_loss

    def simulation_samples(
            fig_savepath: str | os.PathLike | None = "/plots/simulation_samples",
            results_dir: str = "./results/",
            sampled_param_dict_filepath: str | os.PathLike = "/task_outputs/iteration=*/"
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
        fig_savepath = f"{results_dir}/{fig_savepath}"
        sampled_param_dict_filepath = f"{results_dir}/{sampled_param_dict_filepath}"

        df_list: List[pd.DataFrame] = []

        for iteration_dir in glob.glob(sampled_param_dict_filepath):
            
            for simulation_dir in glob.glob(iteration_dir + "/simulation_task_id=*"):

                df = SimulationParameterDictionary.from_json(
                    simulation_dir + "/param_dict.json"
                ).to_df(types="continuous")
                df["Iteration"] = int(re.search(r"/iteration=(\d+)/", iteration_dir).group(1))
                df["Task_ID"] = int(re.search(r"task_id=(\d+)", simulation_dir).group(1))
                df_list.append(df)

        df_params = pd.concat(df_list)
        df_params = df_params.sort_values(["Iteration", "Task_ID"]).reset_index(drop=True)

        if fig_savepath is not None:
            df_optim, sigma = AIDOPlotting.parameter_evolution(None, results_dir=results_dir)

            plt.figure(figsize=(8, 6), dpi=400)
            plt.plot(df_optim, label=df_optim.columns)

            for i, col in enumerate(df_optim.columns):
                if np.any(sigma[i]):
                    plt.fill_between(
                        df_optim[col].index,
                        df_optim[col] - sigma[i],
                        df_optim[col] + sigma[i],
                        alpha=0.5
                    )

            plt.gca().set_prop_cycle(None)

            for i, col in enumerate(df_params.columns.drop(["Iteration", "Task_ID"])):
                plt.scatter(df_params["Iteration"], df_params[col].values, marker="+", s=100)

            plt.xlabel("Iteration", loc="right")
            plt.ylabel("Parameter Value", loc="top")
            plt.legend()
            plt.savefig(fig_savepath)
            plt.close()

        return df_params, sigma

    def probability_evolution(
            fig_savepath: str | os.PathLike | None = "/plots/probability_evolution",
            results_dir: str = "./results/",
            parameter_dir: str | os.PathLike = "/parameters"
            ):

        def plot_probabilities(
                name: str,
                param_dicts_list: List[SimulationParameterDictionary],
                fig_savepath_absolute: str | os.PathLike,
                ):

            probabilities_over_iterations = []
            iterations = []

            for param_dict in param_dicts_list:
                discrete_values = param_dict[name].discrete_values
                iterations.append(param_dict.iteration)
                probabilities_over_iterations.append(param_dict[name].probabilities)

            probabilities_over_iterations = np.array(probabilities_over_iterations)[np.argsort(iterations)]
            iterations = np.array(iterations)[np.argsort(iterations)]

            fig, ax = plt.subplots(figsize=(8, 6))

            for i, discrete_value in enumerate(discrete_values):
                ax.bar(
                    iterations,
                    probabilities_over_iterations[:, i],
                    bottom=probabilities_over_iterations[:, :i].sum(axis=1),
                    label=discrete_value,
                    width=1,
                    align="edge"
                )

            ax.set_xlabel("Iteration")
            ax.set_ylabel("Probabilities")
            plt.legend()
            plt.xlim(iterations[0], iterations[-1])
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f"{fig_savepath_absolute}_{name}")
            plt.close()
            return None

        fig_savepath_absolute = f"{results_dir}/{fig_savepath}"
        parameter_dir_absolute = f"{results_dir}/{parameter_dir}/*"
        param_dicts_list: List[SimulationParameterDictionary] = []

        for param_dict_dir in glob.glob(parameter_dir_absolute):
            param_dicts_list.append(SimulationParameterDictionary.from_json(param_dict_dir))

        if not param_dicts_list:
            raise FileNotFoundError(f"No parameter dicts files could be found in {parameter_dir_absolute}")

        for parameter in param_dicts_list[0]:
            if parameter.discrete_values:
                plot_probabilities(parameter.name, param_dicts_list, fig_savepath_absolute)

        return None
