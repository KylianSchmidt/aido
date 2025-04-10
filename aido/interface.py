import os
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from aido.simulation_helpers import SimulationParameterDictionary


class _UserInterfaceBase(ABC):

    def __init__(self) -> None:
        self.results_dir: str | os.PathLike

    def create_surrogate_dataset(
            parameter_dict: SimulationParameterDictionary,
            user_reco_loss: pd.Series | pd.DataFrame | np.ndarray,
            user_context: pd.Series | pd.DataFrame | None = None,
            ):
        pass


class UserInterfaceBase(_UserInterfaceBase):

    @abstractmethod
    def simulate(self, parameter_dict_path: str, sim_output_path: str) -> None:
        """ This method must be implemented

        Starts the simulation process. We recommend starting a container and passing the arguments
        from the command line.

        Open the parameter dict using:

            parameter_dict = json.load(parameter_dict_path)

        Access its items by name and the key 'current_value':

            foo_value = parameter_dict["foo"]["current_value]

        The simulation should output exactly one file, which must be saved at 'sim_output_path'. You
        are free to choose the output format of the simulation (e.g. root file)

        Args:
            parameter_dict_path (str): The path to the parameter dictionary file.
            sim_output_path (str): The path to save the simulation output.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(
            self,
            parameter_dict_file_paths: List[str],
            simulation_file_paths: List[str],
            reco_input_path: str
            ) -> None:
        """ This method must be implemented

        This method must merge the parameter dicts and the simulation outputs into a single file.
        Its file path will be passed by the scheduler to the 'reconstruct' method as the first
        argument ('reco_input_path'). You are free to choose the file format of 'reco_input_path'.

        Args:
            parameter_dict_file_paths (List[str]): List of the simulation parameter dictionary paths
            simulation_file_paths (List[str]): List of the simulation output paths
            reco_input_path (str): Path for the merged file created by this method.
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, reco_input_path: str, reco_output_path: str, is_validation: bool = False) -> None:
        """ This method must be implemented

        Start your reconstruction algorithm here. We recommend using a container and starting the
        reconstruction from the command line.

        Args:
            reco_input_path (str): Path of the input file for your reconstruction process. It is the same
                path as the output of the 'merge' method.
            reco_output_path (str): Path of the output file generated by your reconstruction process. Since
                this file interfaces with the AIDO Optimizer, it must have a specific format detailed in the
                following.
            is_validation (bool): Useful to define a distinct behavior for regular reconstruction and for
                evaluation.

        Output file format (IMPORTANT):
            The output file generated by this method must be a parquet file of a pandas.DataFrame.
        """
        raise NotImplementedError

    def constraints(
            self,
            parameter_dict: SimulationParameterDictionary,
            parameter_dict_as_tensor: Dict[str, torch.Tensor]
            ) -> None | torch.Tensor:
        """ This method is optional

        Use this method to compute additional constraints such as cost or dimensions using pytorch. The resulting
        Tensor must be one-dimensional and include gradients.
        """
        return None

    def plot(self, parameter_dict: SimulationParameterDictionary) -> None:
        """ This method is optional

        Use this method to execute code after each iteration. This can be anything used to track the
        progress of the Optimization process.
        """
        return None

    def loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """ This method is optional

        Use this method to compute the loss of the internal Optimizer. This must be an equivalent
        implementation to your reconstruction loss.
        """
        raise NotImplementedError
