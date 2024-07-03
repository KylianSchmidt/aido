import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):

    def __init__(self, input_df: pd.DataFrame):
        """Convert the files from the simulation to simple lists.

        Args:
            input_df: Instance of pd.DataFrame. Must contain as first level columns:
            ["Parameters", "Inputs", "Targets", "Context"]

        Returns:
            torch.DataSet instance
        """
        self.df = input_df
        self.parameters = self.df["Parameters"].to_numpy("float32")
        self.inputs = self.df["Inputs"].to_numpy("float32")
        self.targets = self.df["Targets"].to_numpy("float32")
        self.context = self.df["Context"].to_numpy("float32")

        self.shape = (
            self.parameters.shape[1],
            self.inputs.shape[1],
            self.targets.shape[1],
            self.context.shape[1]
        )
        self.means = [
            np.mean(self.parameters, axis=0),
            np.mean(self.inputs, axis=0),
            np.mean(self.targets, axis=0),
            np.mean(self.context, axis=0)
        ]
        self.stds = [
            np.std(self.parameters, axis=0) + 1e-3,
            np.std(self.inputs, axis=0) + 1e-3,
            np.std(self.targets, axis=0) + 1e-3,
            np.std(self.context, axis=0) + 1e-3
        ]

        self.parameters = (self.parameters - self.means[0]) / self.stds[0]
        self.inputs = (self.inputs - self.means[1]) / self.stds[1]
        self.targets = (self.targets - self.means[2]) / self.stds[2]
        self.context = (self.context - self.means[3]) / self.stds[3]

        self.c_means = [torch.tensor(a).to('cuda') for a in self.means]
        self.c_stds = [torch.tensor(a).to('cuda') for a in self.stds]
        self.df = self.filter_infs_and_nans(self.df)

    def filter_infs_and_nans(self, df: pd.DataFrame):
        '''
        Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def unnormalise_target(self, target):
        '''
        receives back the physically meaningful target from the normalised target
        '''
        return target * self.c_stds[2] + self.c_means[2]
    
    def normalise_target(self, target):
        '''
        normalises the target
        '''
        return (target - self.c_means[2]) / self.c_stds[2]
    
    def unnormalise_detector(self, detector):
        '''
        receives back the physically meaningful detector from the normalised detector
        '''
        return detector * self.c_stds[1] + self.c_means[1]
    
    def normalise_detector(self, detector):
        '''
        normalises the detector
        '''
        return (detector - self.c_means[1]) / self.c_stds[1]
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.parameters[idx], self.inputs[idx], self.targets[idx]


class SurrogateDataset(ReconstructionDataset):
    """This dataset requires an existing ReconstructionDataset instance. This dataset
    adds the output of the reconstruction model.

    Args:
        reco_dataset (ReconstructionDataset): An existing ReconstructionDataset instance.
        reconstructed_array (np.ndarray): The reconstructed array to be added to the dataset. Must not be normailized.

    Attributes:
        df (pd.DataFrame): The concatenated dataframe containing the original dataset and the reconstructed array.
        parameters (np.ndarray): The parameters from the original dataset.
        inputs (np.ndarray): The inputs from the original dataset.
        targets (np.ndarray): The targets from the original dataset.
        context (np.ndarray): The context from the original dataset.
        reconstructed (np.ndarray): The reconstructed array. Columns are identical to 'targets'.
        shape (tuple): The shape of the dataset: [parameters, inputs, targets, context, reconstructed].
        means (list): The means of the original dataset and the reconstructed array.
        stds (list): The standard deviations of the original dataset and the reconstructed array.
        c_means (list): The means converted to torch tensors and moved to the 'cuda' device.
        c_stds (list): The standard deviations converted to torch tensors and moved to the 'cuda' device.

    Methods:
        __getitem__(self, idx: int): [parameters, targets, context, reconstructed] at the given index

    """

    def __init__(
            self,
            reco_dataset: ReconstructionDataset,
            reconstructed_array: np.ndarray
            ):
        reconstructed_df = pd.DataFrame(reconstructed_array, columns=reco_dataset.df["Targets"].columns)
        reconstructed_df = pd.concat({"Reconstructed": reconstructed_df}, axis=1)
        self.df: pd.DataFrame = pd.concat([reco_dataset.df, reconstructed_df], axis=1)
        
        self.parameters = reco_dataset.parameters
        self.inputs = reco_dataset.inputs
        self.targets = reco_dataset.targets
        self.context = reco_dataset.context
        self.reconstructed = self.df["Reconstructed"].to_numpy("float32")

        self.shape = (
            self.parameters.shape[1],
            self.inputs.shape[1],
            self.targets.shape[1],
            self.context.shape[1],
            self.reconstructed.shape[1]
        )
        self.means = [
            np.mean(self.parameters, axis=0),
            np.mean(self.inputs, axis=0),
            np.mean(self.targets, axis=0),
            np.mean(self.context, axis=0),
            np.mean(self.reconstructed, axis=0)
        ]
        self.stds = [
            np.std(self.parameters, axis=0) + 1e-3,
            np.std(self.inputs, axis=0) + 1e-3,
            np.std(self.targets, axis=0) + 1e-3,
            np.std(self.context, axis=0) + 1e-3,
            np.std(self.reconstructed, axis=0) + 1e-3
        ]
        self.reconstructed = (self.reconstructed - self.means[4]) / self.stds[4]

        self.c_means = [torch.tensor(a).to('cuda') for a in self.means]
        self.c_stds = [torch.tensor(a).to('cuda') for a in self.stds]
        self.df = self.filter_infs_and_nans(self.df)

    def __getitem__(self, idx):
        return self.parameters[idx], self.targets[idx], self.context[idx], self.reconstructed[idx]
