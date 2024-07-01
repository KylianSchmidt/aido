

import numpy as np
import pandas as pd
import time
import multiprocessing
import os
import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from G4Calo import G4System, GeometryDescriptor


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

        # Normalize
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

        # A normalised target is important for the surrogate to work given the scheduling we have here
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
