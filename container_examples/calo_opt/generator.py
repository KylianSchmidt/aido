

import numpy as np
import pandas as pd
import time
import multiprocessing
import os
import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from  G4Calo import G4System, GeometryDescriptor

        
class ReconstructionDataset(Dataset):
    def __init__(
            self,
            input_df: pd.DataFrame,
            ):
        """Convert the files from the simulation to simple lists.

        Args:
            parameter_dict: Instance of Parameter Dictionary.
            simulation_df: Instance of pd.DataFrame
            df["Inputs"]_keys (list of keys in df): Keys of input features to be used by the model.
            target_features_keys (list of keys in df): Keys of target features of the reconstruction model.
            context_information (list of keys in df): (Optional) Keys of additional information for each 
                event.

        Returns:
            torch.DataSet instance
        """
        self.df = input_df

        self.shape = (
            self.df["Parameters"].shape[1],
            self.df["Inputs"].shape[1],
            self.df["Targets"].shape[1],
            self.df["Context"].shape[1]
        )

        # Normalize
        self.means = [
            np.mean(self.df["Parameters"], axis=0),
            np.mean(self.df["Inputs"], axis=0),
            np.mean(self.df["Targets"], axis=0),
            np.mean(self.df["Context"], axis=0)
        ]
        self.stds = [
            np.std(self.df["Parameters"], axis=0) + 1e-3,
            np.std(self.df["Inputs"], axis=0) + 1e-3,
            np.std(self.df["Targets"], axis=0) + 1e-3,
            np.std(self.df["Context"], axis=0) + 1e-3
        ]

        self.df["Parameters"] = (self.df["Parameters"] - self.means[0]) / self.stds[0]
        self.df["Inputs"] = (self.df["Inputs"] - self.means[1]) / self.stds[1]

        # A normalised target is important for the surrogate to work given the scheduling we have here
        self.df["Targets"] = (self.df["Targets"] - self.means[2]) / self.stds[2]
        self.df["Context"] = (self.df["Context"] - self.means[3]) / self.stds[3]

        self.c_means = [torch.tensor(a).to('cuda') for a in self.means]
        self.c_stds = [torch.tensor(a).to('cuda') for a in self.stds]

        self.filter_infs_and_nans()

    def filter_infs_and_nans(self):
        '''
        Removes all events that contain infs or nans.
        '''
        mask = np.ones(len(self.df["Inputs"]), dtype=bool)

        for i in range(len(self.df["Inputs"])):
            if np.any(np.isinf(self.df["Inputs"][i])) or np.any(np.isnan(self.df["Inputs"][i])):
                mask[i] = False
    
        self.df["Inputs"] = self.df["Inputs"][mask]
        self.df["Parameters"] = self.df["Parameters"][mask]
        self.df["Targets"] = self.df["Targets"][mask]
        self.df["Context"] = self.df["Context"][mask]

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
        return len(self.df["Inputs"])
    
    def __getitem__(self, idx):
        return self.df["Parameters"][idx], self.df["Inputs"][idx], self.df["Targets"][idx]
    