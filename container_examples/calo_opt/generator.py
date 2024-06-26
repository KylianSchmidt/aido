

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
            parameter_dict: Dict,
            simulation_df: pd.DataFrame,
            input_keys: List[str],
            target_keys: List[str],
            context_keys: List[str] = None,
            ):
        """Convert the files from the simulation to simple lists.

        Args:
            parameter_dict: Instance of Parameter Dictionary.
            simulation_df: Instance of pd.DataFrame
            input_features_keys (list of keys in df): Keys of input features to be used by the model.
            target_features_keys (list of keys in df): Keys of target features of the reconstruction model.
            context_information (list of keys in df): (Optional) Keys of additional information for each 
                event.

        Returns:
            torch.DataSet instance
        """
        
        self.input_features_keys = input_keys
        self.target_keys = target_keys
        self.context_keys = context_keys
        
        # 1. Simulation Parameter list
        if isinstance(parameter_dict, str):
            with open(parameter_dict, "r") as file:
                parameter_dict: Dict = json.load(file)

        simulation_parameter_list = []

        for parameter in parameter_dict.values():
            if parameter["optimizable"] is True:
                simulation_parameter_list.append(parameter["current_value"])

        self.simulation_parameters = np.array(simulation_parameter_list, dtype='float32')

        # 2. Simulation output (pd.DataFrame -> linear array)
        self.input_features = np.array(
            [simulation_df[par].to_numpy() for par in self.input_features_keys], dtype='float32'
        )
        self.input_features = np.swapaxes(self.input_features, 0, 1)
        self.input_features = np.reshape(self.input_features, (len(self.input_features), -1))

        # 3. Reconstruction targets
        self.target_array = simulation_df[self.target_keys].to_numpy(dtype='float32')

        # 4. Context information from simulation
        self.context_array = simulation_df[self.context_keys].to_numpy(dtype='float32')

        # Reshape parameters to (N, num_parameters)
        self.simulation_parameters = np.repeat([self.simulation_parameters], len(self.target_array), axis=0)

        self.shape = (
            self.simulation_parameters.shape[1],
            self.input_features.shape[1],
            self.target_array.shape[1],
            self.context_array.shape[1]
        )

        # Normalize
        self.means = [
            np.mean(self.simulation_parameters, axis=0),
            np.mean(self.input_features, axis=0),
            np.mean(self.target_array, axis=0),
            np.mean(self.context_array, axis=0)
        ]
        self.stds = [
            np.std(self.simulation_parameters, axis=0) + 1e-3,
            np.std(self.input_features, axis=0) + 1e-3,
            np.std(self.target_array, axis=0) + 1e-3,
            np.std(self.context_array, axis=0) + 1e-3
        ]

        self.simulation_parameters = (self.simulation_parameters - self.means[0]) / self.stds[0]
        self.input_features = (self.input_features - self.means[1]) / self.stds[1]

        # A normalised target is important for the surrogate to work given the scheduling we have here
        self.target_array = (self.target_array - self.means[2]) / self.stds[2]
        self.context_array = (self.context_array - self.means[3]) / self.stds[3]

        self.c_means = [torch.tensor(a).to('cuda') for a in self.means]
        self.c_stds = [torch.tensor(a).to('cuda') for a in self.stds]

        self.filter_infs_and_nans()

    def filter_infs_and_nans(self):
        '''
        Removes all events that contain infs or nans.
        '''
        mask = np.ones(len(self.input_features), dtype=bool)

        for i in range(len(self.input_features)):
            if np.any(np.isinf(self.input_features[i])) or np.any(np.isnan(self.input_features[i])):
                mask[i] = False
    
        self.input_features = self.input_features[mask]
        self.simulation_parameters = self.simulation_parameters[mask]
        self.target_array = self.target_array[mask]
        self.context_array = self.context_array[mask]

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
        return len(self.input_features)
    
    def __getitem__(self, idx):
        return self.simulation_parameters[idx], self.input_features[idx], self.target_array[idx]
    