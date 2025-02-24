from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

""" Reconstruction Dataset and Model

Written for python 3.9
"""


class ReconstructionDataset(Dataset):

    def __init__(
            self,
            input_df: pd.DataFrame,
            means: Optional[List[np.float32]] = None,
            stds: Optional[List[np.float32]] = None
            ):
        """Convert the files from the simulation to simple lists.

        Args:
        ----
            input_df: Instance of pd.DataFrame. Must contain as first level columns:
            ["Parameters", "Inputs", "Targets", "Context"]

        Returns:
        -------
            torch.DataSet instance
        """
        self.df = input_df
        self.df = self.filter_infs_and_nans(self.df)
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
        if means is None:
            self.means = [
                np.mean(self.parameters, axis=0),
                np.mean(self.inputs, axis=0),
                np.mean(self.targets, axis=0),
                np.mean(self.context, axis=0)
            ]
        else:
            self.means = means

        if stds is None:
            self.stds = [
                np.std(self.parameters, axis=0) + 1e-10,
                np.std(self.inputs, axis=0) + 1e-10,
                np.std(self.targets, axis=0) + 1e-10,
                np.std(self.context, axis=0) + 1e-10
            ]
        else:
            self.stds = stds

        self.inputs = (self.inputs - self.means[1]) / self.stds[1]
        self.targets = (self.targets - self.means[2]) / self.stds[2]
        self.context = (self.context - self.means[3]) / self.stds[3]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_means = [torch.tensor(a).to(dev) for a in self.means]
        self.c_stds = [torch.tensor(a).to(dev) for a in self.stds]

    def filter_infs_and_nans(self, df: pd.DataFrame):
        '''
        Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df
    
    def filter_empty_events(self, df: pd.DataFrame):
        df = df[df["Inputs"]["sensor_energy_0"] > 0.0]
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def unnormalise_target(self, target: torch.Tensor):
        '''
        receives back the physically meaningful target from the normalised target
        '''
        return target * self.c_stds[2] + self.c_means[2]
    
    def normalise_target(self, target: torch.Tensor):
        '''
        normalises the target
        '''
        return (target - self.c_means[2]) / self.c_stds[2]
    
    def unnormalise_detector(self, detector: torch.Tensor):
        '''
        receives back the physically meaningful detector from the normalised detector
        '''
        return detector * self.c_stds[1] + self.c_means[1]
    
    def normalise_detector(self, detector: torch.Tensor):
        '''
        normalises the detector
        '''
        return (detector - self.c_means[1]) / self.c_stds[1]
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int):
        return self.parameters[idx], self.inputs[idx], self.targets[idx]


class Reconstruction(torch.nn.Module):
    def __init__(
            self,
            num_parameters: int,
            num_input_features: int,
            num_target_features: int,
            num_context_features: int,
            initial_means: List[np.float32],
            initial_stds: List[np.float32],
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
            ):
        """Initialize the shape of the model.

        Args:
            num_parameters (int): The number of detector parameters (length of the list of simulation parameters).
            num_input_features (int): The number of input features (sensors and other simulation outputs).
            num_target_features (int): The number of target features (quantities which are representatives
                of the detector's capabilities).
        """
        super().__init__()

        self.n_parameters = num_parameters
        self.n_input_features = num_input_features
        self.n_target_features = num_target_features
        self.n_context_features = num_context_features
        self.means = initial_means
        self.stds = initial_stds
        self.preprocessing_layers = torch.nn.Sequential(
            torch.nn.Linear(num_parameters, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, num_input_features),
            torch.nn.ReLU()
        )
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_parameters + num_input_features, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.Linear(100, num_target_features),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.device = torch.device(device)

    def forward(self, parameters, x) -> torch.Tensor:
        """ Concatenate the detector parameters and the input
        """
        x = torch.multiply(self.preprocessing_layers(parameters), x)
        x = torch.cat([parameters, x], dim=1)
        return self.layers(x)

    def loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """ Remark: filters nans and in order to make it more stable.
        Uses an L2 loss with with 1/sqrt(E) weighting

        Alternatively: 'torch.nn.MSELoss()(y_pred, y)**(1/2)'
        """
        y_den = torch.where(y > 1., y, torch.ones_like(y))
        return (y_pred - y)**2 / y_den**2

    def train_model(self, dataset: ReconstructionDataset, batch_size: int, n_epochs: int, lr: float):
        print(f"Reconstruction Training: {lr=}, {batch_size=}")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.to(self.device)
        self.train()

        for epoch in range(n_epochs):
    
            for batch_idx, (detector_parameters, x, y) in enumerate(train_loader):
                detector_parameters: torch.Tensor = detector_parameters.to(self.device)
                x: torch.Tensor = x.to(self.device)
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(detector_parameters, x)
                loss_per_event = self.loss(
                    dataset.unnormalise_target(y),
                    dataset.unnormalise_target(y_pred)
                )
                loss = loss_per_event.clone().mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Reco Epoch: {epoch} \tLoss: {loss.item():.8f}")

        self.eval()

    def apply_model_in_batches(
            self,
            dataset: ReconstructionDataset,
            batch_size: int,
            ) -> Tuple[np.ndarray, np.ndarray, float]:
        """ Apply the model in batches this is necessary because the model is too large to apply it to the
        whole dataset at once. The model is applied to the dataset in batches and the results are concatenated
        (the batch size is a hyperparameter).
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros(len(dataset))
        loss_array = torch.zeros(len(dataset))
        mean_loss = 0.

        self.to(self.device)
        self.eval()

        for batch_idx, (detector_parameters, x, y) in enumerate(data_loader):
            detector_parameters = detector_parameters.to(self.device)
            x: torch.Tensor = x.to(self.device)
            y: torch.Tensor = y.to(self.device)
            y_pred: torch.Tensor = self(detector_parameters, x)

            loss_per_event = self.loss(
                dataset.unnormalise_target(y),
                dataset.unnormalise_target(y_pred)
            )
            loss = loss_per_event.clone().mean()
            mean_loss += loss.item()

            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = dataset.unnormalise_target(y_pred.flatten())
            loss_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = loss_per_event.flatten()

        mean_loss /= len(data_loader)
        results = results.detach().cpu().numpy()
        loss_array = loss_array.detach().cpu().numpy()
        return results, loss_array, mean_loss
