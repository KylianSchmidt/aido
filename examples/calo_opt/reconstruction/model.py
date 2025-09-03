"""
Reconstruction model. Predicts the deposited energy in the calorimeter
"""
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ReconstructionDataset


class Reconstruction(torch.nn.Module):
    def __init__(
        self,
        num_parameters: int,
        num_input_features: int,
        num_target_features: int,
        num_context_features: int,
        initial_means: List[np.float32],
        initial_stds: List[np.float32],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
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

    @staticmethod
    def loss(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        L2 Loss with extra weighting

        Alternatives:
            
            1. torch.nn.MSELoss()(y_pred, y)**(1/2)
            2. (y_pred - y)**2 / (y**2 + 1)
            3. (y_pred - y)**2 / (torch.where(y > 1., y, torch.ones_like(y)))**2

        """
        y = torch.where(torch.isnan(y_pred), torch.zeros_like(y) + 1., y)
        y = torch.where(torch.isinf(y_pred), torch.zeros_like(y) + 1., y)
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred), y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred), y_pred)

        return (y_pred - y)**2 / (torch.abs(y) + 1)

    def train_model(
        self,
        dataset: ReconstructionDataset,
        batch_size: int,
        n_epochs: int,
        lr: float,
    ):
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
                    dataset.unnormalize_target(y),
                    dataset.unnormalize_target(y_pred)
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
                dataset.unnormalize_target(y),
                dataset.unnormalize_target(y_pred)
            )
            loss = loss_per_event.clone().mean()
            mean_loss += loss.item()

            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = dataset.unnormalize_target(y_pred.flatten())
            loss_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = loss_per_event.flatten()

        mean_loss /= len(data_loader)
        results = results.detach().cpu().numpy()
        loss_array = loss_array.detach().cpu().numpy()
        return results, loss_array, mean_loss
