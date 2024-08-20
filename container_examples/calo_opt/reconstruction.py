import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset


class ReconstructionDataset(Dataset):

    def __init__(self, input_df: pd.DataFrame):
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


class Reconstruction(torch.nn.Module):
    def __init__(self, num_parameters, num_input_features, num_target_features, num_context_features):
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

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_parameters + num_input_features, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, num_target_features),
        )

        # Placeholders for a simpler training loop
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.device = torch.device('cuda')

    def forward(self, detector_parameters, x):
        """ Concatenate the detector parameters and the input
        """
        x = torch.cat([detector_parameters, x], dim=1)
        return self.layers(x)
    
    def loss(self, y_pred, y):
        """ Remark: filters nans and in order to make it more stable.
        Uses an L2 loss with with 1/sqrt(E) weighting

        Alternatively: 'torch.nn.MSELoss()(y_pred, y)**(1/2)'
        """
        y = torch.where(torch.isnan(y_pred), torch.zeros_like(y) + 1., y)
        y = torch.where(torch.isinf(y_pred), torch.zeros_like(y) + 1., y)
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred), y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred), y_pred)

        return ((y_pred - y)**2 / (torch.abs(y) + 1.)).mean()
    
    def train_model(self, dataset: ReconstructionDataset, batch_size, n_epochs, lr):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer.lr = lr
        self.to(self.device)
        self.train()

        for epoch in range(n_epochs):
    
            for (detector_parameters, x, y) in train_loader:
                detector_parameters = detector_parameters.to(self.device)
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self(detector_parameters, x)
                loss = self.loss(dataset.unnormalise_target(y_pred), dataset.unnormalise_target(y))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
            print(f'Reco Epoch: {epoch} \tLoss: {loss.item():.8f}')

        self.eval()

    def apply_model_in_batches(self, dataset: ReconstructionDataset, batch_size):
        """ Apply the model in batches
        this is necessary because the model is too large to apply it to the whole dataset at once.
        The model is applied to the dataset in batches and the results are concatenated.
        (the batch size is a hyperparameter).
        """
        self.to(self.device)
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = None
        mean_loss = 0.

        for batch_idx, (detector_parameters, x, y) in enumerate(data_loader):
            detector_parameters = detector_parameters.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self(detector_parameters, x)

            loss = self.loss(dataset.unnormalise_target(y_pred), dataset.unnormalise_target(y))
            mean_loss += loss.item()

            if results is None:
                results = torch.zeros(len(dataset), y_pred.shape[1])
            
            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred

        mean_loss /= len(data_loader)

        return results, mean_loss
