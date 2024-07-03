""" Contains the reconstrction module
neural network model for reconstruction of energy from detector depostits and parameters
contains also the training loop
the dataset is provided by the Generator as a pandas dataframe
Data loader etc are in here, too
"""
import sys
from typing import Dict, List, Union
import torch
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
from generator import ReconstructionDataset, SurrogateDataset
from surrogate import Surrogate


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


def pre_train(model: Union[Reconstruction, Surrogate], dataset: Dataset, n_epochs: int):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalised ones, 
    but also requires adjustments to the surrogate dataset
    """
    model.to('cuda')

    print('pre-training 0')
    model.train_model(dataset, batch_size=256, n_epochs=10, lr=0.03)

    print('pre-training 1')
    model.train_model(dataset, batch_size=256, n_epochs=n_epochs, lr=0.01)

    print('pre-training 2')
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print('pre-training 3')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)

    model.apply_model_in_batches(reco_dataset, batch_size=128)
    model.to('cpu')


input_df_path = sys.argv[1]
output_path = sys.argv[2]

simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)
print("DEBUG simulation df\n", simulation_df)

reco_dataset = ReconstructionDataset(simulation_df)
print("RECO Shape of model inputs:", reco_dataset.shape)

reco_model = Reconstruction(*reco_dataset.shape)

n_epochs_pre = 3
n_epochs_main = 10

pre_train(reco_model, reco_dataset, n_epochs_pre)

for i in range(3):
    # Reconstruction:
    reco_model.to('cuda')
    reco_model.train_model(reco_dataset, batch_size=256, n_epochs=n_epochs_main // 4, lr=0.003)
    reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
    reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
    reco_result, reco_loss = reco_model.apply_model_in_batches(reco_dataset, batch_size=128)
    print(f"RECO Block {i} DONE, loss={reco_loss:.8f}")

    # Reconstructed array unnormalized:
    reco_result = reco_result.detach().cpu().numpy()
    reco_result = reco_result * reco_dataset.stds[2] + reco_dataset.means[2]

    # Surrogate:
    print("Surrogate training")
    surrogate_dataset = SurrogateDataset(reco_dataset, reco_result)
    print("DEBUG Get item from Surrogate DataSet, idx=0", surrogate_dataset[0])
    surrogate_model = Surrogate(*surrogate_dataset.shape)

    surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
    surrogate_loss = surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0003)

    best_surrogate_loss = 1e10

    while surrogate_loss < 4.0 * best_surrogate_loss:

        if surrogate_loss < best_surrogate_loss:
            break
        else:
            print("Surrogate re-training")
            pre_train(surrogate_model, surrogate_dataset, n_epochs_pre)
            surrogate_model.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 5, lr=0.005)
            surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
            surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
            sl = surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0001)

    surr_out, reco_out, true_in = surrogate_model.apply_model_in_batches(surrogate_dataset, batch_size=512)

    print("Finished, surrogate model output:", surr_out)

surrogate_dataset.df.to_parquet(output_path, index=range(len(surrogate_dataset.df)))
