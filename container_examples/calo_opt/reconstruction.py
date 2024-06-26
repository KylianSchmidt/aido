""" Contains the reconstrction module
neural network model for reconstruction of energy from detector depostits and parameters
contains also the training loop
the dataset is provided by the Generator as a pandas dataframe
Data loader etc are in here, too
"""
import sys
import json
from typing import Dict
import torch
import numpy as np
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader
from generator import CaloDataset


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

    def create_torch_dataset(self, simulation_parameter_list, input_features, target_features, context_information):
        self.means = [
            np.mean(simulation_parameter_list, axis=0),
            np.mean(input_features, axis=0),
            np.mean(target_features, axis=0),
            np.mean(context_information, axis=0)
        ]
        self.stds = [
            np.std(simulation_parameter_list, axis=0) + 1e-3,
            np.std(input_features, axis=0) + 1e-3,
            np.std(target_features, axis=0) + 1e-3,
            np.std(context_information, axis=0) + 1e-3
        ]
        return CaloDataset(
            simulation_parameter_list,
            input_features,
            target_features,
            context_information,
            self.means,
            self.stds
        )

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
    
    def train_model(self, dataset, batch_size, n_epochs, lr):
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

    def apply_model_in_batches(self, dataset, batch_size):
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

        # loop over the batches
        for batch_idx, (detector_parameters, x, y) in enumerate(data_loader):
            detector_parameters = detector_parameters.to(self.device)
            x = x.to(self.device)
            y = y.to(self.device)
            # apply the model
            y_pred = self(detector_parameters, x)
            # calculate the loss
            loss = self.loss(dataset.unnormalise_target(y_pred), dataset.unnormalise_target(y))
            mean_loss += loss.item()
            if results is None:
                results = torch.zeros(len(dataset), y_pred.shape[1])
            # store the results
            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred
        mean_loss /= len(data_loader)

        return results, mean_loss


def create_features(parameter_dict_file_path: str, simulation_output_file_path: str):
    """Convert the files from the simulation to simple lists.

    Args:
        parameter_dict_file_path (str): The file path to the parameter dictionary file.
        simulation_output_file_path (str): The file path to the simulation output file.

    Returns:
        Tuple[List, pd.DataFrame, pd.DataFrame]: A tuple containing the simulation parameter list,
        input features, and target features.
    TODO Return everything as lists or as dicts (relevant for 'create_torch_dataset' function)
    """
    # 1. Simulation Parameter list
    with open(parameter_dict_file_path, "r") as file:
        parameter_dict: Dict = json.load(file)

    simulation_parameter_list = []

    for parameter in parameter_dict.values():
        if parameter["optimizable"] is True:
            simulation_parameter_list.append(parameter["current_value"])

    simulation_parameters = np.array(simulation_parameter_list, dtype='float32')

    # 2. Simulation output (pd.DataFrame -> linear array)
    simulation_output_df: pd.DataFrame = pd.read_pickle(simulation_output_file_path)
    input_features_keys = [
        'sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z',
        'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer'
        ]
    input_features = np.array([simulation_output_df[par].to_numpy() for par in input_features_keys], dtype='float32')
    input_features = np.swapaxes(input_features, 0, 1)
    input_features = np.reshape(input_features, (len(input_features), -1))

    # 3. Reconstruction targets
    target_features_keys = ["true_energy"]
    target_features = simulation_output_df[target_features_keys].to_numpy(dtype='float32')

    # 4. Context information from simulation
    context_information_keys = ["true_pid"]
    context_information = simulation_output_df[context_information_keys].to_numpy(dtype='float32')

    # Reshape parameters to (N, num_parameters)
    simulation_parameters = np.repeat([simulation_parameters], len(target_features), axis=0)

    shape = (simulation_parameters.shape[1], input_features.shape[1], target_features.shape[1], context_information.shape[1])
    return simulation_parameters, input_features, target_features, context_information, shape
    

def pre_train(model: Reconstruction, data_set: CaloDataset, n_epochs: int):
    """ Pre-train the reconstruction algorithm

    TODO Reconstruction results are normalized. In the future only expose the un-normalised ones, 
    but also requires adjustments to the surrogate dataset
    """
    
    model.to('cuda')

    print('pre-training 0')
    model.train_model(data_set, batch_size=256, n_epochs=10, lr=0.03)

    print('pre-training 1')
    model.train_model(data_set, batch_size=256, n_epochs=n_epochs, lr=0.01)

    print('pre-training 2')
    model.train_model(data_set, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print('pre-training 3')
    model.train_model(data_set, batch_size=1024, n_epochs=n_epochs, lr=0.001)

    reco_result, reco_loss = model.apply_model_in_batches(data_set, batch_size=128)
    reco_model.to('cpu')

    # Mising bit of the surrogate model 


simulation_output_file_path = sys.argv[1]
parameter_dict_file_path = sys.argv[2]
output_path = sys.argv[3]

simulation_parameter_list, input_features, target_features, context_information, shape = create_features(parameter_dict_file_path, simulation_output_file_path)
print("DATA_SET SHAPE:", shape)

reco_model = Reconstruction(*shape)

n_epochs_pre = 3
n_epochs_main = 10
data_set = reco_model.create_torch_dataset(simulation_parameter_list, input_features, target_features, context_information)

pre_train(reco_model, data_set, n_epochs_pre)

for i in range(3):
    reco_model.to('cuda')
    reco_model.train_model(data_set, batch_size=256, n_epochs=n_epochs_main // 4, lr=0.003)
    reco_model.train_model(data_set, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
    reco_model.train_model(data_set, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
    reco_result, reco_loss = reco_model.apply_model_in_batches(data_set, batch_size=128)
    print(f"Reconstruction Block {i} DONE, loss={reco_loss:.8f}")

reco_result.detach().cpu().numpy().tofile(output_path)
