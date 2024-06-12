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
import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader


class Reconstruction(torch.nn.Module):
    def __init__(self, num_parameters, num_input_features, num_target_features):
        """Initialize the shape of the model.

        Args:
            num_parameters (int): The number of detector parameters (length of the list of simulation parameters).
            num_input_features (int): The number of input features (sensors and other simulation outputs).
            num_target_features (int): The number of target features (quantities which are representatives
                of the detector's capabilities).
        """
        super().__init__()

        self.n_parameters = num_parameters
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
    
    def train_model(self, dataset, batch_size, n_epochs, lr):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer.lr = lr
        self.to(self.device)
        self.train()

        for epoch in range(n_epochs):
            for _, (detector_parameters, x, y) in enumerate(train_loader):
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


def create_torch_dataset(parameter_dict: dict):
    ...


simulation_output_file_path = sys.argv[1]
parameter_dict_file_path = sys.argv[2]
output_path = sys.argv[3]

# Simulation Parameter dicts -> linear array
with open(parameter_dict_file_path, "r") as file:
    parameter_dict: Dict = json.load(file)

print("RECO", parameter_dict)

simulation_parameter_list = []

for parameter in parameter_dict.values():
    if parameter["optimizable"] is True:
        simulation_parameter_list.append(parameter["current_value"])

print("RECO", simulation_parameter_list)

# Simulation output pd.DataFrame -> linear array
simulation_output_df: pd.DataFrame = pd.read_pickle(simulation_output_file_path)

input_features_keys = [
    'sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z',
    'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer'
    ]
input_features = simulation_output_df[input_features_keys]

target_features_keys = ["true energy"]
target_features = simulation_output_df[target_features_keys]

print("RECO", input_features)

reco_model = Reconstruction(
    len(simulation_parameter_list),
    len(input_features_keys),
    len(target_features_keys)
    )

n_epochs_pre = 30//divide
n_epochs_main = 100//divide
parameters = gen.parameters

evolution = []

def save_evolution(parameters, losses, reco_losses, reco_model, means, std):
        '''
        add to evolution list and save it with pickle.
        save it in human readable format as well
        '''
        ps_dict = gen.translate_parameters(parameters)
        ps = {k: ps_dict[k] for k in ps_dict.keys()}
        ps.update(
            {'reco_model': reco_model.state_dict(),
             'means': means,
             'stds': std}
        )
        evolution.append([ps, losses, reco_losses])
        with open(outpath+'evolution.pkl', 'wb') as f:
            pickle.dump(evolution, f)
        # human readable
        with open(outpath+'evolution.txt', 'w') as f:
            for e in evolution:
                f.write(str(e[0]) + ' ' + str(e[1]) + ' ' + str(e[2]) + '\n')

        # now plot all entries of the dictionary ps in a plot and save it, make a new point for each of the items in the evolution list and use the dictionary keys as legend
        # 'transpose' dictionary
        pltdict = {k: [] for k in ps_dict.keys()}
        for i in range(len(evolution)):
            for k in pltdict.keys():
                pltdict[k] += [evolution[i][0][k]]
