import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from surrogate import Surrogate, SurrogateDataset
from reconstruction import Reconstruction
from typing import Dict


class Optimizer(object):
    '''
    The optimizer uses the surrogate model to optimise the detector parameters in batches.
    It is also linked to a generator object, to check if the parameters are still in bounds using the function is_local(parameters)
    of the generator.

    Once the parameters are not local anymore, the optimizer will return the last parameters that were local and stop.
    For this purpose, the surrogate model will need to be applied using fixed weights.
    Then the reconstruction model loss will be applied based on the surrogate model output.
    The gradient w.r.t. the detector parameters will be calculated and the parameters will be updated.
    '''
    def __init__(
            self,
            surrogate_model: Surrogate,
            reconstruction_model: Reconstruction,
            starting_parameter_dict: Dict,
            lr=0.001,
            batch_size=128,
            ):
        
        self.surrogate_model = surrogate_model
        self.reconstruction_model = reconstruction_model                

        self.starting_parameter_dict = starting_parameter_dict
        self.parameter_dict = self.starting_parameter_dict
        self.parameter_dict = {k: v for k, v in self.parameter_dict.items() if v.get("optimizable")}

        self.n_time_steps = surrogate_model.n_time_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device('cuda')

        self.starting_parameters = self.parameter_dict_to_cuda()
        self.parameters = self.parameter_dict_to_cuda()
        self.parameter_box = self.parameter_constraints_to_cuda_box()
        self.covariance = self.get_covariance_matrix()

        self.to(self.device)
        print("DEBUG Parameters", self.parameters)
        self.optimizer = torch.optim.Adam([torch.nn.Parameter(self.parameters, requires_grad=True)], lr=self.lr)

    def parameter_dict_to_cuda(self):
        """ Parameter list as cuda tensor
        Shape: (Parameter, 1)
        """
        parameters = [parameter["current_value"] for parameter in self.parameter_dict.values()]
        return torch.tensor(np.array(parameters, dtype="float32"))

    def parameter_constraints_to_cuda_box(self):
        """ Convert the constraints of parameters to a multi-dimensional 'box'. Parameters with no constraints
        will have entries marked as np.Nan.
        Shape: (Parameter, Constraint)
        Where Constraint is [min, max]

        TODO Make compatible with discrete parameters
        """
        parameter_box = []

        for parameter in self.parameter_dict:
            parameter_box.append([
                self.parameter_dict[parameter]["min_value"],
                self.parameter_dict[parameter]["max_value"]
            ])
        
        parameter_box = np.array(parameter_box, dtype="float32")
        return torch.tensor(parameter_box, dtype=torch.float32)
    
    def get_covariance_matrix(self):
        covariance_matrix = []

        for parameter in self.parameter_dict:
            covariance_matrix.append(self.parameter_dict[parameter]["sigma"])

        return np.array(covariance_matrix, dtype="float32")

    def to(self, device: str):
        self.device = device
        self.surrogate_model.to(device)
        self.parameters.to(device)
        self.parameter_box.to(device)

    def other_constraints(self, constraints: Dict = {"length": 25}):
        """ Keep parameters such that within the box size of the generator, there are always some positive values even if the 
        central parameters are negative. Both box size and raw_detector_parameters_list are in non-normalised space, so this is straight forward
        the generator will have to provide the box size
        this will avoid mode collapse
        """
        self.constraints = constraints
        detector_length = torch.sum(self.parameters)

        total_length_loss = torch.mean(100. * torch.nn.ReLU()(detector_length - self.constraints["length"])**2)
        box_loss = (
            torch.mean(100. * torch.nn.ReLU()(-self.parameter_box / 1.1 - self.parameters[:, 0])**2)
            + torch.mean(100. * torch.nn.ReLU()(-self.parameter_box / 1.1 - self.parameters[:, 0])**2)
        )

        return total_length_loss + box_loss

    def adjust_covariance(self, direction: torch.Tensor, min_scale=2.0):
        """ Stretches the box_covariance of the generator in the directon specified as input
        Direction is a vector in parameter space
        """
        parameter_direction_vector = direction.detach().cpu().numpy()
        parameter_direction_length = np.linalg.norm(parameter_direction_vector)

        scaling_factor = min_scale * np.max([1., 4. * parameter_direction_length])
        # Create the scaling adjustment matrix
        M_scaled = (scaling_factor - 1) * np.outer(
            parameter_direction_vector / parameter_direction_length,
            parameter_direction_vector / parameter_direction_length
        )
        # Adjust the original covariance matrix
        self.covariance = np.diag(self.covariance**2) + M_scaled
        print("Optimizer: New covariance matrix\n", self.covariance)

    def check_parameter_are_local(self, updated_parameters: torch.Tensor, scale=1.0) -> bool:
        diff = updated_parameters - self.parameters
        diff = diff.detach().cpu().numpy()
        return np.dot(diff, np.dot(np.linalg.inv(self.covariance), diff)) < scale

    def optimize(self, dataset: SurrogateDataset, batch_size: int, n_epochs: int, lr, add_constraints=False):
        '''
        keep both models fixed, train only the detector parameters (self.detector_start_parameters)
        using the reconstruction model loss
        '''
        self.optimizer.lr = lr
        self.surrogate_model.eval()
        self.reconstruction_model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        mean_loss = 0

        for epoch in range(n_epochs):
            mean_loss = 0
            stop_epoch = False

            for batch_idx, (_parameters, targets, true_context, reco_result) in enumerate(data_loader):
                self.parameters = self.parameters.to(self.device)
                targets = targets.to(self.device)
                true_context = true_context.to(self.device)
                reco_result = reco_result.to(self.device)

                print("DEBUG Device", self.device)
                reco_surrogate = self.surrogate_model.sample_forward(self.parameters, targets, true_context)
                reco_surrogate = reco_surrogate * dataset.stds[1] + dataset.means[1]
                targets = targets * dataset.stds[1] + dataset.means[1]
                loss = self.reconstruction_model.loss(reco_surrogate, targets)

                if add_constraints:
                    loss += self.other_constraints(dataset)

                self.optimizer.zero_grad()
                loss.backward()

                if np.isnan(loss.item()):
                    # Save parameters, reset the optimizer as if it made a step but without updating the parameters
                    print("Optimizer: NaN loss, exiting.")
                    prev_parameters = self.parameters.detach().cpu().numpy()
                    self.optimizer.step()
                    self.parameters.data = torch.tensor(prev_parameters).to(self.device)
                    return self.parameters.detach().cpu().numpy(), False, mean_loss / (batch_idx + 1)

                self.optimizer.step()
                mean_loss += loss.item()

                if not self.check_parameter_are_local(self.parameters, 0.8):
                    stop_epoch = True
                    break

                if batch_idx % 20 == 0:
                    self.updated_parameter_array = self.parameters.detach().cpu().numpy()

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = self.updated_parameter_array[index]

                    self.parameters.to(self.device)
                    print('Current parameters: \n', json.dumps(self.parameter_dict, indent=4))

            print(f'Optimizer Epoch: {epoch} \tLoss: {loss.item():.8f}')

            if stop_epoch:
                break

        mean_loss /= batch_idx + 1
        self.adjust_covariance(self.parameters - self.starting_parameters)
        return self.updated_parameter_array, True, mean_loss

    def get_optimum(self):
        return self.updated_parameter_array