import torch
import numpy as np
from torch.utils.data import DataLoader
from surrogate import Surrogate, SurrogateDataset
from typing import Dict


class Optimizer(object):
    '''
    The optimizer uses the surrogate model to optimise the detector parameters in batches.
    It is also linked to a generator object, to check if the parameters are still in bounds using
    the function is_local(parameters) of the generator.

    Once the parameters are not local anymore, the optimizer will return the last parameters that were local and stop.
    For this purpose, the surrogate model will need to be applied using fixed weights.
    Then the reconstruction model loss will be applied based on the surrogate model output.
    The gradient w.r.t. the detector parameters will be calculated and the parameters will be updated.
    '''
    def __init__(
            self,
            surrogate_model: Surrogate,
            starting_parameter_dict: Dict,
            lr=0.001,
            batch_size=128,
            ):
        
        self.surrogate_model = surrogate_model
        self.starting_parameter_dict = starting_parameter_dict
        self.parameter_dict = {k: v for k, v in self.starting_parameter_dict.items() if v.get("optimizable")}
        self.n_time_steps = surrogate_model.n_time_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda"

        self.starting_parameters = self.parameter_dict_to_cuda()
        self.parameters = self.parameter_dict_to_cuda()
        self.parameter_box = self.parameter_constraints_to_cuda()
        self.covariance = self.get_covariance_matrix()

        self.to(self.device)
        self.optimizer = torch.optim.Adam([self.parameters], lr=self.lr)

    def parameter_dict_to_cuda(self):
        """ Parameter list as cuda tensor
        Shape: (Parameter, 1)
        """
        parameters = [parameter["current_value"] for parameter in self.parameter_dict.values()]
        parameters = torch.tensor(np.array(parameters, dtype="float32")).to(self.device)
        return torch.nn.Parameter(parameters, requires_grad=True)

    def parameter_constraints_to_cuda(self):
        """ Convert the constraints of parameters to a multi-dimensional 'box'. Parameters with no constraints
        will have entries marked as np.Nan.
        Shape: (Parameter, Constraint)
        Where Constraint is [min, max]

        TODO Make compatible with discrete parameters
        """
        parameter_box = []

        for parameter in self.parameter_dict:
            min_value = self.parameter_dict[parameter]["min_value"]
            max_value = self.parameter_dict[parameter]["max_value"]
            min_value = np.nan_to_num(min_value, nan=-10e10)  # NOTE Fix to avoid NaN loss in 'other_constraints'
            max_value = np.nan_to_num(max_value, nan=10e10)
            parameter_box.append([min_value, max_value])

        parameter_box = np.array(parameter_box, dtype="float32")
        return torch.tensor(parameter_box, dtype=torch.float32).to(self.device)

    def get_covariance_matrix(self):
        covariance_matrix = []

        for parameter in self.parameter_dict:
            covariance_matrix.append(self.parameter_dict[parameter]["sigma"])

        return np.diag(np.array(covariance_matrix, dtype="float32"))

    def to(self, device: str):
        self.device = device
        self.surrogate_model.to(device)
        self.starting_parameters.to(device)
        self.parameters.to(device)
        self.parameter_box.to(device)

    def other_constraints(self, constraints: Dict = {"length": 25}):
        """ Keep parameters such that within the box size of the generator, there are always some positive values even
        if the central parameters are negative.
        TODO Improve doc string
        """
        self.constraints = {key: torch.tensor(value) for key, value in constraints.items()}
        detector_length = torch.sum(self.parameters)
        total_length_loss = torch.mean(100. * torch.nn.ReLU()(detector_length - self.constraints["length"])**2)
        box_loss = (
            torch.mean(100. * torch.nn.ReLU()(self.parameter_box[:, 0] / 1.1 - self.parameters))
            + torch.mean(100. * torch.nn.ReLU()(- self.parameter_box[:, 1] / 1.1 + self.parameters))
        )
        return total_length_loss + box_loss

    def adjust_covariance(self, direction: torch.Tensor, min_scale=2.0):
        """ Stretches the box_covariance of the generator in the directon specified as input.
        Direction is a vector in parameter space
        """
        parameter_direction_vector = direction.detach().cpu().numpy()
        parameter_direction_length = np.linalg.norm(parameter_direction_vector)

        scaling_factor = min_scale * np.max([1., 4. * parameter_direction_length])
        # Create the scaling adjustment matrix
        parameter_direction_normed = parameter_direction_vector / parameter_direction_length
        M_scaled = (scaling_factor - 1) * np.outer(parameter_direction_normed, parameter_direction_normed)
        # Adjust the original covariance matrix
        self.covariance = np.diag(self.covariance**2) + M_scaled
        return np.diag(self.covariance)

    def check_parameter_are_local(self, updated_parameters: torch.Tensor, scale=1.0) -> bool:
        """ Assure that the predicted parameters by the optimizer are within the bounds of the covariance
        matrix spanned by the 'sigma' of each parameter.
        """
        diff = updated_parameters - self.parameters
        diff = diff.detach().cpu().numpy()

        if self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        return np.dot(diff, np.dot(np.linalg.inv(self.covariance), diff)) < scale
    
    def loss(self, y_pred, y_true) -> torch.Tensor:
        """ Loss function for the optimizer. TODO Should be the same loss as the Reconstruction model
        but since they are in different containers, that will be tricky to implement.
        """
        y_true = torch.where(torch.isnan(y_pred), torch.zeros_like(y_true) + 1., y_true)
        y_true = torch.where(torch.isinf(y_pred), torch.zeros_like(y_true) + 1., y_true)
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred), y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred), y_pred)
        return ((y_pred - y_true)**2 / (torch.abs(y_true) + 1.)).mean()

    def optimize(self, dataset: SurrogateDataset, batch_size: int, n_epochs: int, lr: float, add_constraints=False):
        """ Keep Surrogate model fixed, train only the detector parameters (self.detector_start_parameters)
        TODO Improve documentation of this method.
        """
        self.optimizer.lr = lr
        self.surrogate_model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer_loss = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            stop_epoch = False

            for batch_idx, (_parameters, targets, true_context, reco_result) in enumerate(data_loader):
                targets = targets.to(self.device)
                true_context = true_context.to(self.device)
                reco_result = reco_result.to(self.device)

                reco_surrogate = self.surrogate_model.sample_forward(
                    dataset.normalise_detector(self.parameters),
                    targets,
                    true_context
                )
                reco_surrogate = reco_surrogate * dataset.c_stds[1] + dataset.c_means[1]
                targets = targets * dataset.c_stds[1] + dataset.c_means[1]
                loss = self.loss(reco_surrogate, targets)

                if add_constraints:
                    loss += self.other_constraints()

                self.optimizer.zero_grad()
                loss.backward()

                if np.isnan(loss.item()):
                    # Save parameters, reset the optimizer as if it made a step but without updating the parameters
                    print("Optimizer: NaN loss, exiting.")
                    prev_parameters = self.parameters.detach().cpu().numpy()
                    self.optimizer.step()
                    self.parameters.data = torch.tensor(prev_parameters).to(self.device)

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = float(prev_parameters[index])

                    return self.parameter_dict, False, epoch_loss / (batch_idx + 1)

                self.optimizer.step()
                epoch_loss += loss.item()

                if not self.check_parameter_are_local(self.parameters, 0.8):
                    stop_epoch = True
                    break

                if batch_idx % 20 == 0:
                    self.updated_parameter_array = self.parameters.detach().cpu().numpy()

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = float(self.updated_parameter_array[index])

                    self.parameters.to(self.device)

            print(
                f"Optimizer Epoch: {epoch} \tLoss: {(self.loss(reco_surrogate, targets)):.5f} (reco)\t"
                f"+ {(self.other_constraints()):.5f} (constraints)\t = {loss.item():.5f} (total)"
            )
            epoch_loss /= batch_idx + 1
            optimizer_loss.append(epoch_loss)

            if stop_epoch:
                break

        self.covariance = self.adjust_covariance(self.parameters - self.starting_parameters.to(self.device))
        return self.parameter_dict, True, np.array(optimizer_loss)

    def get_optimum(self):
        return self.parameter_dict
