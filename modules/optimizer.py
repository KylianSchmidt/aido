import torch
import numpy as np
from torch.utils.data import DataLoader
from surrogate import Surrogate, SurrogateDataset
from optimization_helpers import ParameterModule
from typing import Dict


class Optimizer(torch.nn.Module):
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
        
        super().__init__()
        self.surrogate_model = surrogate_model
        self.parameter_dict = {k: v for k, v in starting_parameter_dict.items() if v.get("optimizable")}
        self.n_time_steps = surrogate_model.n_time_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda"

        self.parameter_module = ParameterModule(self.parameter_dict)
        self.starting_parameters_continuous = self.parameter_module.tensor("continuous")
        self.parameter_box = self.parameter_module.constraints
        self.covariance = self.parameter_module.covariance

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameter_module.parameters(), lr=self.lr)

    def to(self, device: str):
        self.device = device
        self.surrogate_model.to(device)
        self.starting_parameters_continuous.to(device)
        self.parameter_box.to(device)
        super().to(device)
        return self

    def other_constraints(self, constraints: Dict = {}):
        """ Keep parameters such that within the box size of the generator, there are always some positive values even
        if the central parameters are negative.
        TODO Improve doc string
        TODO Total detector length is an example of possible additional constraints. Very specific use now, must be 
        added in the UserInterface class later on.
        TODO Fix module dict for continuous parameters!
        """
        self.constraints = {key: torch.tensor(value) for key, value in constraints.items()}
        parameters_continuous_tensor = self.parameter_module.tensor("continuous")

        loss = (
            torch.mean(100. * torch.nn.ReLU()(self.parameter_box[:, 0] - parameters_continuous_tensor)) +
            torch.mean(100. * torch.nn.ReLU()(- self.parameter_box[:, 1] + parameters_continuous_tensor))
        )

        if "length" in self.constraints.keys():
            detector_length = torch.sum(parameters_continuous_tensor)
            loss += torch.mean(100. * torch.nn.ReLU()(detector_length - self.constraints["length"])**2)

        return loss

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ Loss function for the optimizer. TODO Should be the same loss as the Reconstruction model
        but since they are in different containers, that will be tricky to implement.
        """
        y_true = torch.where(torch.isnan(y_pred), torch.zeros_like(y_true) + 1., y_true)
        y_true = torch.where(torch.isinf(y_pred), torch.zeros_like(y_true) + 1., y_true)
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred), y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred), y_pred)
        return ((y_pred - y_true)**2 / (torch.abs(y_true) + 1.)).mean()

    def optimize(
            self,
            dataset: SurrogateDataset,
            batch_size: int,
            n_epochs: int,
            lr: float,
            add_constraints=True,
            ):
        """ Keep Surrogate model fixed, train only the detector parameters (self.detector_start_parameters)
        TODO Improve documentation of this method.
        """
        self.optimizer.lr = lr
        self.surrogate_model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer_loss = []
        self.constraints_loss = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            stop_epoch = False

            for batch_idx, (_parameters, targets, true_context, reco_result) in enumerate(data_loader):
                targets = targets.to(self.device)
                true_context = true_context.to(self.device)
                reco_result = reco_result.to(self.device)

                parameters_batch = self.parameter_module.forward()
                reco_surrogate = self.surrogate_model.sample_forward(
                    parameters_batch,
                    targets,
                    true_context
                )  # TODO module dict, normalize parameters??
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
                    prev_parameters = parameters_batch
                    self.optimizer.step()

                    for i, parameter in enumerate(self.parameter_module):
                        parameter.data = prev_parameters[i].to(self.device)

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = float(prev_parameters[index])

                    return self.parameter_dict, False, epoch_loss / (batch_idx + 1)

                self.optimizer.step()
                epoch_loss += loss.item()

                if not self.parameter_module.check_parameters_are_local(self.parameter_module.tensor("continuous")):
                    stop_epoch = True
                    break

                if batch_idx % 20 == 0:
                    
                    parameter_array = self.parameter_module.tensor("all").cpu().detach().numpy()

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = float(parameter_array[index])  # TODO correct dtype

            print(
                f"Optimizer Epoch: {epoch} \tLoss: {(self.loss(reco_surrogate, targets)):.5f} (reco)\t"
                #f"+ {(self.other_constraints()):.5f} (constraints)\t = {loss.item():.5f} (total)"
                f"Parameters: {self.parameter_module.tensor()}, Probabilities = {self.parameter_module["num_blocks"].probabilities}"
            )
            epoch_loss /= batch_idx + 1
            self.optimizer_loss.append(epoch_loss)
            #self.constraints_loss.append(self.other_constraints().detach().cpu().numpy())

            if stop_epoch:
                break

        self.covariance = self.parameter_module.adjust_covariance(
            self.parameter_module.tensor("continuous").to(self.device) - self.starting_parameters_continuous.to(self.device)
            )
        return self.parameter_dict, True

    def get_optimum(self):
        return self.parameter_dict
