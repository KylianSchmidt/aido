from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from modules.optimization_helpers import ParameterModule
from modules.simulation_helpers import SimulationParameterDictionary
from modules.surrogate import Surrogate, SurrogateDataset


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
            starting_parameter_dict: SimulationParameterDictionary,
            lr=0.001,
            batch_size=128,
            ):
        """
        Initializes the optimizer with the given surrogate model and parameters.
        Args:
            surrogate_model (Surrogate): The surrogate model to be optimized.
            starting_parameter_dict (Dict): A dictionary containing the initial parameters.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            batch_size (int, optional): Batch size for the optimizer. Defaults to 128.
        """

        super().__init__()
        self.surrogate_model = surrogate_model
        self.parameter_dict = starting_parameter_dict
        self.n_time_steps = surrogate_model.n_time_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda")

        self.parameter_module = ParameterModule(self.parameter_dict)
        self.starting_parameters_continuous = self.parameter_module.tensor("continuous")
        self.parameter_box = self.parameter_module.constraints
        self.covariance = self.parameter_module.covariance
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameter_module.parameters(), lr=self.lr)

    def to(self, device: str):
        """ Move all Tensors and modules to 'device'.
        """
        self.device = device
        self.surrogate_model.to(device)
        self.starting_parameters_continuous = self.starting_parameters_continuous.to(device)
        self.parameter_box = self.parameter_box.to(device)
        self.parameter_module = self.parameter_module.to(device)
        super().to(device)
        return self
    
    def loss_box_constraints(self) -> torch.Tensor:
        """ Adds penalties for parameters that are outside of the boundaries spaned by 'self.parameter_box'. This
        ensures that the optimizer does not propose new values that are outside of the scope of the Surrogate and
        therefore largely unknown to the current iteration.
        Returns:
        -------
            float
        """
        if len(self.parameter_box) != 0:
            parameters_continuous_tensor = self.parameter_module.tensor("continuous")
            return (
                torch.mean(10. * torch.nn.Softmax()(self.parameter_box[:, 0] - parameters_continuous_tensor))
                + torch.mean(10. * torch.nn.Softmax()(- self.parameter_box[:, 1] + parameters_continuous_tensor))
            )

    def other_constraints(
            self,
            constraints_func: None | Callable[[SimulationParameterDictionary], float | torch.Tensor]
            ) -> float:
        """ Adds user-defined constraints defined in 'interface.py:AIDOUserInterface.constraints()'. If no constraints
        were added manually, this method defaults to calculating constraints based on the cost per parameter specified
        in ParameterDict. Returns a float or torch.Tensor which can be considered as a penalty loss.
        """
        if constraints_func is None:
            loss = self.parameter_module.cost_loss
        else:
            loss = constraints_func(self.parameter_dict)
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        return loss

    def optimize(
            self,
            dataset: SurrogateDataset,
            batch_size: int,
            n_epochs: int,
            lr: float,
            additional_constraints: None | Callable[[SimulationParameterDictionary], float | torch.Tensor] = None,
            ):
        """ Perform the optimization step. First, the ParameterModules forward() method generates new parameters.
        Second, the Surrogate Model computes the corresponding Reconstruction Loss (based on its interpolation).
        The Optimizer Loss is the Sum of the Reconstruction Loss, user-defined Parameter Loss (e.g. cost constraints)
        and the Parameter Box Loss (which ensures that the Parameters stay within acceptable boundaries during
        training). Fourth, the optimizer applies backprogation and updates the current ParameterDict
        """
        self.optimizer.lr = lr
        self.surrogate_model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer_loss = []
        self.constraints_loss = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            stop_epoch = False

            for batch_idx, (_parameters, context, reconstructed) in enumerate(data_loader):
                context = context.to(self.device)
                reconstructed = reconstructed.to(self.device)

                parameters_batch = self.parameter_module()
                self.parameter_dict.update_current_values(self.parameter_module.physical_values(format="dict"))
                self.parameter_dict.update_probabilities(self.parameter_module.get_probabilities())

                surrogate_output = self.surrogate_model.sample_forward(
                    parameters_batch,
                    torch.unsqueeze(context[batch_idx], 0)
                )
                loss = surrogate_output.clone()
                loss += self.other_constraints(additional_constraints)
                loss += self.loss_box_constraints()

                self.optimizer.zero_grad()
                loss.backward()

                if np.isnan(loss.item()):
                    # Save parameters, reset the optimizer as if it made a step but without updating the parameters
                    print("Optimizer: NaN loss, exiting.")
                    prev_parameters = parameters_batch.clone()
                    self.optimizer.step()

                    for i, parameter in enumerate(self.parameter_module):
                        parameter.data = prev_parameters[i].to(self.device)

                    return self.parameter_dict, False, epoch_loss / (batch_idx + 1)

                self.optimizer.step()
                epoch_loss += loss.item()

                if not self.parameter_module.check_parameters_are_local(self.parameter_module.tensor("continuous")):
                    stop_epoch = True
                    break

            print(f"Optimizer Epoch: {epoch} \tLoss: {surrogate_output.item():.5f} (reco)", end="\t")
            print(f"+ {(self.other_constraints(additional_constraints)):.5f} (constraints)", end="\t")
            print(f"+ {(self.loss_box_constraints()):.5f} (boundaries)", end="\t")
            print(f"= {loss.item():.5f} (total)")
            epoch_loss /= batch_idx + 1
            self.optimizer_loss.append(epoch_loss)
            self.constraints_loss.append(self.other_constraints(additional_constraints))

            if stop_epoch:
                break

        self.covariance = self.parameter_module.adjust_covariance(
            self.parameter_module.tensor("continuous").to(self.device)
            - self.starting_parameters_continuous.to(self.device)
            )
        return self.parameter_dict, True

    def get_optimum(self):
        return self.parameter_dict
