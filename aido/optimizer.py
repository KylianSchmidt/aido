import os
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aido.logger import logger
from aido.optimization_helpers import ParameterModule
from aido.simulation_helpers import SimulationParameterDictionary
from aido.surrogate import Surrogate, SurrogateDataset


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
            parameter_dict: SimulationParameterDictionary,
            device: str | None = None
            ):
        """
        Initializes the optimizer with the given surrogate model and parameters.
        Args:
            starting_parameter_dict (Dict): A dictionary containing the initial parameters.
            device (str): Defaults to 'cuda'
        """
        super().__init__()
        self.parameter_dict = parameter_dict
        
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev or torch.device(dev)
        
        self.parameter_module = ParameterModule(self.parameter_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameter_module.parameters())

    def to(self, device: str | torch.device, **kwargs) -> "Optimizer":
        """ Move all Tensors and modules to 'device'.
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        super().to(self.device, **kwargs)
        return self

    def check_parameters_are_local(self, updated_parameters: torch.Tensor, scale=1.0) -> bool:
        """ Assure that the predicted parameters by the optimizer are within the bounds of the covariance
        matrix spanned by the 'sigma' of each parameter.
        """
        diff = updated_parameters - self.starting_parameters_continuous
        diff = diff.detach().cpu().numpy()
        return np.dot(diff, np.dot(np.linalg.inv(self.parameter_dict.covariance), diff)) < scale

    @property
    def boundaries(self) -> torch.Tensor:
        """ Adds penalties for parameters that are outside of the boundaries spaned by 'self.parameter_box'. This
        ensures that the optimizer does not propose new values that are outside of the scope of the Surrogate and
        therefore largely unknown to the current iteration.

        Returns:
        --------
            torch.Tensor
        """
        parameter_box = self.parameter_module.constraints.to(self.device)
        if len(parameter_box) != 0:
            parameters_continuous_tensor = self.parameter_module.continuous_tensors()
            lower_boundary_loss = torch.mean(
                0.5 * torch.nn.ReLU()(parameter_box[:, 0] - parameters_continuous_tensor)**2
            )
            upper_boundary_loss = torch.mean(
                0.5 * torch.nn.ReLU()(parameters_continuous_tensor - parameter_box[:, 1])**2
            )
            return lower_boundary_loss + upper_boundary_loss
        else:
            return torch.Tensor([0.0])

    def other_constraints(
            self,
            constraints_func: None | Callable[[SimulationParameterDictionary, Dict], torch.Tensor],
            parameter_dict_as_tensor: Dict[str, torch.nn.Parameter | torch.Tensor]
            ) -> torch.Tensor:
        """ Adds user-defined constraints defined in 'interface.py:AIDOUserInterface.constraints()'. If no constraints
        were added manually, this method defaults to calculating constraints based on the cost per parameter specified
        in ParameterDict. Returns a float or torch.Tensor which can be considered as a penalty loss.
        """
        if constraints_func is None:
            loss = self.parameter_module.cost_loss
        else:
            loss = constraints_func(self.parameter_dict, parameter_dict_as_tensor)
        return loss if loss is not None else torch.tensor(0.0)

    def save_parameters(
            self,
            epoch: int,
            batch_index: int,
            loss: float,
            filepath: str | os.PathLike = "parameter_optimizer_df.parquet",
            ) -> None:
        df = self.parameter_dict.to_df(display_discrete="as_probabilities")
        df["Epoch"] = epoch
        df["Batch"] = batch_index
        df["Surrogate_Prediction"] = loss

        if not os.path.exists(filepath):
            df.to_parquet(filepath)
        else:
            try:
                updated_parameter_optimizer_df = pd.concat([pd.read_parquet(filepath), df], ignore_index=True)
                updated_parameter_optimizer_df.to_parquet(filepath)
            except KeyboardInterrupt:
                logger.warning("Saving the Optimizer Parameters to file before stopping...")
                updated_parameter_optimizer_df.to_parquet(filepath)
                raise

    def print_grads(self) -> None:
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith("surrogate_model"):
                logger.debug(f"Optimizer {name}: Data={param.data}, grads={param.grad}")

    def optimize(
            self,
            surrogate_model: Surrogate,
            dataset: SurrogateDataset,
            batch_size: int,
            n_epochs: int,
            reconstruction_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            additional_constraints: None | Callable[[SimulationParameterDictionary, Dict], torch.Tensor] = None,
            parameter_optimizer_savepath: str | os.PathLike | None = None,
            device: str | None = None,
            lr: float = 0.01,
            ) -> Tuple[SimulationParameterDictionary, bool]:
        """ Perform the optimization step.

        1. The ParameterModule().forward() method generates new parameters.
        2. The Surrogate Model computes the corresponding Reconstruction Loss (based on its interpolation).
        3. The Optimizer Loss is the Sum of the Reconstruction Loss, user-defined Parameter Loss
            (e.g. cost constraints) and the Parameter Box Loss (which ensures that the Parameters stay
            within acceptable boundaries during training).
        4. The optimizer applies backprogation and updates the current ParameterDict

        Returns:
        --------
            SimulationParameterDictionary
            bool
        """
        self.starting_parameter_dict = self.parameter_dict
        self.surrogate_model = surrogate_model
        self.device = device or self.device
        self.to(self.device)

        self.starting_parameters_continuous = self.parameter_module.continuous_tensors().clone().detach()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.surrogate_model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.optimizer_loss = []
        self.constraints_loss = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            epoch_constraints_loss = 0.0
            stop_epoch = False

            for batch_idx, (_parameters, context, targets, _reconstructed) in enumerate(data_loader):
                context: torch.Tensor = context.to(self.device)
                targets: torch.Tensor = targets.to(self.device)
                parameters_batch: torch.Tensor = self.parameter_module()

                surrogate_output = self.surrogate_model.sample_forward(
                    parameters_batch,
                    context,
                    targets
                )
                surrogate_reconstruction_loss = reconstruction_loss(
                    dataset.unnormalize_features(targets, index=2),
                    dataset.unnormalize_features(surrogate_output, index=2)
                )
                loss = surrogate_reconstruction_loss.mean()
                surrogate_loss_detached = loss.item()
                constraints_loss = self.other_constraints(
                    additional_constraints,
                    self.parameter_module.current_values()
                )
                loss += constraints_loss
                loss += self.boundaries

                loss.backward()

                if np.isnan(loss.item()):
                    logger.error("Optimizer: NaN loss, exiting.")
                    self.optimizer.step()
                    return self.parameter_dict, False

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.parameter_dict.update_current_values(self.parameter_module.physical_values(format="dict"))
                self.parameter_dict.update_probabilities(self.parameter_module.probabilities)
                self.save_parameters(epoch, batch_idx, surrogate_loss_detached, parameter_optimizer_savepath)

                epoch_loss += loss.item()
                epoch_constraints_loss += constraints_loss.item()

                if not self.check_parameters_are_local(
                    updated_parameters=self.parameter_module.continuous_tensors(),
                    scale=0.8
                ):
                    stop_epoch = True
                    logger.error("Optimizer: Parameters are not local")
                    break

            logger.info(
                f"Optimizer Epoch: {epoch} \tLoss: {surrogate_loss_detached:.5f} (reco)\t"
                + f"+ {(constraints_loss.item()):.5f} (constraints)\t"
                + f"+ {(self.boundaries().item()):.5f} (boundaries)\t"
                + f"= {loss.item():.5f} (total)"
            )

            epoch_loss /= batch_idx + 1
            epoch_constraints_loss /= batch_idx + 1
            self.optimizer_loss.append(epoch_loss)
            self.constraints_loss.append(epoch_constraints_loss)

            if stop_epoch:
                break

        self.parameter_dict.covariance = self.parameter_module.adjust_covariance(
            self.parameter_module.continuous_tensors().to(self.device)
            - self.starting_parameters_continuous.to(self.device)
        ).astype(float)
        return self.parameter_dict, True

    @property
    def boosted_parameter_dict(self) -> SimulationParameterDictionary:
        r""" Compute a new set of parameters by taking the current parameter dict and boosting it along
        the direction of change between the previous and the current values (only continuous parameters).

        Formula:
            \[
            p_{n+1} = p_{opt} + \frac{1}{2} \left( p_{opt} - p_n \right)
            \]

            Where:
            - \( p_{n+1} \) is the updated parameter dict.
            - \( p_{opt} \) is the current (optimized) parameter dict.
            - \( p_n \) is the starting parameter dict.
        """
        current_values = self.parameter_dict.get_current_values("dict", types="continuous")
        previous_values = self.starting_parameter_dict.get_current_values("dict", types="continuous")

        return self.parameter_dict.update_current_values(
            {key: current_values[key] + 0.5 * (current_values[key] - previous_values[key]) for key in current_values}
        )
