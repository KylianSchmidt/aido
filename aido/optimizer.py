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
    """
    Optimize detector parameters in batches using a surrogate model.

    This optimizer uses the surrogate model to optimize detector parameters while ensuring they
    remain within specified bounds. It works in conjunction with a generator object that validates
    parameter locality through its is_local(parameters) function.

    The optimization process continues until parameters become non-local, at which point it returns
    the last valid set of parameters. The process involves:
    1. Applying the surrogate model with fixed weights
    2. Computing reconstruction model loss from surrogate output
    3. Calculating gradients with respect to detector parameters
    4. Updating parameters based on computed gradients

    Attributes
    ----------
    parameter_dict : SimulationParameterDictionary
        Dictionary containing parameter configurations and constraints
    device : torch.device
        Device on which to perform computations
    parameter_module : ParameterModule
        Module handling parameter transformations and constraints
    optimizer : torch.optim.Optimizer
        Optimization algorithm instance (Adam)
    """
    def __init__(
            self,
            parameter_dict: SimulationParameterDictionary,
            device: str | None = None
            ):
        """
        Initialize the optimizer with given parameters.

        Parameters
        ----------
        parameter_dict : SimulationParameterDictionary
            Dictionary containing initial parameters and their constraints
        device : str or None, optional
            Computing device to use, by default None which selects CUDA if available,
            otherwise CPU

        Notes
        -----
        The optimizer is initialized with Adam optimization algorithm and moves
        all parameters to the specified device.
        """
        super().__init__()
        self.parameter_dict = parameter_dict
        
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev or torch.device(dev)
        
        self.parameter_module = ParameterModule(self.parameter_dict).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameter_module.parameters())

    def to(self, device: str | torch.device, **kwargs):
        """
        Move all Tensors and modules to specified device.

        Parameters
        ----------
        device : str or torch.device
            Target device to move tensors and modules to
        **kwargs : dict
            Additional arguments passed to parent's to() method

        Returns
        -------
        Optimizer
            Self with all components moved to specified device
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        super().to(self.device, **kwargs)
        return self
    
    def check_parameters_are_local(self, updated_parameters: torch.Tensor, scale=1.0) -> bool:
        """
        Verify if predicted parameters are within covariance bounds.

        Check if the predicted parameters lie within the bounds defined by the
        covariance matrix, scaled by the 'sigma' of each parameter.

        Parameters
        ----------
        updated_parameters : torch.Tensor
            New parameter values to check
        scale : float, optional
            Scaling factor for the covariance bounds, by default 1.0

        Returns
        -------
        bool
            True if parameters are within bounds, False otherwise
        """
        diff = updated_parameters - self.starting_parameters_continuous
        diff = diff.detach().cpu().numpy()
        return np.dot(diff, np.dot(np.linalg.inv(self.parameter_dict.covariance), diff)) < scale

    def boundaries(self) -> torch.Tensor:
        """
        Calculate boundary penalty losses for parameters.

        Computes penalties for parameters that lie outside the boundaries defined
        by 'self.parameter_box'. This prevents the optimizer from proposing values
        outside the Surrogate's known scope in the current iteration.

        Returns
        -------
        torch.Tensor
            Sum of lower and upper boundary penalty losses, or zero if no
            constraints are defined
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
        """
        Apply additional user-defined constraints.

        Handles constraints defined in 'interface.py:AIDOUserInterface.constraints()'.
        If no custom constraints are provided, defaults to calculating constraints
        based on the cost per parameter from ParameterDict.

        Parameters
        ----------
        constraints_func : callable or None
            Function that takes SimulationParameterDictionary and parameter dict
            as input and returns a tensor of constraint losses
        parameter_dict_as_tensor : dict
            Dictionary mapping parameter names to their tensor values

        Returns
        -------
        torch.Tensor
            Computed constraint loss value, or zero if no constraints apply
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

        Return:
        ------
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
                loss += self.boundaries()

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
