from typing import Iterable, List, Literal, Tuple

import numpy as np
import torch

from aido.logger import logger
from aido.simulation_helpers import SimulationParameter, SimulationParameterDictionary


class OneHotEncoder(torch.nn.Module):
    """
    OneHotEncoder is a module that performs one-hot encoding on discrete values.

    Attributes:
        logits (torch.Tensor): A set of unnormalized, real-valued scores for each category. These logits
            represent the model's confidence in each category prior to normalization. They can take any
            real value, including negatives, and are not probabilities themselves. Use the probabilities
            property to convert the logits to probabilities.
    
    TODO Restrict the learning rate of the logits since they converge much faster than Continuous parameters.
    """
    def __init__(self, parameter: SimulationParameter):
        """
        Args:
            parameter (dict): A dictionary containing the parameter information.
        """
        super().__init__()
        self.discrete_values: list = parameter.discrete_values
        self.starting_value = torch.tensor(self.discrete_values.index(parameter.current_value))
        self.logits = torch.nn.Parameter(
            torch.log(torch.tensor(parameter.probabilities, dtype=torch.float32)),
            requires_grad=True
        )
        self._cost = parameter.cost if parameter.cost is not None else 0.0

    def forward(self) -> torch.Tensor:
        """ Passes the probabilities of each entry """
        return self.probabilities

    @property
    def current_value(self) -> torch.Tensor:
        """ Returns the probability Tensor """
        return self.probabilities

    @property
    def physical_value(self) -> torch.Tensor:
        """ Returns the value of the highest scoring entry """
        return self.discrete_values[torch.argmax(self.current_value.clone().detach()).item()]

    @property
    def probabilities(self) -> torch.Tensor:
        """ Probabilities for each entry"""
        return torch.nn.functional.softmax(self.logits, dim=0)

    @property
    def cost(self) -> torch.Tensor:
        """ Costs associated to each entry """
        return torch.dot(self.probabilities, torch.tensor(self._cost, device=self.probabilities.device))


class ContinuousParameter(torch.nn.Module):
    def __init__(self, parameter: SimulationParameter):
        """
        Initializes the optimization helper with the given parameters.

        Args:
            parameter (dict): A parameter dict with the format given by
            'modules/simulation_helpers.py:SimulationParameterDictionary'.

        Attributes:
            starting_value (torch.Tensor): The initial value as a tensor.
            parameter (torch.nn.Parameter): The parameter wrapped in a PyTorch Parameter object.
            min_value (float): The minimum allowable value.
            max_value (float): The maximum allowable value.
            boundaries (torch.Tensor): A tensor containing the min and max values.
            sigma (numpy.ndarray): The standard deviation for the parameter.
            _cost (float): The cost associated with the parameter.
        """
        super().__init__()
        self.reset(parameter)
        self.starting_value = torch.tensor(parameter.current_value)
        self.parameter = torch.nn.Parameter(self.starting_value.clone(), requires_grad=True)
        self.min_value = parameter.min_value if parameter.min_value is not None else -10E10
        self.max_value = parameter.max_value if parameter.max_value is not None else +10E10
        self.boundaries = torch.tensor(np.array([
            (- self.sigma + self.min_value) / 1.1,
            (self.sigma + self.max_value) / 1.1
        ], dtype="float32"))
        self._cost = parameter.cost if parameter.cost is not None else 0.0

    def reset(self, parameter: SimulationParameter):
        self.sigma = np.array(parameter.sigma)

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(self.parameter, 0)

    @property
    def current_value(self) -> torch.Tensor:
        if torch.isnan(self.parameter.data):
            logger.debug(self.__dict__)
        return self.parameter

    @property
    def physical_value(self) -> float:
        return self.current_value.item()

    @property
    def cost(self) -> float:
        return self.physical_value * self._cost
    

class ParameterModule(torch.nn.ModuleDict):
    def __init__(self, parameter_dict: SimulationParameterDictionary):
        self.parameter_dict = parameter_dict
        super().__init__()

        for parameter in self.parameter_dict:
            if not parameter.optimizable:
                continue
            if parameter.discrete_values is not None:
                self[parameter.name] = OneHotEncoder(parameter)
            else:
                self[parameter.name] = ContinuousParameter(parameter)

    def items(self) -> Iterable[Tuple[str, OneHotEncoder | ContinuousParameter]]:
        return super().items()

    def values(self) -> Iterable[OneHotEncoder | ContinuousParameter]:
        return super().values()

    def __getitem__(self, key: str) -> OneHotEncoder | ContinuousParameter:
        return super().__getitem__(key)

    def __call__(self) -> torch.Tensor:
        return super().__call__()

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(torch.concat([parameter() for parameter in self.values()]), 0)

    def continuous_tensors(self) -> torch.Tensor:
        tensor_list: List[torch.Tensor] = []

        for parameter in self.values():
            if isinstance(parameter, ContinuousParameter):
                tensor_list.append(parameter.current_value)

        return torch.stack(tensor_list)

    def current_values(self) -> dict:
        return {name: parameter.current_value for name, parameter in self.items()}

    def physical_values(self, format: Literal["list", "dict"] = "list") -> list | dict:
        if format == "list":
            return [parameter.physical_value for parameter in self.values()]
        elif format == "dict":
            return {name: parameter.physical_value for name, parameter in self.items()}

    @property
    def probabilities(self) -> dict[str, np.ndarray]:
        probability_dict = {}

        for name, parameter in self.items():
            if isinstance(parameter, OneHotEncoder):
                probability_dict[name] = parameter.probabilities.detach().cpu().numpy()

        return probability_dict

    @property
    def constraints(self) -> torch.Tensor:
        """ A tensor of shape (P, 2) where P is the number of continuous parameters. In the second index,
        the order is the same as in the ContinuousParameter class (min, max)
        """
        tensor_list: List[torch.Tensor] = []

        for parameter in self.values():
            if isinstance(parameter, ContinuousParameter):
                tensor_list.append(parameter.boundaries)

        return torch.stack(tensor_list)

    @property
    def cost_loss(self) -> torch.Tensor:
        return sum(parameter.cost for parameter in self.values())

    @property
    def covariance(self) -> np.ndarray:
        return self.parameter_dict.covariance

    @covariance.setter
    def covariance(self, new_covariance: np.ndarray):
        self.parameter_dict.covariance = new_covariance

    def adjust_covariance(self, direction: torch.Tensor, min_scale: float = 2.0):
        """ Stretches the box_covariance of the generator in the directon specified as input.
        Direction is a vector in parameter space
        """
        direction = direction.cpu().detach().numpy()
        direction_normed = direction / (np.linalg.norm(direction) + 1e-4)
        scale_factor = min_scale * max(1, 4 * np.linalg.norm(direction))
        adjustement_matrix = (scale_factor - 1) * np.outer(direction_normed, direction_normed)
        self.covariance = self.parameter_dict.sigma_array**2 + adjustement_matrix
        return self.covariance
