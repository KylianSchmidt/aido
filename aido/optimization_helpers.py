from typing import Iterable, Literal, Tuple

import numpy as np
import torch

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
            torch.tensor(np.repeat(1.0 / len(self.discrete_values), len(self.discrete_values)), dtype=torch.float32),
            requires_grad=True
        )
        self._cost = parameter.cost if parameter.cost is not None else 0.0

    def forward(self) -> torch.Tensor:
        """ Passes the probabilities of each entry """
        return self.probabilities

    @property
    def current_value(self) -> torch.Tensor:
        """ Returns the index corresponding to highest scoring entry """
        return self.probabilities

    @property
    def physical_value(self) -> torch.Tensor:
        """ Returns the value of the highest scoring entry """
        return self.discrete_values[torch.argmax(self.current_value.clone().detach()).item()]

    @property
    def probabilities(self) -> torch.Tensor:
        """ Probabilities for each entry, with a minimal probability of 1%"""
        probabilities = torch.nn.functional.softmax(self.logits, dim=0)
        probabilities = torch.clamp(probabilities, min=0.01)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        return probabilities

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
        self.starting_value = torch.tensor(parameter.current_value)
        self.parameter = torch.nn.Parameter(self.starting_value.clone(), requires_grad=True)
        self.min_value = parameter.min_value or -10E10
        self.max_value = parameter.max_value or +10E10
        self.boundaries = torch.tensor(np.array([self.min_value, self.max_value], dtype="float32"))
        self._cost = parameter.cost if parameter.cost is not None else 0.0
        self.reset(parameter)

    def reset(self, parameter: SimulationParameter):
        self.parameter.data = torch.clamp(self.parameter.data, self.min_value, self.max_value)
        assert (
            torch.isclose(torch.tensor(parameter.current_value), torch.tensor(self.physical_value))
        ), f"Values are {parameter.current_value} != {self.physical_value} and {self.parameter}"
        self.sigma = np.array(parameter.sigma)

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(self.parameter, 0)

    @property
    def current_value(self) -> torch.Tensor:
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
        self.parameters_discrete: dict[str, OneHotEncoder] = {}
        self.parameters_continuous: dict[str, ContinuousParameter] = {}
        self.covariance = self.reset_covariance()
        super().__init__()

        for parameter in self.parameter_dict:
            if not parameter.optimizable:
                continue

            if parameter.discrete_values:
                self.parameters_discrete[parameter.name] = OneHotEncoder(parameter)
                self[parameter.name] = self.parameters_discrete[parameter.name]
            else:
                self.parameters_continuous[parameter.name] = ContinuousParameter(parameter)
                self[parameter.name] = self.parameters_continuous[parameter.name]

    def items(self) -> Iterable[Tuple[str, OneHotEncoder | ContinuousParameter]]:
        return super().items()

    def values(self) -> Iterable[OneHotEncoder | ContinuousParameter]:
        return super().values()

    def reset_covariance(self) -> np.ndarray:
        """ Returns the current covariance matrix for all continuous parameters in order.
        Covariance is a misnomer as we use the standard deviation and not the variance
        """
        return np.diag(np.array(
            [parameter.sigma.item() for parameter in self.parameters_continuous.values()],
            dtype="float32"
        ))

    def __call__(self) -> torch.Tensor:
        return super().__call__()

    def forward(self) -> torch.Tensor:
        return torch.unsqueeze(torch.concat([parameter() for parameter in self.values()]), 0)

    @property
    def discrete(self) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(self.parameters_discrete)

    @property
    def continuous(self) -> torch.nn.ModuleDict:
        return torch.nn.ModuleDict(self.parameters_continuous)

    def tensor(self, parameter_types: Literal["all", "discrete", "continuous"] = "all"):
        types = {
            "all": self,
            "discrete": self.parameters_discrete,
            "continuous": self.parameters_continuous
        }
        assert parameter_types != "all", NotImplementedError("Not implemented yet due to dimension mismatch")
        module: dict[str, OneHotEncoder | ParameterModule] = types[parameter_types]
        tensor_list = [parameter.current_value for parameter in module.values()]

        if tensor_list == []:
            return torch.tensor([])
        else:
            return torch.stack(tensor_list)

    def current_values(self) -> dict:
        return {name: parameter.current_value for name, parameter in self.items()}

    def physical_values(self, format: Literal["list", "dict"] = "list") -> list | dict:
        if format == "list":
            return [parameter.physical_value for parameter in self.values()]
        elif format == "dict":
            return {name: parameter.physical_value for name, parameter in self.items()}
        
    def get_probabilities(self) -> dict[str, np.ndarray]:
        return {
            name: parameter.probabilities.detach().cpu().numpy() for name, parameter in self.parameters_discrete.items()
        }

    @property
    def constraints(self) -> torch.Tensor:
        """ A tensor of shape (P, 2) where P is the number of continuous parameters. In the second index,
        the order is the same as in the ContinuousParameter class (min, max)
        """
        tensor_list = [parameter.boundaries for parameter in self.parameters_continuous.values()]
        if tensor_list == []:
            return torch.tensor([])
        else:
            return torch.stack(tensor_list)

    @property
    def cost_loss(self) -> torch.Tensor:
        return sum(parameter.cost for parameter in self.values())
    
    def reset_continuous_parameters(self, parameter_dict: SimulationParameterDictionary):
        for name, parameter in self.parameters_continuous.items():
            parameter.reset(parameter_dict[name])
        self.covariance = self.reset_covariance()

    def adjust_covariance(self, direction: torch.Tensor, min_scale: float = 2.0):
        """ Stretches the box_covariance of the generator in the directon specified as input.
        Direction is a vector in parameter space
        """
        direction = direction.cpu().detach().numpy()
        norm = np.linalg.norm(direction)
        direction_normed = direction / (norm + 1e-4)

        scale_factor = min_scale * max(1, 4 * norm)

        for index, parameter in enumerate(self.parameters_continuous.values()):
            new_variance = parameter.sigma**2 * 1 + (scale_factor - 1) * (direction_normed[index]**2)
            parameter.sigma = np.sqrt(new_variance)

        return self.reset_covariance()

    def check_parameters_are_local(self, updated_parameters: torch.Tensor, scale=1.0) -> bool:
        """ Assure that the predicted parameters by the optimizer are within the bounds of the covariance
        matrix spanned by the 'sigma' of each parameter.
        """
        diff = updated_parameters - self.tensor("continuous")
        diff = diff.detach().cpu().numpy()

        if np.any(self.covariance >= 10E3) or not np.any(self.covariance):
            self.covariance = self.reset_covariance()

        if self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        return np.dot(diff, np.dot(np.linalg.inv(self.covariance), diff)) < scale
