import copy
import datetime
import json
from typing import Any, Dict, Iterable, Iterator, List, Literal, Type

import numpy as np
import pandas as pd


class SimulationParameter:
    """Base class for all parameters used in the simulation
    """

    def __init__(
            self,
            name: str,
            starting_value: Any,
            current_value: Any | None = None,
            units: str | None = None,
            optimizable: bool = True,
            min_value: float | None = None,
            max_value: float | None = None,
            sigma: float | None = None,
            discrete_values: Iterable | None = None,
            probabilities: Iterable[float] | None = None,
            cost: float | Iterable | None = None
            ):
        """ Create a new Simulation Parameter

        Args
        ----
            name (str): The name of the parameter.
            starting_value (Any): The starting value of the parameter.
            current_value (Any, optional): The current value of the parameter. Defaults to None.
            units (str, optional): The units of the parameter. Defaults to None.
            optimizable (bool, optional): Whether the parameter is optimizable. Defaults to True.
            min_value (float, optional): The minimum value of the parameter. Defaults to None.
            max_value (float, optional): The maximum value of the parameter. Defaults to None.
            sigma (float, optional): The standard deviation of the parameter. Defaults to None.
            discrete_values (Iterable, optional): The allowed discrete values of the parameter. Defaults to None.
            probabilities (Iterable[float], optional): A list of the same length as 'discrete_values' used to
                sample from 'discrete_values', if set to None, an equally-distributed array is created.
                Only for discrete parameters. Defaults to None
            cost (float, Iterable, optional): A float that quantifies the cost per unit of this Parameter.
                Defaults to None. For discrete parameters, this parameter must be an Iterable (e.g. list) of the
                same length as 'discrete_values'.
        """
        assert isinstance(name, str), "Name must be a string"

        self.name = name
        self._starting_value = starting_value
        self._optimizable = optimizable
        self.sigma = sigma
        self.units = units

        assert (
            isinstance(min_value, type(starting_value)) or min_value is None
            ), "Only float parameters are allowed to have lower bounds"

        assert (
            isinstance(max_value, type(starting_value)) or max_value is None
        ), "Only float parameters are allowed to have upper bounds"

        self.min_value = min_value
        self.max_value = max_value

        if current_value is not None:
            self._current_value = current_value
        else:
            self._current_value = starting_value

        assert (
            isinstance(self._current_value, float) is True or (discrete_values is not None or optimizable is False)
        ), "Specify discrete values when the parameter is not a float, but is optimizable"

        self.discrete_values = discrete_values

        if self.discrete_values is not None:
            if optimizable is False:
                raise AssertionError("Non-optimizable parameters are excluded from requiring allowed discrete values")
            assert (
                self._current_value in self.discrete_values
            ), "Current value must be included in the list of allowed discrete values"
            assert (
                starting_value in self.discrete_values
            ), "Starting value must be included in the list of allowed discrete values"
            assert (
                self.min_value is None and self.max_value is None
            ), "Not allowed to specify min and max value for parameter with discrete values"
            if probabilities is None:
                self.probabilities = np.full(len(self.discrete_values), 1.0 / len(self.discrete_values))
            else:
                self.probabilities = probabilities

        if sigma is None:
            if discrete_values is None and optimizable is True:
                self.sigma = 0.1 * self.current_value
        else:
            assert (
                isinstance(sigma, float) and discrete_values is None and optimizable is True
            ), "Unable to asign standard deviation to discrete or non-optimizable parameter."

        if cost is None:
            pass
        elif self.discrete_values:
            assert (
                isinstance(cost, Iterable)
            ), "Parameter 'cost' must be an iterable of the same length as 'discrete_values'"
            assert (
                len(cost) == len(self.discrete_values)
            ), f"Length of 'cost' ({len(cost)}) is different"
            f"from that of 'discrete_values' ({len(self.discrete_values)})"
            assert (
                (isinstance(cost_item, float) and cost_item > 0.0 for cost_item in cost)
            ), "Entries of argument 'cost' must be positive floats"
        elif self.discrete_values is None:
            assert (
                isinstance(cost, float) and cost > 0.0
            ), "Cost argument must be a positive float for continuous parameters."
        self.cost = cost

    def __str__(self) -> str:
        """Return the dict representation of the class, with human-readable indentation
        TODO Do not indent lists e.g. in discrete_values=[]
        """
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self) -> Dict:
        """Convert to dictionary

        Protected attributes are written to file as public attributes.
        """
        return {key.removeprefix("_"): value for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, attribute_dict: Dict):
        """Create from dictionary"""
        return cls(**attribute_dict)

    @property
    def current_value(self) -> Any:
        return self._current_value

    @current_value.setter
    def current_value(self, value: Any) -> None:
        if isinstance(self._starting_value, int) and isinstance(value, float) and abs(value - round(value)) < 10E-15:
            value = round(value)
        assert (
            isinstance(value, type(self._starting_value))
        ), f"The updated value is of another type ({type(value)}) than before ({type(self._starting_value)})"
        if self.optimizable is False and value != self._starting_value:
            raise AttributeError("Do not change the current value of non-optimizable parameter")
        assert (
            self.discrete_values is None or value in self.discrete_values
        ), "Updated discrete parameter value is not in the list of allowed discrete values."
        if isinstance(value, float):
            if self.max_value is not None and value > self.max_value:
                value = self.max_value
            if self.min_value is not None and value < self.min_value:
                value = self.min_value
        self._current_value = value

    @property
    def optimizable(self) -> bool:
        return self._optimizable

    @property
    def probabilities(self) -> List[float]:
        return self._probabilities

    @probabilities.setter
    def probabilities(self, value: Iterable[float]):
        assert (
            len(value) == len(self.discrete_values)
        ), f"Length of 'probabilities' ({len(value)}) must match length"
        f"of 'discrete values' ({len(self.discrete_values)})"
        prob_array = np.array(value, dtype=float)
        assert (
            np.all(prob_array >= 0)
        ), "All entries must be non-negative numerical values"
        assert (
            np.isclose(np.sum(prob_array), 1.0, atol=10E-8)
        ), f"Probabilities are not normed, their sum is {np.sum(prob_array)} but must equal 1"
        prob_array: np.ndarray = prob_array / np.sum(prob_array)
        self._probabilities: List[float] = prob_array.tolist()

    @property
    def weighted_cost(self) -> None | float:
        if self.cost is None:
            return None
        if self.discrete_values is None:
            return self.current_value * self.cost
        else:
            return float(np.dot(np.array(self.probabilities), np.array(self.cost)))


class SimulationParameterDictionary:
    """Dictionary containing all the parameters used by the simulation.

    Attributes:
    parameter_list: List[Type[SimulationParameter]]
    parameter_dict: Dict[str, Type[SimulationParameter]]

    Provides simple methods to easily write and read with the json format. Instances of this
    class have the following methods, with additional options listed in the method's docstring

      1. Accessing the information stored in the class:

        - Indexing as a list (in the same order as given during instantiation) or as a dict
            using the parameter's 'name'. Also usable with list or dict comprehension.
        - 'to_dict' returns a dict with the parameters also in a dict format. Alternatively,
            it returns a dict of 'SimulationParameter' (same as indexing this class by 'name')
            if you add the keyword argument 'serialized=False'.
        - 'to_json' will write all data from this class to a specified .json file in a dict
            format.
        - 'to_df' returns a pd.DataFrame of your parameters. More options are listed in the
            method's docstring.
        - 'get_current_values' returns all the current values of each parameter in a list or
            dict format.
        - 'get_probabilities' returns a dict whose values are the probabilities of each
            discrete parameter. More information about the usage of probabilities for one-hot
            encoded parameters is listed in the docstring of SimulationParameter.

      2. Instantiating this class from other objects:
        - 'from_dict' instantiates this class from a nested dict, where each entry is the dict
            representation of a SimulationParameter (which has a similar method)
        - 'from_json' instantiates this class from a json file. Specially useful in the combination
            with 'to_json' to transfer this class between programs.

      3. Changing the information stored:
        - 'update_current_values' will change all the current values of the class provided a
            simple dict [str, <update_value>].
        - 'update_probabilities' does the same but with all the probabilities of the discrete
            parameters of the class.
        - 'generate_new' will return a new instance of this class with new current values for each
            parameter

      4. Properties:
        - 'covariance' returns a diagonal matrix with the 'sigma' of every continuous parameter.
        - 'metadata' returns additional information about this class such as its creation time,
            the current iteration of the optimization process and user-defined descriptions.
    """

    def __init__(
            self,
            parameter_list: List[Type[SimulationParameter]] = [],
            ):
        """ Initialize with a list of parameters 'SimulationParameter'
        """
        self.iteration: int = 0
        self.creation_time = str(datetime.datetime.now())
        self.rng_seed: int | None = None
        self.description = ""
        self.parameter_list = parameter_list
        self.parameter_dict = self.to_dict(serialized=False)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def __setitem__(self, name: str, simulation_parameter: Type[SimulationParameter]):
        assert (
            name == simulation_parameter.name
        ), "Key does not match name assigned in 'SimulationParameter'"
        self.parameter_list.append(simulation_parameter)
        self.parameter_dict[name] = simulation_parameter

    def __getitem__(self, key: str | int) -> SimulationParameter:
        if isinstance(key, str):
            return self.parameter_dict[key]
        if isinstance(key, int):
            return self.parameter_list[key]

    def __len__(self) -> int:
        return len(self.parameter_list)

    def to_dict(self, serialized=True) -> Dict[str, SimulationParameter | Dict]:
        """Converts the parameter list to a dictionary.

        :param serialized: A boolean indicating whether to return a Dict of SimulationParameters or
            a Dict of Dict (by serializing each SimulationParameter in turn)
            If False, the SimulationParameter objects will be included as is. This is used by this class to allow
                dictionary-style access to the individual parameters\n
        :return: A dictionary where the keys are the names of the SimulationParameter objects and the values are either
            the serialized dictionaries or the SimulationParameter objects themselves.
        """
        if serialized is False:
            parameter_dict = {parameter.name: parameter for parameter in self.parameter_list}
        else:
            parameter_dict = {parameter.name: parameter.to_dict() for parameter in self.parameter_list}
        parameter_dict["metadata"] = self.metadata
        return parameter_dict

    def to_json(self, file_path: str):
        """Write the parameter list to a .json file

        TODO Check for the existence of the file path or otherwise set as default to ../
        """
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file)

    def to_df(
            self,
            df_length: int | None = 1,
            include_non_optimizables: bool = False,
            display_discrete: Literal["default", "as_probabilities", "as_one_hot"] = "default",
            types: Literal["all", "continuous", "discrete"] = "all",
            **kwargs
            ) -> pd.DataFrame:
        """ Convert parameter dictionary to a pd.DataFrame

        Args
        ----
            df_length (int): The length of the DataFrame to be created. Default is None.
            include_non_optimizables (bool): Whether to include non-optimizable parameters in the
                df. Defaults to False.
            display_discrete (Literal):
                - 'default': Simply write the current value of the Parameter (default)
                - 'as_probabilities': Write the probability of each category (from the list of available
                    discrete parameter).
                - 'as_one_hot':Write the current value as a one-hot encoded array. All categories are set
                    to zero except for the 'current_value' which is set to one.
                    Ref. https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
            types (str): Choose what kind of parameters will be added to the pd.DataFrame. All includes
                all parameters, continuous only those with no 'discrete_values' and discrete only
                those with 'discrete_values'.
            kwargs: Additional keyword arguments to be passed to the pd.DataFrame constructor.
        Return
        ------
            A pandas DataFrame containing the current parameter values.
        """
        if df_length is not None:
            kwargs["index"] = range(df_length)

        return pd.DataFrame(
            self.get_current_values("dict", include_non_optimizables, display_discrete=display_discrete, types=types),
            **kwargs,
        )

    def get_current_values(
            self,
            format: Literal["list", "dict"] = "dict",
            include_non_optimizables: bool = False,
            display_discrete: Literal["default", "as_probabilities", "as_one_hot"] = "default",
            types: Literal["all", "continuous", "discrete"] = "all",
            ) -> List | Dict:
        if format == "list" and display_discrete != "default":
            raise NotImplementedError("One-Hot Encoding is only available with the 'list' format")
        if types == "continuous" and display_discrete != "default":
            raise ValueError(
                "Continuous parameters can not be displayed as one-hot encoded, only discrete parameters."
                "Setting types == 'continuous' and 'display_discrete' other than 'default' are incompatible"
            )

        def get_parameters() -> Iterator[SimulationParameter]:
            for parameter in self.parameter_list:
                if (
                    not parameter.optimizable and not include_non_optimizables
                    or parameter.discrete_values is not None and types == "continuous"
                    or parameter.discrete_values is None and types == "discrete"
                ):
                    continue
                yield parameter

        if format == "list":
            current_values = []
            for parameter in get_parameters():
                current_values.append(parameter.current_value)

        elif format == "dict":
            current_values = {}

            for parameter in get_parameters():
                if parameter.discrete_values and not display_discrete == "default":

                    for index, val in enumerate(parameter.discrete_values):
                        key = f"{parameter.name}_{parameter.discrete_values[index]}"
                        if display_discrete == "as_probabilities":
                            current_values[key] = parameter.probabilities[index]
                        elif display_discrete == "as_one_hot":
                            current_values[key] = 1 if val == parameter.current_value else 0
                else:
                    current_values[parameter.name] = parameter.current_value

        return current_values

    def get_probabilities(self) -> dict[str, List[float]]:
        probabilities = {}

        for parameter in self.parameter_list:
            if parameter.discrete_values:
                probabilities[parameter.name] = parameter.probabilities

        return probabilities

    def update_current_values(self, current_values_parameter_dict: dict):
        for key, value in current_values_parameter_dict.items():
            assert (
                key in self.parameter_dict.keys()
            ), f"Key {key} was not in previous Parameter Dictionary"
            self.parameter_dict[key].current_value = value

        return self

    def update_probabilities(self, probabilities_dict: dict):
        for key, value in probabilities_dict.items():
            assert (
                key in self.parameter_dict.keys()
            ), f"Key {key} was not in previous Parameter Dictionary"

            if self.parameter_dict[key].discrete_values is not None:
                self.parameter_dict[key].probabilities = value

        return self

    @property
    def covariance(self) -> np.ndarray:
        covariance_matrix = []

        for parameter in self.parameter_list:
            if parameter.optimizable is True:
                covariance_matrix.append(parameter.sigma)

        return np.diag(np.array(covariance_matrix))

    @property
    def metadata(self) -> Dict[str, int | str]:
        return {
            "iteration": self.iteration,
            "creation_time": self.creation_time,
            "rng_seed": self.rng_seed,
            "description": self.description
        }

    @metadata.setter
    def metadata(self, new_metadata_dict: Dict[str, int | str]):
        for name, value in new_metadata_dict.items():
            if name in self.metadata:
                self.__setattr__(name, value)

    @classmethod
    def from_dict(cls, parameter_dict: Dict):
        """Create an instance from dictionary
        TODO Make sure it is a serialized dict, not a dict of SimulationParameters
        """
        metadata = parameter_dict.pop("metadata")
        instance = cls([SimulationParameter.from_dict(parameter) for parameter in parameter_dict.values()])
        instance.metadata = metadata
        return instance

    @classmethod
    def from_json(cls, file_path: str):
        """Create an instance from a .json file"""
        with open(file_path, "r") as file:
            parameter_dicts: Dict = json.load(file)
        return cls.from_dict(parameter_dicts)

    def generate_new(self, rng_seed: int | None = None):
        """
        Generates a new set of values for each parameter, bounded by specified minimum and maximum
        values for float parameters. For discrete parameters, the new value is randomly chosen from
        the list of allowed values.

        Args:
        ----
        rng_seed (int | None): Optional seed for the random number generator. If an integer is provided,
            the random number generator is initialized with that seed to ensure reproducibility. If None,
            a pseudo-random seed is generated based on the current time and the process ID, ensuring that
            each execution results in different random values:

            rng_seed = int(time.time()) + os.getpid()
        """

        def generate_continuous(parameter: SimulationParameter, retries: int = 100):
            if parameter.optimizable is False:
                return parameter.current_value

            for i in range(retries):
                new_value = rng.normal(parameter.current_value, parameter.sigma)

                if (
                    parameter.min_value is not None and parameter.current_value >= parameter.min_value
                    or parameter.max_value is not None and parameter.current_value <= parameter.max_value
                    or parameter.min_value is None and parameter.max_value is None
                ):
                    break
            else:
                new_value = parameter.current_value
                print(f"Warning: unable to set new current value for parameter {parameter.name}")
            return new_value

        def generate_discrete(parameter: SimulationParameter):
            return parameter.discrete_values[
                rng.choice(len(parameter.discrete_values), p=parameter.probabilities)
            ]

        rng = np.random.default_rng(rng_seed)
        rng_seed = rng_seed or int(rng.integers(9_999_999))
        new_parameter_list = copy.deepcopy(self.parameter_list)  # Prevents the modification of this instance

        for parameter in new_parameter_list:
            if parameter.optimizable is False:
                continue

            elif parameter.discrete_values is not None:
                parameter.current_value = generate_discrete(parameter)

            elif isinstance(parameter.current_value, float) and parameter.optimizable:
                parameter.current_value = generate_continuous(parameter)

        new_instance = type(self)(new_parameter_list)
        new_instance.metadata = self.metadata
        new_instance.rng_seed = rng_seed
        return new_instance
