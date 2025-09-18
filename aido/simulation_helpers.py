import copy
import datetime
import json
from typing import Any, Dict, Iterable, Iterator, List, Literal, Type

import numpy as np
import pandas as pd

from aido.config import AIDOConfig

config = AIDOConfig.from_json("config.json")


class SimulationParameter:
    """Base class for all parameters used in the simulation.
    
    A simulation parameter represents a single variable that can be optimized
    during the simulation process.
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
            sigma_mode: Literal["flat", "scale"] | None = None,
            discrete_values: Iterable | None = None,
            probabilities: Iterable[float] | None = None,
            cost: float | Iterable | None = None
            ):
        """Create a new Simulation Parameter.

        Parameters
        ----------
        name : str
            The name of the parameter.
        starting_value : Any
            The starting value of the parameter.
        current_value : Any, optional
            The current value of the parameter. Defaults to None, in which case
            it set to the starting value instead.
        units : str, optional
            The units of the parameter. Defaults to None.
        optimizable : bool, optional
            Whether the parameter is optimizable. Defaults to True.
        min_value : float, optional
            The minimum value of the parameter. Defaults to None.
        max_value : float, optional
            The maximum value of the parameter. Defaults to None.
        sigma : float, optional
            The standard deviation of the parameter. Defaults to 0.5 for continuous
            (float) parameters and remains None otherwise.
        sigma_mode : {"flat", "scale"}, optional
            Whether to set the sampling distribution standard deviation to sigma ("flat")
            or scale sigma with the current value ("scale"). Defaults to "flat". If "sigma"
            is not defined, this parameter has no action. Can be changed class-wide using
            the cls.set_sigma_mode() classmethod.
        discrete_values : Iterable, optional
            The allowed discrete values of the parameter. Defaults to None.
        probabilities : Iterable[float], optional
            A list of the same length as 'discrete_values' used to sample from
            'discrete_values'. If set to None, an equally-distributed array is created.
            Only for discrete parameters. Defaults to None.
        cost : float or Iterable, optional
            A float that quantifies the cost per unit of this Parameter.
            Defaults to None. For discrete parameters, this parameter must be an
            Iterable (e.g. list) of the same length as 'discrete_values'.
        """
        def check_boundaries() -> None:
            assert (
                isinstance(min_value, type(starting_value)) or min_value is None
                ), "Only float parameters are allowed to have lower bounds"

            assert (
                isinstance(max_value, type(starting_value)) or max_value is None
            ), "Only float parameters are allowed to have upper bounds"

        def check_discrete_parameters() -> None:
            assert (
                isinstance(self._current_value, float) is True or (discrete_values is not None or optimizable is False)
            ), "Specify discrete values when the parameter is not a float, but is optimizable"

            if discrete_values is not None:
                if optimizable is False:
                    raise AssertionError(
                        "Non-optimizable parameters are excluded from requiring allowed discrete values"
                    )
                assert (
                    self._current_value in discrete_values
                ), "Current value must be included in the list of allowed discrete values"
                assert (
                    starting_value in discrete_values
                ), "Starting value must be included in the list of allowed discrete values"
                assert (
                    min_value is None and max_value is None
                ), "Not allowed to specify min and max value for parameter with discrete values"

        def check_sigma() -> None:
            if discrete_values is None and optimizable:
                if sigma is not None:
                    assert sigma > 0.0, "Sigma parameter must be a positive float"
            else:
                assert (
                    sigma is None
                ), "Unable to assign standard deviation to discrete or non-optimizable parameter."

        def check_cost() -> None:
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

        assert isinstance(name, str), "Name must be a string"
        self.name = name

        check_boundaries()
        self._starting_value = starting_value
        self._optimizable = optimizable
        self.units = units
        self.min_value = min_value
        self.max_value = max_value
        self._current_value = current_value if current_value is not None else starting_value

        check_discrete_parameters()
        self.discrete_values = discrete_values
        self.probabilities = probabilities if discrete_values is not None else None

        check_sigma()
        self.sigma = sigma
        self.sigma_mode = sigma_mode or config.simulation.sigma_mode

        check_cost()
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
        self._current_value = value

    @property
    def optimizable(self) -> bool:
        return self._optimizable
    
    @property
    def sigma(self) -> float | None:
        if self.discrete_values is not None or not self.optimizable:
            return None
        if self.sigma_mode == "scale":
            return self._sigma * self.current_value
        elif self.sigma_mode == "flat":
            return self._sigma

    @sigma.setter
    def sigma(self, value: float | None):
        if self.discrete_values is None and self.optimizable:
            if value is None:
                value = config.simulation.sigma
            assert value > 0.0
        else:
            assert value is None
        self._sigma = value

    @property
    def probabilities(self) -> List[float]:
        return self._probabilities

    @probabilities.setter
    def probabilities(self, value: Iterable[float] | None):
        if self.discrete_values is None:
            return None
        if value is None:
            value = np.full(len(self.discrete_values), 1.0 / len(self.discrete_values))
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
    -----------
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
        self._covariance: np.ndarray | None = None
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
        """Convert the parameter list to a dictionary.

        Parameters
        ----------
        serialized : bool, default=True
            If True, returns a Dict of Dict by serializing each SimulationParameter.
            If False, returns a Dict of SimulationParameters, which is used by this class
            to allow dictionary-style access to the individual parameters.

        Returns
        -------
        Dict[str, Union[SimulationParameter, Dict]]
            A dictionary where the keys are parameter names and the values are either
            serialized dictionaries or SimulationParameter objects.
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
                - 'as_one_hot': Write the current value as a one-hot encoded array. All categories are set
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

    def update_probabilities(self, probabilities_dict: dict[str, List | np.ndarray]):
        for key, value in probabilities_dict.items():
            assert (
                key in self.parameter_dict.keys()
            ), f"Key {key} was not in previous Parameter Dictionary"

            if self.parameter_dict[key].discrete_values is not None:
                self.parameter_dict[key].probabilities = value

        return self
    
    @property
    def sigma_array(self) -> np.ndarray:
        """ Diagonal matrix with the standard deviation of each continuous parameter
        """
        sigma_array = []

        for parameter in self.parameter_list:
            if parameter.optimizable is True and parameter.discrete_values is None:
                sigma_array.append(parameter.sigma**2)

        return np.diag(np.array(sigma_array))

    @property
    def covariance(self) -> np.ndarray:
        """ Get the current covariance matrix. If no covariance was set, it defaults
        to the 'sigma_array' squared (covariance matrix with no correlations).
        """
        if self._covariance is not None:
            return self._covariance
        else:
            return self.sigma_array

    @covariance.setter
    def covariance(self, new_covariance=np.ndarray) -> None:
        """ Set the input matrix as the new covariance matrix
        """
        assert (
            isinstance(new_covariance, np.ndarray)
        ), f"Covariance must be a numpy array, but is type {type(new_covariance)}"
        assert (
            np.allclose(new_covariance, new_covariance.T)
        ), "Covariance matrix is not symmetric"
        assert (
            np.all(np.linalg.eigvals(new_covariance) >= 0)
        ), "Covariance matrix is not positive semi-definite (negative eigenvalues detected)"
        assert (
            np.all(np.diag(new_covariance) >= 0)
        ), "Covariance matrix has negative variances on the diagonal"
        self._covariance = new_covariance

    @property
    def metadata(self) -> Dict[str, int | str | List]:
        return {
            "iteration": self.iteration,
            "creation_time": self.creation_time,
            "rng_seed": self.rng_seed,
            "description": self.description,
            "covariance": self.covariance.tolist()
        }

    @metadata.setter
    def metadata(self, new_metadata_dict: Dict[str, int | str]):
        if "covariance" in new_metadata_dict.keys():
            self.covariance = np.array(new_metadata_dict.pop("covariance"))

        for name, value in new_metadata_dict.items():
            if name in self.metadata:
                self.__setattr__(name, value)

    @classmethod
    def from_dict(cls, parameter_dict: Dict):
        """Create an instance from dictionary
        TODO Make sure it is a serialized dict, not a dict of SimulationParameters
        """
        metadata: Dict = parameter_dict.pop("metadata")
        instance = cls([SimulationParameter.from_dict(parameter) for parameter in parameter_dict.values()])
        instance.metadata = metadata
        return instance

    @classmethod
    def from_json(cls, file_path: str):
        """Create an instance from a .json file"""
        with open(file_path, "r") as file:
            parameter_dicts: Dict = json.load(file)
        return cls.from_dict(parameter_dicts)

    def generate_new(
            self,
            rng_seed: int | None = None,
            discrete_index: int | None = None,
            scaling_factor: float = 1.0
            ):
        """Generates a new set of values for each parameter.
        
        Generates new values bounded by specified minimum and maximum values for float
        parameters. For discrete parameters, the new value is randomly chosen from the
        list of allowed values.

        Parameters
        ----------
        rng_seed : int or None, optional
            Seed for the random number generator. If an integer is provided,
            the random number generator is initialized with that seed to ensure
            reproducibility. If None, a pseudo-random seed is generated.
        discrete_index : int or None, optional
            Optional indexing of the discrete parameters categories to ensure
            that a certain category is represented. If 'discrete_index' >
            len(parameter.discrete_values), random sampling is used. Otherwise
            the chosen value is parameter.discrete_values['discrete_index'].
        scaling_factor : float, default=1.0
            Factor to scale the covariance matrix by when generating new values.
        """

        def generate_discrete(parameter: SimulationParameter):
            if parameter.discrete_values is None:
                return None
            if discrete_index is not None and discrete_index < len(parameter.discrete_values):
                return parameter.discrete_values[discrete_index]
            else:
                return parameter.discrete_values[
                    rng.choice(len(parameter.discrete_values), p=parameter.probabilities)
                ]

        rng = np.random.default_rng(rng_seed)
        rng_seed = rng_seed or int(rng.integers(9_999_999))
        new_instance = copy.deepcopy(self)  # Prevents the modification of this instance

        for parameter in self.parameter_list:
            if parameter.discrete_values is not None:
                new_instance[parameter.name].current_value = generate_discrete(parameter)

        new_continuous_parameter_list = rng.multivariate_normal(
            mean=np.array(self.get_current_values(format="list", types="continuous")),
            cov=self.covariance * scaling_factor
        )

        for index, key in enumerate(self.get_current_values(types="continuous").keys()):
            new_instance[key].current_value = float(new_continuous_parameter_list[index])

        new_instance.metadata = self.metadata
        new_instance.rng_seed = rng_seed
        return new_instance
