from typing import Type, Dict, List, Iterable, Literal, Any
import json
import numpy as np
import pandas as pd
import random
from warnings import warn


class SimulationParameter:
    """Base class for all parameters used in the simulation

    TODO Write warnings in case the base class is used directly in the dictionary
    ref: https://stackoverflow.com/questions/46092104/subclass-in-type-hinting
    Update: dont know if this is necessary, this class has already most capabilities.

    TODO method to convert the parameter that is not a float to a float (discrete
    parameters for the surrogate model)

    TODO Make min and max also private attributes
    """

    def __init__(
            self,
            name: str,
            starting_value: Any,
            current_value: Any | None = None,
            units: str | None = None,
            optimizable=True,
            min_value: float | None = None,
            max_value: float | None = None,
            sigma: float | None = None,
            discrete_values: Iterable | None = None
            ):
        assert isinstance(name, str), "Name must be a string"

        self.name = name
        self._starting_value = starting_value
        self._optimizable = optimizable
        self.sigma = sigma
        self.units = units
        
        if min_value is not None:
            assert (
                isinstance(min_value, type(starting_value))
            ), "Only float parameters are allowed to have lower bounds"

        if max_value is not None:
            assert (
                isinstance(max_value, type(starting_value))
            ), "Only float parameters are allowed to have upper bounds"

        self.min_value = min_value
        self.max_value = max_value

        if current_value is not None:
            self._current_value = current_value
        else:
            self._current_value = starting_value

        assert (
            isinstance(self._current_value, float) is True or (discrete_values is not None or optimizable is False)
        ), "Specify discrete values when parameter is not float"

        if discrete_values is not None:
            if optimizable is False:
                raise AssertionError("Non-optimizable parameters are excluded from requiring allowed discrete values")
            assert (
                self._current_value in discrete_values
            ), "Current value must be included in the list of allowed discrete values"
            assert (
                starting_value in discrete_values
            ), "Starting value must be included in the list of allowed discrete values"
            
            assert (
                self.min_value is None and self.max_value is None
            ), "Not allowed to specify min and max value for parameter with discrete values"

        self.discrete_values = discrete_values

        if sigma is None:
            if discrete_values is None and optimizable is True:
                self.sigma = 0.1 * self.current_value
        else:
            assert (
                isinstance(sigma, float) and discrete_values is None and optimizable is True
            ), "Unable to asign standard deviation to discrete or non-optimizable parameter."

    def __str__(self):
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
    def current_value(self):
        return self._current_value

    @current_value.setter
    def current_value(self, value):
        assert (
            isinstance(value, type(self._starting_value))
        ), f"The updated value is of another type ({type(value)}) than before ({type(self._starting_value)})"
        if self._optimizable is False:
            warn("Do not change the current value of non-optimizable parameter")
        self._current_value = value

    @property
    def optimizable(self):
        return self._optimizable


class SimulationParameterDictionary:
    """Dictionary containing the list of parameters used by the simulation.

    Attributes:
    parameter_list: List[Type[SimulationParameter]]

    Provides IO methods to easily write and read with json format.

    TODO Additional information such as current iteration number, date of creation, etc...
    """

    def __init__(
            self,
            parameter_list: List[Type[SimulationParameter]] = [],
            ):
        """Initialize a list of parameters"""
        self.parameter_list = parameter_list
        self.parameter_dict = self.to_dict(serialized=False)

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)

    def __setitem__(self, name: str, simulation_parameter: Type[SimulationParameter]):
        assert name == simulation_parameter.name, "Key does not match name assigned in 'SimulationParameter'"
        self.parameter_list.append(simulation_parameter)
        self.parameter_dict[name] = simulation_parameter

    def __getitem__(self, key: str) -> SimulationParameter:
        return self.parameter_dict[key]

    def __len__(self) -> int:
        return len(self.parameter_list)

    def to_dict(self, serialized=True) -> Dict[str, SimulationParameter] | List[SimulationParameter]:
        """Converts the parameter list to a dictionary.

        :param serialized: A boolean indicating whether to serialize the SimulationParameter objects. \n
            If True, the SimulationParameter objects will be converted to dictionaries using their `to_dict` method.
                Used for example to generate json-readable strings.\n
            If False, the SimulationParameter objects will be included as is. This is used by this class to allow
                dictionary-style access to the individual parameters\n
        :return: A dictionary where the keys are the names of the SimulationParameter objects and the values are either
            the serialized dictionaries or the SimulationParameter objects themselves.
        """
        names = [parameter.name for parameter in self.parameter_list]

        if serialized is False:
            parameter_dicts = [parameter for parameter in self.parameter_list]
        else:
            parameter_dicts = [parameter.to_dict() for parameter in self.parameter_list]
        return dict(zip(names, parameter_dicts))

    def to_json(self, file_path: str):
        """Write the parameter list to a .json file

        TODO Check for the existence of the file path or otherwise set as default to ../
        """
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file)

    def to_df(self, df_length: int = 1) -> pd.DataFrame:
        """ Create parameter dict from file if path given. Remove all parameters that are not
        optimizable and also only keep current values. Output is a df of length 'df_length', so
        that it can be concatenated with the other df's.
        """
        return pd.DataFrame(self.get_current_values("dict"), index=range(df_length))

    def get_current_values(
            self,
            format: Literal["list", "dict"] = "dict",
            include_non_optimizables=False
            ):
        if format == "list":
            current_values = []

            for parameter in self.parameter_list:
                if parameter.optimizable is True or include_non_optimizables is True:
                    current_values.append(parameter.current_value)

        elif format == "dict":
            current_values = {}

            for parameter in self.parameter_list:
                if parameter.optimizable is True or include_non_optimizables is True:
                    current_values[parameter.name] = parameter.current_value

        return current_values
    
    def update_current_values(self, current_values_parameter_dict: dict):
        for key, value in current_values_parameter_dict.items():
            assert key in self.parameter_dict.keys(), f"Key {key} was not in previous Parameter Dictionary"
            self.parameter_dict[key].current_value = value

        return self
    
    @property
    def covariance(self):
        covariance_matrix = []

        for parameter in self.parameter_list:
            if parameter.optimizable is True:
                covariance_matrix.append(parameter.sigma)

        return np.diag(np.array(covariance_matrix))

    @classmethod
    def from_dict(cls, parameter_dict: Dict):
        """Create an instance from dictionary"""
        instance = cls(
            [SimulationParameter.from_dict(parameter) for parameter in parameter_dict]
        )
        return instance

    @classmethod
    def from_json(cls, file_path: str):
        """Create an instance from a .json file"""
        with open(file_path, "r") as file:
            parameter_dicts: Dict = json.load(file)
            return cls.from_dict(parameter_dicts.values())

    def generate_new(self, rng_seed: int = 42):
        """ Generate a set of new values for each parameter, bounded by the min_value and max_value
        for float parameters. For discrete parameters, the new current_value is randomly chosen from
        the list of allowed values.
        TODO Decrease sigma if unable to find a new current_value
        """
        new_parameter_list = self.parameter_list

        for parameter in new_parameter_list:
            if parameter.optimizable is False:
                continue

            elif parameter.discrete_values is not None:
                parameter.current_value = random.choice(parameter.discrete_values)
                continue

            elif isinstance(parameter.current_value, float):
                rng = np.random.default_rng(rng_seed)

                for i in range(100):
                    parameter.current_value = rng.normal(parameter.current_value, parameter.sigma)

                    if parameter.min_value is not None:
                        if parameter.current_value >= parameter.min_value:
                            break
                    if parameter.max_value is not None:
                        if parameter.current_value <= parameter.max_value:
                            break
                    if parameter.min_value is None and parameter.max_value is None:
                        break
                else:
                    print(f"Warning: unable to set new current value for parameter {parameter}")

        return SimulationParameterDictionary(new_parameter_list)


if __name__ == "__main__":
    sim_param_dict = SimulationParameterDictionary([
        SimulationParameter("absorber_thickness", 10.0, min_value=0.5, max_value=40.0),
        SimulationParameter("absorber_material", "LEAD", discrete_values=["LEAD", "TUNGSTEN"]),
        SimulationParameter("energy", 1000, optimizable=False),
        SimulationParameter("num_absorber_plates", 2, discrete_values=list(range(0, 5))),
    ])

    sim_param_dict.to_json("./sim_param_dict")

    sim_param_dict_2 = SimulationParameterDictionary.from_json("./sim_param_dict")
    print(sim_param_dict_2)
    print("Covariance matrix:\n", sim_param_dict_2.covariance)

    sim_param_dict_3 = sim_param_dict.generate_new()
    print(sim_param_dict_3)
