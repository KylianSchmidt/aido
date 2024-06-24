from typing import Type, Dict, List, Iterable, Literal, Any
import json
import numpy as np
from warnings import warn


class SimulationParameter:
    """Base class for all parameters used in the simulation

    TODO Write warnings in case the base class is used directly in the dictionary
    ref: https://stackoverflow.com/questions/46092104/subclass-in-type-hinting
    Update: dont know if this is necessary, this class has already most capabilities.

    TODO method to convert the parameter that is not a float to a float (discrete
    parameters for the surrogate model)
    """

    def __init__(
            self,
            name: str,
            starting_value: Any,
            current_value: Any | None = None,
            optimizable=True,
            min_value: float | None = None,
            max_value: float | None = None,
            discrete_values: Iterable | None = None
            ):
        assert isinstance(name, str), "Name must be a string"

        self.name = name
        self._starting_value = starting_value
        self._optimizable = optimizable
        
        if min_value is not None:
            assert (
                isinstance(min_value, type(starting_value))
            ), "Only float parameters are allowed to have lower bounds"

        if max_value is not None:
            assert (
                isinstance(max_value, type(starting_value))
            ), "Only float parameters are allowed to have upper bounds"
    
        self._min_value = min_value
        self._max_value = max_value

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
            else:
                assert (
                    self._current_value in discrete_values
                ), "Current value must be included in the list of allowed discrete values"
            assert (
                starting_value in discrete_values
            ), "Starting value must be included in the list of allowed discrete values"
            
            assert (
                self._min_value is None and self._max_value is None
            ), "Not allowed to specify min and max value for parameter with discrete values"

            self.discrete_values = discrete_values

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
        assert isinstance(value, type(self._starting_value)), (
            f"The updated value is of another type ({type(value)}) "
            + f"than the starting value ({type(self._starting_value)})"
        )
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

    def __setitem__(self, name, simulation_parameter: Type[SimulationParameter]):
        assert name == simulation_parameter.name, "Key does not match name assigned in 'SimulationParameter'"
        self.parameter_list.append(simulation_parameter)
        self.parameter_dict[name] = simulation_parameter

    def __getitem__(self, key: str) -> SimulationParameter:
        return self.parameter_dict[key]

    def __len__(self) -> int:
        return len(self.parameter_list)

    def to_dict(self, serialized=True) -> Dict[str, SimulationParameter]:
        """Converts the parameter list to a dictionary.

        :param serialized: A boolean indicating whether to serialize the SimulationParameter objects. \n
            If True, the SimulationParameter objects will be converted to dictionaries using their `to_dict` method. Used \
                for example to generate json-readable strings.\n
            If False, the SimulationParameter objects will be included as is. This is used by this class to allow \
                dictionary-style access to the individual parameters\n
        :return: A dictionary where the keys are the names of the SimulationParameter objects and the values are either \
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

    def get_current_values(
        self, format: Literal["list", "dict"] = "dict", include_non_optimizables=False
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


class GatherResults:

    def from_numpy_files(file_paths: List[str], **kwargs):
        """Combine the output files from the reconstruction Task into one 2D numpy array.

        Args:
            file_paths (List[str]): A list of file paths to the output files. The file paths must end
            with .npy (not .npz).
            **kwargs: Additional keyword arguments to be passed to `np.fromfile` function.

        Returns:
            numpy.ndarray: A 2D numpy array containing the combined data from all the output files.
        """
        reconstruction_array_all_tasks = []

        for file_path in file_paths:
            if file_path.endswith(".npy"):
                arr = np.load(file_path)
            else:
                arr = np.loadtxt(file_path, **kwargs)
            if arr.ndim > 1:
                warn(
                    f"Reconstruction output array has {arr.ndim} dimensions, but should only have 1. Array "
                    + "will be flattened in order to have the correct shape."
                )
                arr = arr.flatten()
            reconstruction_array_all_tasks.append(arr)

        return np.array(reconstruction_array_all_tasks)

    def from_parameter_dicts(file_paths: List[str]):
        """For all parameter dicts found at 'file_paths', construct a nested list (2D) with
        all their optimizable parameters.

        Args:
            file_paths (List[str]): A list of file paths containing parameter dictionaries as .json.

        Returns:
            List[List]]: A nested list containing the optimizable parameters from all
            the parameter dictionaries.

        TODO implement control to only include floats to this array for best surrogate model handling.
        """
        parameter_list = []

        for file_path in file_paths:
            param_dict = SimulationParameterDictionary.from_json(file_path)
            parameter_list.append(param_dict.get_current_values(format="list"))

        return parameter_list


if __name__ == "__main__":
    sim_param_dict = SimulationParameterDictionary(
        [
            SimulationParameter("absorber_thickness", 10.0, min_value=0.5, max_value=40.0),
            SimulationParameter("absorber_material", "LEAD", discrete_values=["LEAD", "TUNGSTEN"]),
            SimulationParameter("energy", 1000, optimizable=False),
            SimulationParameter("num_absorber_plates", 2, discrete_values=list(range(0, 5))),
        ]
    )

    sim_param_dict.to_json("./sim_param_dict")

    sim_param_dict_2 = SimulationParameterDictionary.from_json("./sim_param_dict")
    print(sim_param_dict_2)
