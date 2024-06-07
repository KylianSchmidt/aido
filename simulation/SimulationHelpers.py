from typing import Type, Dict, List
import json
import numpy as np
from warnings import warn


class SimulationParameter():
    """ Base class for all parameters used in the simulation

    TODO Write warnings in case the base class is used directly in the dictionary
    ref: https://stackoverflow.com/questions/46092104/subclass-in-type-hinting
    """

    def __init__(self, name: str, starting_value, current_value=None, optimizable=True):
        """ Initialize a new general parameter
        """
        assert isinstance(name, str), "Name must be a string"

        self.name = name
        self._starting_value = starting_value
        self._optimizable = optimizable

        if current_value is not None:
            self.current_value = current_value
        else:
            self.current_value = starting_value

    def to_dict(self) -> Dict:
        """ Convert to dictionary

        Protected attributes are written to file as public attributes.
        """
        return {
            key.removeprefix("_"): value for key, value in self.__dict__.items()
            }

    @classmethod
    def from_dict(cls, attribute_dict: Dict):
        """ Create from dictionary
        """
        return cls(**attribute_dict)

    @property
    def current_value(self):
        return self._current_value

    @current_value.setter
    def current_value(self, value):
        assert isinstance(value, type(self._starting_value)), \
            f"The updated value is of another type ({type(value)}) " + \
            f"than the starting value ({type(self._starting_value)})"
        if self._optimizable is False:
            pass
        self._current_value = value


class SimulationParameterDictionary():
    """ Dictionary containing the list of parameters used by the simulation.

    Attributes:
    parameter_list: List[Type[SimulationParameter]]

    Provides IO methods to easily write and read with json format.
    """

    def __init__(self, parameter_list: List[Type[SimulationParameter]] = []):
        """ Initialize an empty list with no parameters
        """
        self.parameter_list = parameter_list

    def add_parameter(self, simulation_parameter: Type[SimulationParameter]):
        """ Add a parameter to the dictionary
        """
        self.parameter_list.append(simulation_parameter)

    def to_dict(self) -> Dict:
        """ Converts to dictionary

        TODO Is a dict of list the optimal way to print the contents of the class?
        """
        return {"Parameters": [parameter.to_dict() for parameter in self.parameter_list]}

    def to_json(self, file_path: str):
        """ Write the parameter list to a .json file

        TODO Check for the existence of the file path or otherwise set as default to ../
        """
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file)

    @classmethod
    def from_dict(cls, parameter_dict: Dict):
        """ Create an instance from dictionary
        """
        instance = cls([
                SimulationParameter.from_dict(parameter) for parameter in parameter_dict["Parameters"]
            ])
        return instance

    @classmethod
    def from_json(cls, file_path: str):
        """ Create an instance from a .json file
        """
        with open(file_path, "r") as file:
            return cls.from_dict(json.load(file))


class GatherResults():

    def from_numpy(file_paths: List[str], **kwargs):
        """Combine the output files from the reconstruction Task into one 2D numpy array.

        Args:
            file_paths (List[str]): A list of file paths to the output files.
            **kwargs: Additional keyword arguments to be passed to `np.fromfile` function.

        Returns:
            numpy.ndarray: A 2D numpy array containing the combined data from all the output files.
        """
        reconstruction_array_all_tasks = []

        for file_path in file_paths:
            arr = np.fromfile(file_path, **kwargs)
            if arr.ndim > 1:
                warn(
                    f"Reconstruction output array has {arr.ndim} dimensions, but should only have 1. Array \
                    will be flattened in order to have the correct shape."
                )
                arr = arr.flatten()
            reconstruction_array_all_tasks.append(arr)

        return np.array(reconstruction_array_all_tasks)


if __name__ == "__main__":
    param_foo = SimulationParameter("foo", 1.0)
    param_bar = SimulationParameter("bar", "LEAD")
    sim_param_dict = SimulationParameterDictionary(
        [param_foo, param_bar]
    )

    sim_param_dict.to_json("./sim_param_dict")

    sim_param_dict_2 = SimulationParameterDictionary.from_json("./sim_param_dict")

    print(sim_param_dict_2.to_dict())

    sim_param_dict_2.parameter_list[0].current_value = 3.0
    sim_param_dict_2.parameter_list[1].current_value = "TUNGSTEN"

    print(sim_param_dict_2.to_dict())
