from typing import Type, Dict, List
import json


class SimulationParameter():
    """ Base class for all parameters used in the simulation

    TODO Write warnings in case the base class is used directly in the dictionary
    ref: https://stackoverflow.com/questions/46092104/subclass-in-type-hinting
    """

    def __init__(self, starting_value):
        """ Initialize a new general parameter
        """
        self.starting_value = starting_value

    def to_dict(self) -> Dict:
        """ Convert to dictionary
        """
        return {
            "starting_value": self.starting_value
        }

    @classmethod
    def from_dict(cls, parameter_dict: Dict) -> Dict:
        """ Create from dictionary
        """
        return cls(**parameter_dict)


class SimulationParameterDictionary():

    def __init__(self):
        """ Initialize an empty list with no parameters
        """
        self.parameter_list: List[Type[SimulationParameter]] = []

    def add_parameter(self, simulation_parameter: Type[SimulationParameter]):
        """ Add a parameter to the dictionary 
        """
        self.parameter_list.append(simulation_parameter)

    def to_dict(self) -> Dict:
        """ Converts to dictionary
        """
        return {"Parameters": [parameter.to_dict() for parameter in self.parameter_list]}

    def from_dict(self):
        """ Create an instance from dictionary
        """

if __name__ == "__main__":
    param_foo = SimulationParameter(1.0)
    sim_param_dict = SimulationParameterDictionary()
    