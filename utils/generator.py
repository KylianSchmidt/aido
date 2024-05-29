import numpy as np
from utils.SimulationHelpers import SimulationParameterDictionary


class GenerateNewParameters():

    def __init__(self, file_path: str):
        """ Generate a new set of parameters given an input parameter dictionary
        """
        self.file_path = file_path
        self.param_dict = SimulationParameterDictionary.from_json(file_path)

    def increase_by_random_number(self, seed) -> SimulationParameterDictionary:
        """ Simple method that adds a random integer in [0, 1000] to the value of all parameter
        of type floats in the parameter dictionary.
        """
        for parameter in self.param_dict.parameter_list:
            if isinstance(parameter.current_value, float):
                rng = np.random.default_rng(seed)
                parameter.current_value += rng.integers(0, 1000)

        return self.param_dict
