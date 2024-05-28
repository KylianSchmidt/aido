import numpy as np
from utils.SimulationHelpers import SimulationParameterDictionary


class GenerateNewParameters():

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.param_dict = SimulationParameterDictionary.from_json(file_path)

    def increase_by_random_number(self):
        for parameter in self.param_dict.parameter_list:
            if isinstance(parameter.current_value, float):
                parameter.current_value += np.random.randint(0, 1000)

        return self.param_dict
