from typing import Type


class SimulationParameter():

    def __init__(self, starting_value):
        """ Initialize a new general parameter
        """
        self.starting_value = starting_value


class SimulationParameterDictionary():

    def __init__(self):
        """ Initialize an empty dictionary with no parameters
        """


class SimulationIO():

    def __init__(self, parameter_dictionary: Type[SimulationParameterDictionary]):
        """ Store the dictionary with the simulation parameters
        """

    def write_json():
        """ Write the dictionary of parameters to a json file for safekeeping


        """

    def read_json():
        """ Read the dictionary of parameters
        """


if __name__ == "__main__":
    param_foo = SimulationParameter(1.0)
    sim_param_dict = SimulationParameterDictionary()
    