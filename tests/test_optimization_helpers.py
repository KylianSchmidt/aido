import pytest

from modules.optimization_helpers import ParameterModule
from modules.simulation_helpers import SimulationParameterDictionary


@pytest.fixture
def parameter_dict():
    return SimulationParameterDictionary.from_json(
        "results_old/results_discrete_20241014/parameters/param_dict_iter_0.json"
    )


def test_instantiation(parameter_dict):
    ParameterModule(parameter_dict)
