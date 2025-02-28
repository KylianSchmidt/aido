import numpy as np
import pytest
import torch

import aido
from aido.optimization_helpers import ParameterModule


@pytest.fixture
def parameter_dict():
    return aido.SimulationParameterDictionary([
        aido.SimulationParameter(
            "thickness_absorber",
            5.6,
            min_value=0.01,
            max_value=20.0,
            sigma=0.2,
            cost=1.1,
        ),
        aido.SimulationParameter(
            "thickness_scintillator",
            0.1,
            min_value=0.05,
            max_value=1.0,
            sigma=0.2,
            cost=5.0,
        ),
        aido.SimulationParameter(
            "absorber_material",
            "G4_Pb",
            discrete_values=["G4_Pb", "G4_W", "G4_Fe"],
            cost=[1.3, 0.26, 0.092],
        ),
        aido.SimulationParameter(
            "num_blocks",
            3,
            discrete_values=list(range(1, 10)),
            cost=(0.1 * np.arange(1, 10)).tolist(),
        ),
        aido.SimulationParameter(
            "thickness_absorber_1",
            13.9,
            min_value=0.0,
            max_value=20.0,
            sigma=1.5,
            cost=1.1,
        ),
        aido.SimulationParameter(
            "thickness_scintillator_1",
            15.7,
            min_value=0.05,
            max_value=1.0,
            sigma=0.2,
            cost=5.0,
        ),
        aido.SimulationParameter("num_events", 200, optimizable=False),
    ])


def test_instantiation(parameter_dict: aido.SimulationParameterDictionary):
    ParameterModule(parameter_dict)


def test_ordering(parameter_dict: aido.SimulationParameterDictionary):
    parameter_module = ParameterModule(parameter_dict)
    assert round(parameter_module.continuous_tensors()[1].item(), 2) == 0.1
    assert round(parameter_module.continuous_tensors()[2].item(), 2) == 13.9


def test_probabilities(parameter_dict: aido.SimulationParameterDictionary):
    parameter_module = ParameterModule(parameter_dict)
    parameter_module["absorber_material"].logits.data = torch.tensor([0.1, 0.2, 0.7])
    parameter_dict.update_probabilities(parameter_module.probabilities)
    assert np.all(parameter_dict["absorber_material"] != [1 / 3, 1 / 3, 1 / 3])


def test_probabilities_previously_set(parameter_dict: aido.SimulationParameterDictionary):
    parameter_dict["absorber_material"].probabilities = [0.3, 0.4, 0.3]
    parameter_module = ParameterModule(parameter_dict)
    parameter_dict.update_probabilities(parameter_module.probabilities)
    assert (
        np.all(np.round(parameter_dict["absorber_material"].probabilities, decimals=1) == [0.3, 0.4, 0.3])
    ), f"DEBUG {parameter_module.probabilities=}"


def test_boundaries(parameter_dict: aido.SimulationParameterDictionary):
    parameter_module = ParameterModule(parameter_dict)
    assert np.all(
        parameter_module["thickness_scintillator_1"].boundaries.numpy().astype(np.float32)
        == np.array([(-0.2 + 0.05) / 1.1, (0.2 + 1.0) / 1.1], np.float32)
    )
    assert np.all(
        parameter_module["thickness_absorber_1"].boundaries.numpy().astype(np.float32)
        == np.array([-1.5 / 1.1, (1.5 + 20.0) / 1.1], np.float32)
    )
