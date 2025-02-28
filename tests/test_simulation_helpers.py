
import os

import numpy as np
import pytest

import aido
import aido.config

config = aido.config.AIDOConfig.from_json("config.json")


@pytest.fixture
def sim_param_dict() -> aido.SimulationParameterDictionary:
    return aido.SimulationParameterDictionary([
        aido.SimulationParameter("absorber_thickness", 10.0, min_value=0.5, max_value=40.0),
        aido.SimulationParameter("absorber_material", "LEAD", discrete_values=["LEAD", "TUNGSTEN", "BREAD", "WOOD"]),
        aido.SimulationParameter("energy", 1000, optimizable=False),
        aido.SimulationParameter("num_absorber_plates", 2, discrete_values=list(range(0, 5))),
    ])


def test_save_load(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict.to_json("./sim_param_dict")
    aido.SimulationParameterDictionary.from_json("./sim_param_dict")
    os.remove("./sim_param_dict")


def test_generate_new(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict_2 = sim_param_dict.generate_new()
    sim_param_dict_3 = sim_param_dict.generate_new()
    assert isinstance(sim_param_dict_2.rng_seed, int)
    assert sim_param_dict_2.rng_seed != sim_param_dict_3.rng_seed
    sim_param_dict_4 = sim_param_dict.generate_new(discrete_index=0)
    assert sim_param_dict_4["absorber_material"].current_value == "LEAD"
    sim_param_dict_4 = sim_param_dict.generate_new(discrete_index=1)
    assert sim_param_dict_4["absorber_material"].current_value == "TUNGSTEN"
    sim_param_dict.generate_new(discrete_index=6)


def test_update_metadata(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict.iteration = 17
    assert sim_param_dict.iteration == 17


def test_to_df(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict.to_df(types="all", display_discrete="as_one_hot")
    sim_param_dict.to_df(types="discrete", display_discrete="as_one_hot")
    with pytest.raises(ValueError):
        sim_param_dict.to_df(types="continuous", display_discrete="as_one_hot")
    with pytest.raises(NotImplementedError):
        sim_param_dict.get_current_values(format="list", display_discrete="as_one_hot")


def test_cost() -> None:
    with pytest.raises(AssertionError):
        aido.SimulationParameter("foo", 1.0, cost=-10)
        aido.SimulationParameter("foo", 1.0, cost=[0.1, 2.0])
        aido.SimulationParameter("foo", 1, discrete_values=[1, 2, 3], cost=1.7)


def test_weighted_cost() -> None:
    assert (
        aido.SimulationParameter("foo", 2.3, cost=7.2).weighted_cost == 2.3 * 7.2
    )
    assert (
        aido.SimulationParameter(
            "foo", "a", discrete_values=["a", "b"], cost=[1.8, 0.5]
        ).weighted_cost == 0.5 * 1.8 + 0.5 * 0.5
    )


def test_display_discrete(sim_param_dict: aido.SimulationParameterDictionary) -> None:
    df = sim_param_dict.to_df(display_discrete="as_one_hot")
    df["absorber_material_LEAD"][0] == 1.0
    df["absorber_material_TUNGSTEN"][0] == 0.0

    df = sim_param_dict.to_df(display_discrete="as_probabilities")
    df["absorber_material_LEAD"][0] == (
        df["absorber_material_TUNGSTEN"][0] == 1.0 / len(sim_param_dict["absorber_material"].discrete_values)
    )


def test_sigma_mode() -> None:
    sim_param_dict = aido.SimulationParameter("foo", 0.0)
    assert sim_param_dict.sigma_mode == "flat"

    sim_param_dict = aido.SimulationParameter("foo", 0.0, sigma_mode="scale")
    assert sim_param_dict.sigma_mode == "scale"


def test_sigma() -> None:
    sim_param_dict = aido.SimulationParameter("foo", 1.0)
    assert sim_param_dict.sigma == config.simulation.sigma

    sim_param_dict = aido.SimulationParameter("foo", 4.0, sigma_mode="scale", sigma=1.5)
    assert sim_param_dict.sigma == 6.0

    sim_param_dict = aido.SimulationParameter("foo", "LEAD", discrete_values=["LEAD"])
    assert sim_param_dict.sigma is None

    sim_param_dict = aido.SimulationParameter("foo", 1, discrete_values=[1, 2])
    assert sim_param_dict.sigma is None

    sim_param_dict = aido.SimulationParameter("foo", 0.0, sigma=0.1)
    assert sim_param_dict.sigma == 0.1

    sim_param_dict = aido.SimulationParameter("foo", 2.0, sigma_mode="scale", sigma=0.1)
    assert sim_param_dict.sigma == 0.2


def test_set_sigma() -> None:
    sim_param_dict = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", np.random.uniform(0.1, 50), min_value=0.001),
        aido.SimulationParameter("material_absorber_0", -1, optimizable=False)
    ])
    assert sim_param_dict["thickness_absorber_0"].discrete_values is None
    assert sim_param_dict["thickness_absorber_0"].sigma == config.simulation.sigma


def test_covariance() -> None:
    sim_param_dict = aido.SimulationParameterDictionary([
        aido.SimulationParameter("foo", 3.14, sigma=0.1),
        aido.SimulationParameter("bar", "LEAD", discrete_values=["LEAD", "BREAD"]),
        aido.SimulationParameter("foo", 6.28, sigma=0.2),
        aido.SimulationParameter("foo", 9.17, optimizable=False),
        aido.SimulationParameter("foo", 10.0, sigma=1.0)
    ])
    assert np.all(sim_param_dict.covariance == np.diag(np.array([0.1, 0.2, 1])**2))
    sim_param_dict.covariance = np.diag([25, 1, 4])
    assert np.all(sim_param_dict.covariance == np.diag([25, 1, 4]))
    sim_param_dict.to_json("test_param_dict")
    sim_param_dict_2 = aido.SimulationParameterDictionary.from_json("test_param_dict")
    assert np.all(sim_param_dict_2.covariance == np.diag([25, 1, 4]))
    os.remove("./test_param_dict")


def test_current_value(sim_param_dict: aido.SimulationParameterDictionary) -> None:
    assert sim_param_dict["absorber_thickness"].current_value == 10.0
    sim_param_dict["absorber_thickness"].current_value = -5.0
    assert sim_param_dict["absorber_thickness"].current_value == -5.0
