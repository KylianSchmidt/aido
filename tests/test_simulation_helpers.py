import pytest

import aido


@pytest.fixture
def sim_param_dict() -> aido.SimulationParameterDictionary:
    return aido.SimulationParameterDictionary([
        aido.SimulationParameter("absorber_thickness", 10.0, min_value=0.5, max_value=40.0),
        aido.SimulationParameter("absorber_material", "LEAD", discrete_values=["LEAD", "TUNGSTEN"]),
        aido.SimulationParameter("energy", 1000, optimizable=False),
        aido.SimulationParameter("num_absorber_plates", 2, discrete_values=list(range(0, 5))),
    ])


def test_save_load(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict.to_json("./sim_param_dict")
    aido.SimulationParameterDictionary.from_json("./sim_param_dict")


def test_generate_new(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict_2 = sim_param_dict.generate_new()
    sim_param_dict_3 = sim_param_dict.generate_new()
    assert isinstance(sim_param_dict_2.rng_seed, int)
    assert sim_param_dict_2.rng_seed != sim_param_dict_3.rng_seed


def test_update_metadata(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict.iteration = 17
    assert sim_param_dict.iteration == 17


def test_to_df(sim_param_dict: aido.SimulationParameterDictionary):
    sim_param_dict.to_df(types="all", one_hot=True)
    sim_param_dict.to_df(types="discrete", one_hot=True)
    with pytest.raises(ValueError):
        sim_param_dict.to_df(types="continuous", one_hot=True)
    with pytest.raises(NotImplementedError):
        sim_param_dict.get_current_values(format="list", one_hot=True)


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
