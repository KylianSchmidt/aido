import os

from pytest import fixture

import aido
from aido.config import AIDOConfig


@fixture
def aido_config():
    return AIDOConfig()


def test_instantiation(aido_config: AIDOConfig):
    assert aido_config.optimizer.batch_size == 512


def test_set_value(aido_config: AIDOConfig):
    aido_config.set_value("optimizer.batch_size", 256)
    assert aido_config.optimizer.batch_size == 256


def test_from_dict(aido_config: AIDOConfig):
    aido_config.from_dict({"optimizer.batch_size": 128})
    assert aido_config.optimizer.batch_size == 128


def test_to_json(aido_config: AIDOConfig):
    aido_config.to_json("test_config.json")
    aido_config = aido_config.from_json("test_config.json")
    os.remove("test_config.json")


def test_aido_wrapper_version():
    aido.set_config("optimizer.batch_size", 37)
    assert aido.get_config("optimizer.batch_size") == 37
