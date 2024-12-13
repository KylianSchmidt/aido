import os

from pytest import fixture

from aido.config import AIDOConfig


@fixture
def aido_config():
    return AIDOConfig()


def test_to_json(aido_config: AIDOConfig):
    aido_config.to_json("test_config.json")
    aido_config = aido_config.from_json("test_config.json")
    os.remove("test_config.json")
