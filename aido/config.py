import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class OptimizerConfig:
    lr: float = 0.02
    batch_size: int = 512
    n_epochs: int = 40


@dataclass
class SurrogateConfig:
    n_epoch_pre: int = 24
    n_epochs_main: int = 40


@dataclass
class SimulationConfig:
    generate_scaling: float = 1.2
    sigma: float = 1.5
    sigma_mode: str = "flat"


@dataclass
class AIDOConfig:
    """
    Sub-classes:
    
    Optimizer:
        optimizer.lr: float = 0.02
        optimizer.batch_size: int = 512
        optimizer.n_epochs: int = 40

    Surrogate:
        surrogate.n_epoch_pre: int = 24
        surrogate.n_epochs_main: int = 40
    
    Simulation:
        simulation.generate_scaling: float = 1.2
        simulation.sigma: float = 1.5
        simulation.sigma_mode: str = "flat"
    """
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)

        return cls(
            optimizer=OptimizerConfig(**data["optimizer"]),
            surrogate=SurrogateConfig(**data["surrogate"]),
            simulation=SimulationConfig(**data["simulation"])
        )

    def to_json(self, file_path: str):
        with open(file_path, "w") as file:
            json.dump(asdict(self), file, indent=4)

    def get_key(self, key: str):
        """
        Inputs
        ------
        key: str = dot-separated name

        Returns
        -------
            subclass name, key
        """
        keys = key.split(".")
        current_config_subclass = self

        for k in keys[:-1]:
            current_config_subclass = getattr(current_config_subclass, k)

        return current_config_subclass, keys[-1]

    def set_value(self, key: str, value: Any):
        setattr(*self.get_key(key), value)

    def get_value(self, key: str) -> Any:
        return getattr(*self.get_key(key))

    def from_dict(self, new_dict: dict):
        for key, value in new_dict.items():
            self.set_value(key, value)

    def as_dict(self) -> Dict:
        return asdict(self)


if __name__ == "__main__":
    AIDOConfig().to_json("config.json")
