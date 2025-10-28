import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple, Self

from aido.logger import logger


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
class SchedulerConfig:
    training_num_retries: int = 20
    training_delay_between_retries: int | float = 60


@dataclass
class AIDOConfig:
    """
    Config Dataclass for storing the internal parameter such as the hyperparameters of the different
    models and the way new values are sampled for the Simulation Task.

    This class is serializable to and from json. In order to be picked up by the AIDO scheduler, the
    json file with updated values must be placed in the AIDO root directory.

    Default fields:
    
    - Optimizer:
        - optimizer.lr: float = 0.02 (>0)
        - optimizer.batch_size: int = 512
        - optimizer.n_epochs: int = 40

    - Surrogate:
        - surrogate.n_epoch_pre: int = 24
        - surrogate.n_epochs_main: int = 40

    - Simulation:
        - simulation.generate_scaling: float = 1.2 (>0)
        - simulation.sigma: float = 1.5 (>0)
        - simulation.sigma_mode: str = "flat" (or "scale")

    - Scheduler:
        - scheduler.training_num_retries: int = 20
        - scheduler.training_delay_between_retries: int | float = 60 (in seconds)
    """
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_json(cls, file_path: str) -> Self:
        """Create a new instance from a json file

        Args:
            file_path (str): The input file path

        Returns:
            AIDOConfig: New instance of this class

        Raises:
            Warning: If the file could not be found, a warning is displayed instead of raising a
                FileNotFoundError. This is to ensure that the code can still run despite an invalid
                config.
            Warning: If any Exception is raised while reading in the json file, e.g. if it is invalid
                or wrongly formatted, a warning is displayed. Will show the Error message for debugging.        
        """
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {file_path} not found. Using default configuration.")
            return cls()
        except Exception as e:
            logger.error(f"Error reading config file {file_path}. Using default configuration.\nError: {e}")
            return cls()

        return cls(
            optimizer=OptimizerConfig(**data["optimizer"]),
            surrogate=SurrogateConfig(**data["surrogate"]),
            simulation=SimulationConfig(**data["simulation"]),
            scheduler=SchedulerConfig(**data["scheduler"])
        )

    def to_json(self, file_path: str) -> None:
        """Write the current values to a json file

        Args:
            file_path (str): The output file path        
        """
        with open(file_path, "w") as file:
            json.dump(self.as_dict(), file, indent=4)

    def get_key(self, key: str) -> Tuple[Self, Any]:
        """Helper method to find the subclass that corresponds to a given dot-separated name.

        Args:
            key (str): A dot-separated name. For example `"optimizer.lr"`.

        Returns:
            tuple: A tuple of the sub-class together with the attribute name.

        Example:
            >>> AIDOConfig.get_key("optimizer.lr")
            (OptimizerConfig(lr=0.02, batch_size=512, n_epochs=40), 'lr')

            Where the first entry is an instance of the :class:`OptimizerConfig` subclass and
            the second entry is the key that can be used to access that attribute.

            >>> getattr(OptimizerConfig, "lr")
            0.02
        """
        keys = key.split(".")
        current_config_subclass = self

        for k in keys[:-1]:
            current_config_subclass = getattr(current_config_subclass, k)

        return current_config_subclass, keys[-1]

    def set_value(self, key: str, value: Any) -> None:
        """Change the value of a field by providing a dot-separated key and value

        Args:
            key (str): dot-separated name, for example `"optimizer.lr"` will adjust the
                attribute `"lr"` of the sub-class :class:`OptimizerConfig`.
            value (Any): The updated value
        """
        setattr(*self.get_key(key), value)

    def get_value(self, key: str) -> Any:
        """Get the value from one of the subclasses

        Args:
            key (str): A dot-separated name
        
        Returns:
            Any: The attribute value corresponding to the key
        """
        return getattr(*self.get_key(key))

    def __getitem__(self, key: str) -> Any:
        """See :meth:`AIDOConfig.get_value`"""
        return self.get_value(key)

    def from_dict(self, new_dict: dict) -> None:
        """Update values from this instance with values from the provided dict.

        Args:
            new_dict (dict): A dictionary with the new values, with the keys being in the dot-separated
                format used by :meth:`AIDOConfig.set_value`        
        """
        for key, value in new_dict.items():
            self.set_value(key, value)

    def as_dict(self) -> Dict:
        """Return all values from this Config class as a dict
        
        Returns:
            dict: Nested dictionary with subclasses also being dicts
        """
        return asdict(self)


if __name__ == "__main__":
    """ Use as script to reset the values of the config.json file to their defaults
    """
    AIDOConfig().to_json("config.json")
