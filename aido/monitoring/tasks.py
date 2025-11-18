import os
import shlex
import numpy as np
import pandas as pd
from enum import Enum
from typing import Any, Optional, Sequence, Tuple

from aido.monitoring.logger import WandbTaskLogger

class TaskType(Enum):
    RECONSTRUCTION = 0

class OutputConfig:
    def __init__(self, output_dir: str, file_mapping: dict, key_mapping: dict):
        self.output_dir = output_dir
        self.file_mapping = file_mapping
        self.key_mapping = key_mapping

        assert self.file_mapping.keys() == self.key_mapping.keys(), \
            "File_mapping and key_mapping must have the same keys."

    @classmethod
    def __default_dict_from_task_type(cls, task_type: TaskType) -> dict:
        match task_type:
            case TaskType.RECONSTRUCTION:
                return { "Reconstruction Loss": "loss/reconstruction/reconstruction_loss" }
        
        raise ValueError(f"OutputConfig: No default file mapping for task type {task_type}")


    @classmethod
    def __default_key_mapping_from_task_type(cls, task_type: TaskType) -> dict:
        match task_type:
            case TaskType.RECONSTRUCTION:
                return { "Reconstruction Loss": "Reconstruction Loss" }
        
        raise ValueError(f"OutputConfig: No default key mapping for task type {task_type}")

    @classmethod
    def create_default(cls, output_dir: str, task_type: TaskType) -> 'OutputConfig':
        file_mapping = cls.__default_dict_from_task_type(task_type)
        key_mapping = cls.__default_key_mapping_from_task_type(task_type)

        return OutputConfig(output_dir, file_mapping, key_mapping)


class WandbSubprocessWrapper:
    def __init__(self, command: str, subprocess_logger: Optional[WandbTaskLogger] = None, 
                 output_config: Optional[OutputConfig] = None, requires_loss_file: bool = False): 
        
        self.command = command
        self.output_config = output_config
        self.subprocess_logger = subprocess_logger
        self.requires_loss_file = requires_loss_file
        
        if subprocess_logger is not None:
            assert output_config is not None, "OutputConfig must be provided if subprocess_logger is used."


    @staticmethod
    def save_loss_to_file(loss: Sequence[Any], filepath: str, step_offset: int = 0) -> None:
        pd.DataFrame(
            np.array(loss),
            np.arange(step_offset, step_offset + len(loss)),
            columns=["Reconstruction Loss", "Step"]
        ).to_csv(os.path.join(filepath), index=True)


    @staticmethod
    def load_loss_from_file(filepath: str) -> Tuple[pd.DataFrame, int]:
        df = pd.read_csv(os.path.join(filepath))
        try: step_offset = df["Step"].min()
        except KeyError: step_offset = 0
        return df, step_offset


    def run(self, *args):
        command_list = shlex.split(self.command)
        command_list.extend(map(str, args))
        full_command = shlex.join(command_list)
        os.system(full_command)

        self.log_subprocess_output()

    def log_subprocess_output(self):
        if self.subprocess_logger is None:
            return
        
        try:
            for key, filename in self.output_config.file_mapping.items():
                filepath = os.path.join(self.output_config.output_dir, filename)
                df_key = self.output_config.key_mapping.get(key, None)
                if df_key is None:
                    raise Exception(f"No key mapping found for key: {key}.")
                
                df, step_offset = WandbSubprocessWrapper.load_loss_from_file(filepath)
                self.subprocess_logger.log_scalars(key, df[df_key].values, step_offset)


        except FileNotFoundError as e:
            # Ignore error if loss file is not required
            if not self.requires_loss_file:
                print(f"Subprocess output file {filepath} not found.")
                return
            raise e