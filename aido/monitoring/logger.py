
from ctypes import ArgumentError
from email.policy import default
from enum import Enum
import os
import torch
from datetime import datetime
from tokenize import group 
import wandb
from typing import Any, Optional, Sequence, Tuple
from aido.config import AIDOConfig
import numpy as np
from aido.optimization_helpers import ContinuousParameter, ParameterModule


class WandbLogger:
    def __init__(self, project_name: str, optim_name: str, config: Optional[dict] = None):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        self.project_name = project_name
        self.optim_name = f"{optim_name}__{timestamp}"
        self.config = config
        self.task_iters = dict()

    def __enter__(self):
        self.wandb_instance = wandb.init(project=self.project_name, 
                                         name="main_optimization_run", 
                                         tags=["optimization"], 
                                         group=self.optim_name)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.wandb_instance is not None:
            self.wandb_instance.finish()

    def log_scalar(self, tag: str, value: float) -> None:
        self.wandb_instance.log({tag: value})

    def log_scalars(self, tag, values: list[float] | np.ndarray) -> None:
        for value in values:
            self.log_scalar(tag, value)

    def log_gradient_histogram(self, tag: str, parameters: ParameterModule) -> None:
        self.wandb_instance.log({tag: wandb.Histogram(parameters.continuous_tensors().grad.cpu().numpy())})

    def get_task_logger(self, task: str) -> 'WandbTaskLogger':
        task_iter = self.task_iters.get(task, 0)
        self.task_iters[task] = task_iter + 1
        return WandbTaskLogger(self.project_name, self.optim_name, task, task_iter)


class WandbTaskLogger:
    def __init__(self, project_name: str, optim_name: str, task: str, task_iter: int = 0):
        self.project_name = project_name
        self.optim_name = optim_name
        self.task = task
        self.task_iter = task_iter

    def __enter__(self):
        self.wandb_instance = wandb.init(project=self.project_name, 
                                              name=f"{self.task}_iteration={self.task_iter}", 
                                              tags=[self.task], group=self.optim_name)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is torch.cuda.OutOfMemoryError:
            current_tags = self.wandb_instance.tags
            current_tags = () if current_tags is None else current_tags
            self.wandb_instance.tags = current_tags + ("oom_error",)

        if self.wandb_instance is not None:
            self.wandb_instance.finish()

    def log_scalar(self, tag: str, value: float) -> None:
        self.wandb_instance.log({tag: value})

    def log_scalars(self, tag, values: list[float] | np.ndarray) -> None:
        for value in values:
            self.log_scalar(tag, value)