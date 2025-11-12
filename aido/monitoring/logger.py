
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
from matplotlib import pyplot as plt


def flatten_gradients(parameters: ParameterModule) -> torch.Tensor:
    grads = []
    for p in parameters.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    
    if len(grads) == 0:
        raise Exception("WandbLogger: No gradients found to log.")

    flat_grads = torch.cat(grads)
    return flat_grads

class WandbLogger:
    def __init__(self, project_name: str, optim_name: str, config: Optional[dict] = None):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        self.project_name = project_name
        self.optim_name = f"{optim_name}__{timestamp}"
        self.config = config

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

    def log_figure(self, tag: str, figure: plt.Figure) -> None:
        self.wandb_instance.log({tag: wandb.Image(figure)})

    def log_gradient_histogram(self, tag: str, parameters: ParameterModule) -> None:
        flat_grads = flatten_gradients(parameters).cpu().numpy()
        self.wandb_instance.log({f"{tag}/grad_hist": wandb.Histogram(flat_grads)})

    def log_gradients(self, tag: str, parameters: ParameterModule) -> None:
        flat_grads = flatten_gradients(parameters)
        self.wandb_instance.log({f"{tag}/norm": flat_grads.norm().item()})
        self.wandb_instance.log({f"{tag}/min": flat_grads.abs().min().item()})
        self.wandb_instance.log({f"{tag}/max": flat_grads.abs().max().item()})

    def get_task_logger(self, task: str, task_iter: int) -> 'WandbTaskLogger':
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
        if exc_type is not None and issubclass(exc_type, torch.cuda.OutOfMemoryError):
            current_tags = self.wandb_instance.tags
            current_tags = [] if current_tags is None else current_tags
            self.wandb_instance.tags = [*current_tags, "oom"]

        if self.wandb_instance is not None:
            self.wandb_instance.finish()

    def log_scalar(self, tag: str, value: float, step: Optional[int] = None) -> None:
        self.wandb_instance.log({tag: value}, step=step)

    def log_scalars(self, tag, values: list[float] | np.ndarray, step_offset: Optional[int] = None) -> None:
        for i, value in enumerate(values):
            step = step_offset + i if step_offset is not None else None
            self.log_scalar(tag, value, step)