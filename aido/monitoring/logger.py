import os
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
from functools import wraps

original_savefig = plt.savefig
active_logger: "WandbLogger" = None
is_patched: bool = False

@wraps(original_savefig)
def savefig_with_wandb(fname, *args, **kwargs):
    original_savefig(fname, *args, **kwargs)
    
    if active_logger is not None:
        fig = plt.gcf()
        tag = os.path.splitext(os.path.basename(fname))[0]
        active_logger.log_figure(tag, fig, iteration=active_logger.iteration)

def setup_wandb_savefig(wandb_logger):
    global active_logger, is_patched
    active_logger = wandb_logger
    
    if not is_patched:
        plt.savefig = savefig_with_wandb
        is_patched = True


def flatten_gradients(parameters: ParameterModule, silent: bool = True) -> torch.Tensor:
    grads = []
    for p in parameters.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
    
    if not silent and len(grads) == 0:
        raise Exception("WandbLogger: No gradients found to log.")

    flat_grads = torch.cat(grads)
    return flat_grads


class WandbLogger:
    def __init__(self, project_name: str, optim_name: str, config: Optional[dict] = None):
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.project_name = project_name
        self.optim_name = f"{optim_name}__{timestamp}"
        self.iteration: int = 0
        self.config = config

    def __enter__(self):
        self.wandb_instance = wandb.init(project=self.project_name, 
                                         name="main_optimization_run", 
                                         tags=["optimization"], 
                                         group=self.optim_name)
        setup_wandb_savefig(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.wandb_instance is not None:
            self.wandb_instance.finish()

    def synchronize_iteration(self, iteration: int) -> None:
        self.iteration = iteration
        setup_wandb_savefig(self)

    def log_scalars(self, tag, values: list[float] | np.ndarray) -> None:
        self.wandb_instance.define_metric(f"{tag}", step_metric="Iteration")
        for step, value in enumerate(values):
            msg = {"Iteration": step/len(values) + self.iteration, 
                   "Task Step": step, 
                   tag: value}
            
            self.wandb_instance.log(msg)

    def log_figure(self, tag: str, figure: plt.Figure, iteration: Optional[int] = None) -> None:
        self.wandb_instance.log({tag: wandb.Image(figure), "Iteration": iteration})


    def log_gradients(self, tag: str, gradients_norm: list[float], gradients_min: list[float], gradients_max: list[float]) -> None:
        self.wandb_instance.define_metric(f"{tag}/norm", step_metric="Iteration")
        self.wandb_instance.define_metric(f"{tag}/min", step_metric="Iteration")
        self.wandb_instance.define_metric(f"{tag}/max", step_metric="Iteration")
        
        n_steps = len(gradients_norm)
        for step, (norm, min_val, max_val) in enumerate(zip(gradients_norm, gradients_min, gradients_max)):
            msg = {
                "Iteration": step/n_steps + self.iteration,
                "Task Step": step,
                f"{tag}/norm": norm,
                f"{tag}/min": min_val,
                f"{tag}/max": max_val
            }
            self.wandb_instance.log(msg)

    def get_task_logger(self, task: str) -> 'WandbTaskLogger':
        return WandbTaskLogger(self.project_name, self.optim_name, task, self.iteration)


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

    def log_scalars(self, tag, values: list[float] | np.ndarray, step_offset: Optional[int] = None) -> None:
        self.wandb_instance.define_metric(f"{tag}", step_metric="Iteration")
        for step, value in enumerate(values):
            msg = {"Iteration": step/len(values) + self.task_iter, 
                   "Task Step": step, 
                   tag: value}
            
            self.wandb_instance.log(msg, step=step + step_offset)