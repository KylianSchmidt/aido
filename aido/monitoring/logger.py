import os
import uuid
from ctypes import ArgumentError
from email.policy import default
from enum import Enum
import os
import torch
from datetime import datetime
from tokenize import group 
from typing import Any, Optional, Sequence, Tuple
from aido.config import AIDOConfig
import numpy as np
from aido.optimization_helpers import ContinuousParameter, ParameterModule
from matplotlib import pyplot as plt
from functools import wraps


WANDB_ERROR_MESSAGE = (
    "wandb is not installed. Please install it with: pip install wandb " 
    "or install aido with wandb support: pip install .[wandb]"
)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

original_savefig = plt.savefig
active_logger: Optional["WandbLogger"] = None
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
    def __init__(self, project_name: str, optim_name: str):
        """Wandb high-level logging class managing both optimization-level logging and subtask logger.

        Args:
            project_name (str): Wandb project name.
            optim_name (str): Descriptive name of the optimization.

        Raises:
            ImportError: Raises import error if wandb is not installed either directly or via aido[wandb].
        """
        if not WANDB_AVAILABLE:
            raise ImportError(WANDB_ERROR_MESSAGE)
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

        date_str = datetime.now().strftime("%Y-%m-%d")
        uid = uuid.uuid1().hex[:4] 
        sortable_id = f"{date_str}_{uid}"


        self.project_name = project_name
        self.optim_name = f"{optim_name}__{sortable_id}"
        self.iteration: int = 0
        self.wandb_instance = None

    def __enter__(self):
        if not WANDB_AVAILABLE:
            raise ImportError(WANDB_ERROR_MESSAGE)
        self.wandb_instance = wandb.init(project=self.project_name,  # type: ignore
                                         name="main_optimization_run", 
                                         tags=["optimization"], 
                                         group=self.optim_name)
        setup_wandb_savefig(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.wandb_instance is not None:
            self.wandb_instance.finish()

    def synchronize_iteration(self, iteration: int) -> None:
        """Set iteration managed by b2luigi to synchronize current iteration with the worker.
        Additionally monkey-patches plt.savefig to automatically log figures to wandb.

        Args:
            iteration (int): Current optimization iteration.
        """
        self.iteration = iteration
        setup_wandb_savefig(self)

    def log_scalars(self, tag, values: list[float] | np.ndarray) -> None:
        if self.wandb_instance is None:
            return
        self.wandb_instance.define_metric(f"{tag}", step_metric="Iteration")
        for step, value in enumerate(values):
            msg = {"Iteration": step/len(values) + self.iteration, 
                   "Task Step": step, 
                   tag: value}
            
            self.wandb_instance.log(msg)

    def log_figure(self, tag: str, figure, iteration: Optional[int] = None) -> None:
        if self.wandb_instance is None:
            return
        self.wandb_instance.log({tag: wandb.Image(figure), "Iteration": iteration})  # type: ignore

    def log_config(self, config: AIDOConfig) -> None:
        if self.wandb_instance is None:
            return
        self.wandb_instance.config.update(config.as_dict())

    def log_gradients(self, tag: str, 
                      gradients_norm: Sequence[float], 
                      gradients_min: Sequence[float], 
                      gradients_max: Sequence[float]) -> None:
        
        if self.wandb_instance is None:
            return
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
        """Custom task logger generating a individual run object in wandb linked to the optimization group.

        Args:
            project_name (str): Wandb project name.
            optim_name (str): Descriptive name of the optimization.
            task (str): Task name.
            task_iter (int, optional): Optimization iteration. Defaults to 0.

        Raises:
            ImportError: Raises import error if wandb is not installed either directly or via aido[wandb].
        """
        if not WANDB_AVAILABLE:
            raise ImportError(WANDB_ERROR_MESSAGE)
        self.project_name = project_name
        self.optim_name = optim_name
        self.task = task
        self.task_iter = task_iter
        self.wandb_instance = None

    def __enter__(self):
        if not WANDB_AVAILABLE:
            raise ImportError(WANDB_ERROR_MESSAGE)
        self.wandb_instance = wandb.init(project=self.project_name,  # type: ignore
                                         name=f"{self.task}_iteration={self.task_iter}", 
                                         tags=[self.task], group=self.optim_name)
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.wandb_instance is not None:
            if exc_type is not None and issubclass(exc_type, torch.cuda.OutOfMemoryError):
                current_tags = self.wandb_instance.tags
                current_tags = [] if current_tags is None else current_tags
                self.wandb_instance.tags = [*current_tags, "oom"]
            self.wandb_instance.finish()

    def log_scalars(self, tag, values: list[float] | np.ndarray, step_offset: Optional[int] = None) -> None:
        if self.wandb_instance is None:
            return
        self.wandb_instance.define_metric(f"{tag}", step_metric="Iteration")
        for step, value in enumerate(values):
            msg = {"Iteration": step/len(values) + self.task_iter, 
                   "Task Step": step, 
                   tag: value}
            
            if step_offset is None:
                step_offset = 0

            self.wandb_instance.log(msg, step=step + step_offset)
