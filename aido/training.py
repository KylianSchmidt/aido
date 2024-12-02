import json
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd
import torch

from aido.optimizer import Optimizer
from aido.simulation_helpers import SimulationParameterDictionary
from aido.surrogate import Surrogate, SurrogateDataset
from aido.surrogate_validation import SurrogateValidation


def pre_train(model: Surrogate, dataset: SurrogateDataset, n_epochs: int):
    """ Pre-train the Surrogate Model.

    TODO Reconstruction results are normalized. In the future only expose the un-normalised ones,
    but also requires adjustments to the surrogate dataset
    """
    model.to('cuda')

    print('Surrogate: Pre-Training 0')
    model.train_model(dataset, batch_size=256, n_epochs=10, lr=0.03)

    print('Surrogate: Pre-Training 1')
    model.train_model(dataset, batch_size=256, n_epochs=n_epochs, lr=0.01)

    print('Surrogate: Pre-Training 2')
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print('Surrogate: Pre-Training 3')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)

    print('Surrogate: Pre-Training 4')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.0003)


def training_loop(
        reco_file_paths_dict: dict | str | os.PathLike,
        constraints: None | Callable[[SimulationParameterDictionary], float | torch.Tensor] = None
        ):
    if isinstance(reco_file_paths_dict, (str, os.PathLike)):
        with open(reco_file_paths_dict, "r") as file:
            reco_file_paths_dict = json.load(file)

    results_dir = reco_file_paths_dict["results_dir"]
    output_df_path = reco_file_paths_dict["reco_output_df"]
    validation_df_path = reco_file_paths_dict["validation_output_df"]
    parameter_dict_input_path = reco_file_paths_dict["current_parameter_dict"]
    surrogate_previous_path = reco_file_paths_dict["surrogate_model_previous_path"]
    optimizer_previous_path = reco_file_paths_dict["optimizer_model_previous_path"]
    surrogate_save_path = reco_file_paths_dict["surrogate_model_save_path"]
    optimizer_save_path = reco_file_paths_dict["optimizer_model_save_path"]
    optimizer_loss_save_path = reco_file_paths_dict["optimizer_loss_save_path"]
    surrogate_loss_save_path = reco_file_paths_dict["surrogate_loss_save_path"]
    constraints_loss_save_path = reco_file_paths_dict["constraints_loss_save_path"]

    parameter_dict = SimulationParameterDictionary.from_json(parameter_dict_input_path)

    n_epochs_pre = 30
    n_epochs_main = 100

    # Surrogate:
    surrogate_dataset = SurrogateDataset(
        pd.read_parquet(output_df_path),
    )

    if os.path.isfile(surrogate_save_path):
        surrogate = torch.load(surrogate_save_path)
    else:
        if os.path.isfile(surrogate_previous_path):
            surrogate: Surrogate = torch.load(surrogate_previous_path)
        else:
            surrogate = Surrogate(*surrogate_dataset.shape)
            pre_train(surrogate, surrogate_dataset, n_epochs_pre)

    print("Surrogate Training")
    surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
    surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0003)

    print("Surrogate Validation")
    surrogate_validation_dataset = SurrogateDataset(pd.read_parquet(validation_df_path))
    surrogate_validator = SurrogateValidation(surrogate)
    validation_df = surrogate_validator.validate(surrogate_validation_dataset)
    surrogate_validator.plot(
        validation_df,
        fig_savepath=os.path.join(results_dir, "plots", "validation"),
        )

    torch.save(surrogate, surrogate_save_path)

    # Optimization
    if os.path.isfile(optimizer_previous_path):
        optimizer: Optimizer = torch.load(optimizer_previous_path)
    else:
        optimizer = Optimizer(surrogate, parameter_dict, continuous_lr=0.02, discrete_lr=0.01)

    updated_parameter_dict, is_optimal = optimizer.optimize(
        surrogate_dataset,
        parameter_dict,
        batch_size=512,
        n_epochs=40,
        additional_constraints=constraints,
        parameter_optimizer_savepath=os.path.join(results_dir, "models", "parameter_optimizer_df")
    )
    if not is_optimal:
        raise RuntimeError
    else:
        torch.save(optimizer, optimizer_save_path)

    pd.DataFrame(
        np.array(surrogate.surrogate_loss),
        columns=["Surrogate Loss"]
    ).to_csv(surrogate_loss_save_path)
    pd.DataFrame(
        np.array(optimizer.optimizer_loss),
        columns=["Optimizer Loss"]
    ).to_csv(optimizer_loss_save_path)
    pd.DataFrame(
        np.array(optimizer.constraints_loss),
        columns=["Constraints Loss"]
    ).to_csv(constraints_loss_save_path)

    return updated_parameter_dict


if __name__ == "__main__":

    with open(sys.argv[1], "r") as file:
        reco_file_paths_dict = json.load(file)

    training_loop(reco_file_paths_dict, constraints=None)
