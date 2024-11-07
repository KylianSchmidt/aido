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
    model.train_model(dataset, batch_size=256, n_epochs=n_epochs, lr=0.001)

    print('Surrogate: Pre-Training 3')
    model.train_model(dataset, batch_size=256, n_epochs=n_epochs, lr=0.001)


def training_loop(
        reco_file_paths_dict: dict | str | os.PathLike,
        constraints: None | Callable[[SimulationParameterDictionary], float | torch.Tensor] = None
        ):
    if isinstance(reco_file_paths_dict, (str, os.PathLike)):
        with open(reco_file_paths_dict, "r") as file:
            reco_file_paths_dict = json.load(file)

    output_df_path = reco_file_paths_dict["reco_output_df"]
    parameter_dict_input_path = reco_file_paths_dict["current_parameter_dict"]
    parameter_dict_output_path = reco_file_paths_dict["updated_parameter_dict"]
    surrogate_previous_path = reco_file_paths_dict["surrogate_model_previous_path"]
    optimizer_previous_path = reco_file_paths_dict["optimizer_model_previous_path"]
    surrogate_save_path = reco_file_paths_dict["surrogate_model_save_path"]
    optimizer_save_path = reco_file_paths_dict["optimizer_model_save_path"]
    optimizer_loss_save_path = reco_file_paths_dict["optimizer_loss_save_path"]
    surrogate_loss_save_path = reco_file_paths_dict["surrogate_loss_save_path"]
    constraints_loss_save_path = reco_file_paths_dict["constraints_loss_save_path"]

    parameter_dict = SimulationParameterDictionary.from_json(parameter_dict_input_path)

    n_epochs_pre = 5
    n_epochs_main = 100

    # Surrogate:
    surrogate_dataset = SurrogateDataset(pd.read_parquet(output_df_path), norm_reco_loss=True)

    if os.path.isfile(surrogate_previous_path):
        surrogate = torch.load(surrogate_previous_path)
    else:
        surrogate = Surrogate(*surrogate_dataset.shape)

    print("Surrogate Pre-Training")
    if parameter_dict.iteration < 5:
        pre_train(surrogate, surrogate_dataset, n_epochs_pre)
    print("Surrogate Training")
    surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 2, lr=0.005)
    surrogate_loss = surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main, lr=0.0003)

    best_surrogate_loss = 1e10

    while surrogate_loss < 4.0 * best_surrogate_loss:

        if surrogate_loss < best_surrogate_loss:
            break
        else:
            print("Surrogate Re-Training")
            pre_train(surrogate, surrogate_dataset, n_epochs_pre)
            surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 5, lr=0.005)
            surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 2, lr=0.005)
            surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 2, lr=0.0003)
            surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 2, lr=0.0001)

    # Optimization
    if os.path.isfile(optimizer_previous_path):
        optimizer = torch.load(optimizer_previous_path)
    else:
        optimizer = Optimizer(surrogate, parameter_dict)

    updated_parameter_dict, is_optimal = optimizer.optimize(
        surrogate_dataset,
        batch_size=512,
        n_epochs=40,
        lr=0.02,
        additional_constraints=constraints
    )
    if not is_optimal:
        raise RuntimeError

    updated_parameter_dict.to_json(parameter_dict_output_path)

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

    torch.save(surrogate, surrogate_save_path)
    torch.save(optimizer, optimizer_save_path)


if __name__ == "__main__":

    with open(sys.argv[1], "r") as file:
        reco_file_paths_dict = json.load(file)

    training_loop(reco_file_paths_dict, constraints=None)
