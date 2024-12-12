import json
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd
import torch

from aido.config import AIDOConfig
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
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print('Surrogate: Pre-Training 1')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)

    print('Surrogate: Pre-Training 2')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.0003)


def training_loop(
        reco_file_paths_dict: dict | str | os.PathLike,
        reconstruction_loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        constraints: None | Callable[[SimulationParameterDictionary], float | torch.Tensor] = None
        ):
    
    if isinstance(reco_file_paths_dict, (str, os.PathLike)):
        with open(reco_file_paths_dict, "r") as file:
            reco_file_paths_dict = json.load(file)

    config = AIDOConfig.from_json(reco_file_paths_dict["config_path"])

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
    parameter_optimizer_savepath = os.path.join(results_dir, "models", "parameter_optimizer_df")

    parameter_dict = SimulationParameterDictionary.from_json(parameter_dict_input_path)

    # Surrogate:
    surrogate_df = pd.read_parquet(output_df_path)

    if os.path.isfile(surrogate_save_path):
        surrogate: Surrogate = torch.load(surrogate_save_path)
        surrogate_dataset = SurrogateDataset(surrogate_df, means=surrogate.means, stds=surrogate.stds)
    else:
        if os.path.isfile(surrogate_previous_path):
            surrogate: Surrogate = torch.load(surrogate_previous_path)
            surrogate_dataset = SurrogateDataset(surrogate_df, means=surrogate.means, stds=surrogate.stds)
        else:
            surrogate_dataset = SurrogateDataset(surrogate_df)
            surrogate = Surrogate(*surrogate_dataset.shape, surrogate_dataset.means, surrogate_dataset.stds)
            pre_train(surrogate, surrogate_dataset, config.surrogate.n_epoch_pre)

        print("Surrogate Training")
        n_epochs_main = config.surrogate.n_epochs_main
        surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
        surrogate_loss = surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0003)

        while not surrogate.update_best_surrogate_loss(surrogate_loss):
            print("Surrogate retraining")
            pre_train(surrogate, surrogate_dataset, config.surrogate.n_epoch_pre)
            surrogate.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 5, lr=0.005)
            surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
            surrogate.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
            surrogate_loss = surrogate.train_model(
                surrogate_dataset,
                batch_size=1024,
                n_epochs=n_epochs_main // 2,
                lr=0.0001,
            )

    print("Surrogate Validation on Training Data")
    surrogate_validator = SurrogateValidation(surrogate)
    validation_df = surrogate_validator.validate(surrogate_dataset)
    surrogate_validator.plot(
        validation_df,
        fig_savepath=os.path.join(results_dir, "plots", "validation_on_training_data"),
        )

    print("Surrogate Validation")
    surrogate_validation_dataset = SurrogateDataset(
        pd.read_parquet(validation_df_path),
        means=surrogate_dataset.means,
        stds=surrogate_dataset.stds
    )
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
        optimizer = Optimizer(
            surrogate,
            continuous_lr=config.optimizer.continuous_lr,
            discrete_lr=config.optimizer.discrete_lr
        )

    updated_parameter_dict, is_optimal = optimizer.optimize(
        surrogate_dataset,
        parameter_dict,
        batch_size=config.optimizer.batch_size,
        n_epochs=config.optimizer.n_epochs,
        additional_constraints=constraints,
        reconstruction_loss=reconstruction_loss_function,
        parameter_optimizer_savepath=parameter_optimizer_savepath
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

    training_loop(reco_file_paths_dict, reconstruction_loss_function=torch.nn.MSELoss, constraints=None)
