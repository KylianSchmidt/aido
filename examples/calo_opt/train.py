# flake8: noqa: E402
import os
import pathlib
import sys
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))

from reconstruction.dataset import ReconstructionDataset
from reconstruction.model import Reconstruction
from reconstruction.validation_reconstruction import ReconstructionValidation


def pre_train(model: Reconstruction, dataset: Dataset, n_epochs: int):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalized ones,
    but also requires adjustments to the surrogate dataset
    """
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    print("Reconstruction: Pre-training 0")
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print("Reconstruction: Pre-training 1")
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)
    model.to('cpu')


def train(
    input_df_path: Union[str, os.PathLike],
    output_df_path: Union[str, os.PathLike],
    isVal: bool,
    results_dir: Union[str, os.PathLike],
):
    simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)

    if isVal:
        reco_model: Reconstruction = torch.load(os.path.join(results_dir, "reco_model"))
        reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)

        validator = ReconstructionValidation(reco_model)
        output_df_val = validator.validate(reco_dataset)
        output_df_val.to_parquet(output_df_path)
        validator.plot(
            output_df_val,
            os.path.join(results_dir, "plots", "validation", "reco_model", "on_validationData")
        )
    else:
        n_epochs_pre = 24
        n_epochs_main = 40
        reco_model_previous_path = os.path.join(results_dir, "reco_model")

        if os.path.exists(reco_model_previous_path):
            reco_model: Reconstruction = torch.load(reco_model_previous_path)
            reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)
        else:
            reco_dataset = ReconstructionDataset(simulation_df)
            reco_model = Reconstruction(*reco_dataset.shape, reco_dataset.means, reco_dataset.stds)
            pre_train(reco_model, reco_dataset, n_epochs_pre)

        # Reconstruction training:
        reco_model.to("cuda" if torch.cuda.is_available() else "cpu")
        reco_model.train_model(reco_dataset, batch_size=256, n_epochs=n_epochs_main // 4, lr=0.003)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
        
        validator = ReconstructionValidation(reco_model)
        output_df_val = validator.validate(reco_dataset)
        output_df_val.to_parquet(output_df_path)
        torch.save(reco_model, reco_model_previous_path)

        validator.plot(
            output_df_val,
            os.path.join(results_dir, "plots", "validation", "reco_model", "on_trainingData")
        )


if __name__ == "__main__":
    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]
    isVal = sys.argv[3].strip().lower() == "true"
    results_dir = sys.argv[4]
    train(input_df_path, output_df_path, isVal, results_dir)
