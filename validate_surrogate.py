import sys

import pandas as pd
import torch

from aido.surrogate import Surrogate, SurrogateDataset
from aido.surrogate_validation import SurrogateValidation
from aido.training import pre_train

if __name__ == "__main__":
    if len(sys.argv) == 2:
        surrogate_dataset = SurrogateDataset(pd.read_parquet(sys.argv[1]), norm_reco_loss=True)

        surrogate = Surrogate(*surrogate_dataset.shape, n_time_steps=200)

        n_epochs_pre = 50
        n_epochs_main = 100
        pre_train(surrogate, surrogate_dataset, n_epochs_pre)
        surrogate.train_model(
            surrogate_dataset,
            batch_size=256,
            n_epochs=n_epochs_main,
            lr=0.0005
        )
        surrogate_loss = surrogate.train_model(
            surrogate_dataset,
            batch_size=256,
            n_epochs=n_epochs_main,
            lr=0.0001
        )
        validator = SurrogateValidation(surrogate)
        validation_df = validator.validate(surrogate_dataset, batch_size=20)
        validation_df.to_parquet(".validation_df")

    else:
        validation_df = pd.read_parquet(".validation_df")
        print("Validation DataFrame found")

    SurrogateValidation.plot(validation_df, ".")

    surrogate: Surrogate = torch.load("results_full_calorimeter/results_20241125/models/surrogate_1.pt")

    training_dataset = SurrogateDataset(pd.read_parquet(
        "results_full_calorimeter/results_20241125/task_outputs/iteration=1/validation=False/reco_output_df"
    ))
    validation_dataset = SurrogateDataset(
        pd.read_parquet(
            "results_full_calorimeter/results_20241125/task_outputs/iteration=1/validation=True/validation_output_df"
        ),
        means=training_dataset.means,
        stds=training_dataset.stds
    )

    validator = SurrogateValidation(surrogate)
    validation_df = validator.validate(validation_dataset, batch_size=20)
    validation_df.to_parquet(".validation_df")

    SurrogateValidation.plot(validation_df, "./validation_plots/")
