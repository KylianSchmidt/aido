import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from aido.surrogate import Surrogate, SurrogateDataset
from aido.training import pre_train


class SurrogateValidation():
    def __init__(
            self,
            surrogate_model: Surrogate
            ):
        self.surrogate_model = surrogate_model
        self.device = "cuda"

    def validate(
            self,
            dataset: SurrogateDataset,
            ):
        data_loader = DataLoader(dataset, batch_size=1)
        output_df = dataset.df
        surrogate_reconstructed_list = []

        for batch_idx, (parameters, context, reconstructed) in enumerate(data_loader):

            context = context.to(self.device)
            reconstructed = reconstructed.to(self.device)
            parameters = parameters.to(self.device)

            surrogate_output = self.surrogate_model.sample_forward(
                parameters,
                context
            )
            surrogate_reconstructed_list.append(surrogate_output.item())
            print(f"Validation batch {batch_idx} / {len(data_loader)}", end="\r")

        output_df[("Loss", "Surrogate")] = np.array(surrogate_reconstructed_list)
        return output_df


if __name__ == "__main__":
    surrogate_dataset = SurrogateDataset(pd.read_parquet(sys.argv[1]))

    surrogate = Surrogate(*surrogate_dataset.shape)

    n_epochs_pre = 5
    n_epochs_main = 100
    pre_train(surrogate, surrogate_dataset, n_epochs_pre)
    surrogate.train_model(
        surrogate_dataset,
        batch_size=1024,
        n_epochs=n_epochs_main // 2,
        lr=0.005
    )
    surrogate_loss = surrogate.train_model(
        surrogate_dataset,
        batch_size=1024,
        n_epochs=n_epochs_main,
        lr=0.0003
    )

    best_surrogate_loss = 1e10

    while surrogate_loss < 4.0 * best_surrogate_loss:

        if surrogate_loss < best_surrogate_loss:
            break
        else:
            print("Surrogate Re-Training")
            pre_train(surrogate, surrogate_dataset, n_epochs_pre)
            surrogate.train_model(
                surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 5, lr=0.005)
            surrogate.train_model(
                surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
            surrogate.train_model(
                surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
            surrogate.train_model(
                surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0001)

    validator = SurrogateValidation(surrogate)
    output_df = validator.validate(surrogate_dataset)
    output_df.to_parquet(".validation_df")
