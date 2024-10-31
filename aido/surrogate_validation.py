import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from aido.surrogate import Surrogate, SurrogateDataset
from aido.training import pre_train


class SurrogateValidation():
    def __init__(
            self,
            surrogate_model: Surrogate,
            ):
        self.surrogate_model = surrogate_model
        self.device = "cuda"

    def validate(
            self,
            dataset: SurrogateDataset,
            batch_size: int = 512
            ):
        data_loader = DataLoader(dataset, batch_size=batch_size)
        output_df = dataset.df
        surrogate_reconstructed_array = np.full(len(dataset), -1.0)

        for batch_idx, (parameters, context, reconstructed) in enumerate(data_loader):

            context = context.to(self.device)
            reconstructed = reconstructed.to(self.device)
            parameters = parameters.to(self.device)

            surrogate_output = self.surrogate_model.sample_forward(
                parameters,
                context
            )
            surrogate_output = surrogate_output.detach().cpu().numpy().flatten()
            surrogate_output = dataset.unnormalise_features(surrogate_output, index=2)
            surrogate_reconstructed_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = surrogate_output
            print(f"Validation batch {batch_idx} / {len(data_loader)}", end="\r")

        print(f"Validation batch {len(data_loader)} / {len(data_loader)}. Done")
        output_df[("Loss", "Surrogate")] = surrogate_reconstructed_array
        return output_df


if __name__ == "__main__":
    if len(sys.argv) == 2:
        surrogate_dataset = SurrogateDataset(pd.read_parquet(sys.argv[1]), norm_reco_loss=True)

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

    else:
        output_df = pd.read_parquet(".validation_df")
        print("Validation DataFrame found")

    bins = np.linspace(-5, 20, 200 + 1)
    plt.hist(output_df["Loss"]["Reco_loss"], bins=bins, label="Reco", histtype="step")
    plt.hist(output_df["Loss"]["Surrogate"], bins=bins, label="Surrogate", histtype="step")
    plt.xlim(bins[0], bins[-1])
    plt.xlabel("Loss")
    plt.legend()
    plt.savefig(".validation")
    plt.close()

    bins = np.linspace(-5, 5, 100 + 1)
    plt.hist(output_df["Loss"]["Reco_loss"] - output_df["Loss"]["Surrogate"], bins=bins)
    plt.xlabel("Surrogate Accuracy")
    plt.xlim(bins[0], bins[-1])
    plt.savefig(".accuracy")
    plt.close()
    print("Validation Plots Saved")
