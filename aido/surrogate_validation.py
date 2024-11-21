import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from aido.surrogate import Surrogate, SurrogateDataset


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
        validation_df = dataset.df
        surrogate_reconstructed_array = np.full(len(dataset), -1.0)

        for batch_idx, (parameters, context, reconstructed) in enumerate(data_loader):

            context = context.to(self.device)
            reconstructed = reconstructed.to(self.device)
            parameters = parameters.to(self.device)

            surrogate_output = self.surrogate_model.sample_forward(
                parameters,
                context
            )
            surrogate_output = dataset.unnormalise_reconstructed(surrogate_output)
            surrogate_output = surrogate_output.detach().cpu().numpy().flatten()
            surrogate_reconstructed_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = surrogate_output
            print(f"Validation batch {batch_idx} / {len(data_loader)}", end="\r")

        print(f"Validation batch {len(data_loader)} / {len(data_loader)}. Done")
        validation_df[("Loss", "Surrogate")] = surrogate_reconstructed_array
        return validation_df
    
    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: os.PathLike | str):
        bins = np.linspace(-5, 5, 100 + 1)
        plt.hist(np.log(validation_df["Loss"]["Reco_loss"] + 10e-10), bins=bins, label="Reco", histtype="step")
        plt.hist(np.log(validation_df["Loss"]["Surrogate"] + 10e-10), bins=bins, label="Surrogate", histtype="step")
        plt.xlim(bins[0], bins[-1])
        plt.xlabel("Loss")
        plt.ylabel(f"Counts / {(bins[1] - bins[0]):.2f}")
        plt.legend()
        plt.savefig(os.path.join(fig_savepath, f"validation_loss_{datetime.datetime.now()}.png"))
        plt.close()

        bins = np.linspace(-10, 10, 100 + 1)
        plt.hist(validation_df["Loss"]["Reco_loss"] - validation_df["Loss"]["Surrogate"], bins=bins)
        plt.xlabel("Surrogate Accuracy")
        plt.xlim(bins[0], bins[-1])
        plt.savefig(os.path.join(fig_savepath, f"validation_accuracy_{datetime.datetime.now()}.png"))
        plt.close()
        print("Validation Plots Saved")
