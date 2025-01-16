import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from aido.surrogate import Surrogate, SurrogateDataset


class SurrogateValidation():
    def __init__(
            self,
            surrogate_model: Surrogate,
            ):
        self.surrogate_model = surrogate_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(
            self,
            dataset: SurrogateDataset,
            batch_size: int = 512
            ):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_df = dataset.df
        surrogate_reconstructed_array = np.full(len(dataset), -1.0)

        for batch_idx, (parameters, context, targets, reconstructed) in enumerate(data_loader):
            parameters = parameters.to(self.device)
            context = context.to(self.device)
            targets = targets.to(self.device)
            reconstructed = reconstructed.to(self.device)

            surrogate_output = self.surrogate_model.sample_forward(
                parameters,
                context,
                targets
            )
            surrogate_output = dataset.unnormalise_features(surrogate_output, index=2)
            surrogate_output = surrogate_output.detach().cpu().numpy().flatten()
            surrogate_reconstructed_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = surrogate_output
            print(f"Validation batch {batch_idx} / {len(data_loader)}", end="\r")

        print(f"Validation batch {len(data_loader)} / {len(data_loader)}. Done")
        validation_df[("Surrogate")] = surrogate_reconstructed_array
        return validation_df

    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: os.PathLike | str):
        os.makedirs(fig_savepath, exist_ok=True)
        bins = np.linspace(0, 20, 100 + 1)
        val_reco = validation_df["Reconstructed"]["true_energy"]
        surr_reco = validation_df["Surrogate"]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        bins = np.linspace(0, 20, 100 + 1)
        axes[0].hist(val_reco, bins=bins, label="Validation", histtype="step")
        axes[0].hist(surr_reco, bins=bins, label="Surrogate", histtype="step")
        axes[0].set_xlim(bins[0], bins[-1])
        axes[0].set_xlabel("Predicted Energy")
        axes[0].set_ylabel(f"Counts / {(bins[1] - bins[0]):.2f}")
        axes[0].legend()

        bins_diff = np.linspace(-10, 10, 100 + 1)
        axes[1].hist(val_reco - surr_reco, bins=bins_diff, color='orange', alpha=0.7)
        axes[1].set_xlabel("Surrogate Accuracy (Difference)")
        axes[1].set_xlim(bins_diff[0], bins_diff[-1])
        axes[1].set_ylabel("Counts")

        fig.tight_layout()

        filename = os.path.join(fig_savepath, f"validation_surrogate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(filename)
        plt.close()
        print(f"Validation Plots Saved: {filename}")
