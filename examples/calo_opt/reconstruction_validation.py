import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import os
import datetime

from typing import Dict, Generator

from reconstruction import ReconstructionDataset,Reconstruction

class ReconstructionValidation():
    def __init__(
            self,
            reco_model: Reconstruction,
            ):
        
        self.reco_model = reco_model
        
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        
    def validate(
        self,
        val_dataset: ReconstructionDataset,
        batch_size: int = 512
        ):

        val_result, val_loss, _ = self.reco_model.apply_model_in_batches(val_dataset, batch_size=batch_size)
        
        final_validation_df = pd.DataFrame({"true_energy": val_result})
        final_validation_df = pd.concat({"Reconstructed": final_validation_df}, axis=1)
        loss_df_val = pd.DataFrame({"Reco_loss": val_loss.tolist()})
        loss_df_val = pd.concat({"Loss": loss_df_val}, axis=1)
        output_df_val: pd.DataFrame = pd.concat([val_dataset.df, final_validation_df, loss_df_val], axis=1)
        
        return output_df_val

    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: str):
        
        bins_energy = np.linspace(0, 20, 100 + 1)
        bins_difference = np.linspace(-10, 10, 100 + 1)

        val_reco = validation_df["Reconstructed"]["true_energy"].values
        reco_out = validation_df["Targets"]["true_energy"].values

        size = len(val_reco)

        # Initialize an empty array with fixed size
        diff = np.zeros(size)

        # Use the loop to calculate the difference
        for i in range(size):
            diff[i] = val_reco[i] - reco_out[i]

        difference = val_reco - reco_out

        # Create the figure with subplots
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        # First plot: energy distributions
        ax[0].hist(val_reco, bins=bins_energy, label="Reconstruction", histtype="step")
        ax[0].hist(reco_out, bins=bins_energy, label="Validation", histtype="step")
        ax[0].set_xlim(bins_energy[0], bins_energy[-1])
        ax[0].set_xlabel("Predicted Energy")
        ax[0].set_ylabel(f"Counts / {(bins_energy[1] - bins_energy[0]):.2f}")
        ax[0].legend()

        # Second plot: distribution of differences
        ax[1].hist(difference, bins=bins_difference, histtype="step")
        ax[1].set_xlim(bins_difference[0], bins_difference[-1])
        ax[1].set_xlabel("Reco_model Accuracy")
        ax[1].set_ylabel("Counts")

        # Save the combined figure
        plt.tight_layout()
        plt.savefig(os.path.join(fig_savepath, f"validation_recoModel_{datetime.datetime.now()}.png"))
        plt.close()

        print("Validation Plots Saved")
