import datetime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from reconstruction import Reconstruction, ReconstructionDataset

from aido.logger import logger

matplotlib.use("agg")


class ReconstructionValidation():
    """ Validate a given instance of the Reconstruction model
    """
    def __init__(

            self,
            reco_model: Reconstruction,
            ):
        """
        Initializes the Validation object for a given Reconstruction model.
        Args:
            reco_model (Reconstruction): The reconstruction model to be used.
        Attributes:
            device (torch.device): The device to be used for computation, either 'cuda' if a GPU is available or 'cpu'.
            """
        self.reco_model = reco_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate(
            self,
            val_dataset: ReconstructionDataset,
            batch_size: int = 512
            ) -> pd.DataFrame:
        """ Apply the Reconstruction model on the validation dataset `val_dataset` and concatenate the
        results with that dataset, adding the columns ("Loss", "Reco_loss") and ("Reconstructed", "true_energy")
        as a multi-column index.
        Args:
            val_dataset (ReconstructionDataset): A valid dataset trained on simulation data distinct from the
            original Reconstruction model's training dataset.
            batch_size (int): Batch size for validation
        """
        val_result, val_loss, _ = self.reco_model.apply_model_in_batches(val_dataset, batch_size=batch_size)

        validation_df = pd.DataFrame({"true_energy": val_result})
        validation_df = pd.concat({"Reconstructed": validation_df}, axis=1)
        loss_df_val = pd.DataFrame({"Reco_loss": val_loss.tolist()})
        loss_df_val = pd.concat({"Loss": loss_df_val}, axis=1)
        output_df_val: pd.DataFrame = pd.concat([val_dataset.df, validation_df, loss_df_val], axis=1)
        return output_df_val

    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: str) -> None:
        """ Plots
            - The energy distribution of the Reconstructed and validated datasets
            - The reconstruction accuracy
        """

        bins_energy = np.linspace(0, 20, 100 + 1)
        bins_difference = np.linspace(-10, 10, 100 + 1)

        val_reco = validation_df["Reconstructed"]["true_energy"].values
        reco_out = validation_df["Targets"]["true_energy"].values

        diff = np.zeros(len(val_reco))

        for i in range(len(val_reco)):
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

        logger.info("Validation Plots Saved")
        return None
