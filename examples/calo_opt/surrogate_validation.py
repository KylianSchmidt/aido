import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from interface_simple import AIDOUserInterfaceExample
from torch.utils.data import DataLoader

from aido.logger import logger
from aido.surrogate import Surrogate, SurrogateDataset


class SurrogateValidation():
    def __init__(
            self,
            surrogate_model: Surrogate,
            ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.surrogate_model = surrogate_model.to(self.device)

    def validate(
            self,
            dataset: SurrogateDataset,
            batch_size: int = 512,
            ):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        validation_df = dataset.df
        surrogate_reconstructed_array = np.full(len(dataset), -1.0)

        for batch_idx, (parameters, context, targets, reconstructed) in enumerate(data_loader):
            parameters = parameters.to(self.device)
            context = context.to(self.device)
            targets = targets.to(self.device)

            surrogate_output = self.surrogate_model.sample_forward(
                parameters,
                context,
                targets,
            )
            surrogate_output = dataset.unnormalise_features(surrogate_output, index=2)
            surrogate_output = surrogate_output.detach().cpu().numpy().flatten()
            surrogate_reconstructed_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = surrogate_output
            logger.info(f"Validation batch {batch_idx} / {len(data_loader)}\r")

        logger.info(f"Validation batch {len(data_loader)} / {len(data_loader)}. Done")
        validation_df[("Surrogate")] = surrogate_reconstructed_array
        return validation_df

    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: os.PathLike | str) -> None:
        """ Plot the reconstructed 'true_energy'
        """
        os.makedirs(fig_savepath, exist_ok=True)
        bins = np.linspace(0, 20, 40 + 1)
        true_energy = validation_df["Targets"]["true_energy"]
        validation_energy = validation_df["Reconstructed"]["true_energy"]
        surrogate_energy = validation_df["Surrogate"]

        fig, ax = plt.subplots()
        ax = AIDOUserInterfaceExample.add_plot_header(ax)
        plt.hist(
            [validation_energy, surrogate_energy, true_energy],
            bins=bins,
            label=["Validation", "Surrogate", "Simulation"],
            color=["blue", "red", "black"],
            histtype="step",
        )
        plt.legend()
        plt.tight_layout()
        plt.ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} [GeV])")
        plt.xlabel("Initial Energy [GeV]")
        plt.xlim(bins[0], bins[-1])
        plt.ylim(0, 300)
        plt.savefig(os.path.join(fig_savepath, "validation.png"))


def validate_surrogate_func(
        surrogate: Surrogate,
        results_dir: str,
        validation_df_path: str,
        ) -> None:
    if not validation_df_path:
        return None

    logger.info("Surrogate Validation")
    surrogate_validation_dataset = SurrogateDataset(
        pd.read_parquet(validation_df_path),
        normalise_parameters=True,
        means=surrogate.means,
        stds=surrogate.stds,
    )
    surrogate_validator = SurrogateValidation(surrogate)
    validation_df = surrogate_validator.validate(surrogate_validation_dataset)
    surrogate_validator.plot(
        validation_df,
        fig_savepath=os.path.join(results_dir, "plots"),
    )
    return None


if __name__ == "__main__":
    plt.style.use("belle2")
    logger.setLevel("DEBUG")
    dataset_path = "results_paper/results_20250129_3/task_outputs/iteration=100/validation=False/reco_output_df"
    surrogate_model: Surrogate = torch.load("results_paper/results_20250129_3/models/surrogate_100.pt")
    validate_surrogate_func(
        surrogate=surrogate_model,
        validation_df_path=dataset_path,
        results_dir="results_paper/results_20250129_3",
    )
