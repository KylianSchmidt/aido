import sys

import pandas as pd
from reconstruction import Reconstruction, ReconstructionDataset
from torch.utils.data import Dataset


def pre_train(model: Reconstruction, dataset: Dataset, n_epochs: int):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalised ones,
    but also requires adjustments to the surrogate dataset
    """
    model.to('cuda')

    print("Reconstruction: Pre-training 0")
    model.train_model(dataset, batch_size=256, n_epochs=2, lr=0.03)

    print("Reconstruction: Pre-training 1")
    model.train_model(dataset, batch_size=256, n_epochs=n_epochs, lr=0.01)

    print("Reconstruction: Pre-training 2")
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print("Reconstruction: Pre-training 3")
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)
    model.to('cpu')


if __name__ == "__main__":

    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]

    # Load the input df
    simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)

    reco_dataset = ReconstructionDataset(simulation_df)
    reco_model = Reconstruction(*reco_dataset.shape)

    n_epochs_pre = 100
    n_epochs_main = 100

    pre_train(reco_model, reco_dataset, n_epochs_pre)

    # Reconstruction:
    reco_model.to('cuda')
    reco_model.train_model(reco_dataset, batch_size=256, n_epochs=n_epochs_main // 2, lr=0.003)
    reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.001)
    reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0003)
    reco_result, reco_loss, _ = reco_model.apply_model_in_batches(reco_dataset, batch_size=128)

    reconstructed_df = pd.DataFrame({"true_energy": reco_result})
    reconstructed_df = pd.concat({"Reconstructed": reconstructed_df}, axis=1)
    loss_df = pd.DataFrame({"Reco_loss": reco_loss.tolist()})
    loss_df = pd.concat({"Loss": loss_df}, axis=1)
    output_df: pd.DataFrame = pd.concat([reco_dataset.df, reconstructed_df, loss_df], axis=1)
    output_df.to_parquet(output_df_path)

    # validation
    validation_df = pd.read_parquet(
        "results_full_calorimeter/results_20241122/task_outputs/iteration=0/validation=True/validation_input_df"
    )
    validation_dataset = ReconstructionDataset(validation_df)
    val_result, val_loss, _ = reco_model.apply_model_in_batches(validation_dataset, batch_size=128)

    val_df = pd.DataFrame({"true_energy": val_result})
    val_df = pd.concat({"Reconstructed": val_df}, axis=1)
    val_loss_df = pd.DataFrame({"Reco_loss": val_loss.tolist()})
    val_loss_df = pd.concat({"Loss": val_loss_df}, axis=1)
    val_output_df: pd.DataFrame = pd.concat([validation_dataset.df, val_df, val_loss_df], axis=1)
    val_output_df.to_parquet("validation_output_df")
