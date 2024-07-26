import sys
import pandas as pd
from torch.utils.data import Dataset
from reconstruction import ReconstructionDataset, Reconstruction


def pre_train(model: Reconstruction, dataset: Dataset, n_epochs: int):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalised ones, 
    but also requires adjustments to the surrogate dataset
    """
    model.to('cuda')

    print('pre-training 0')
    model.train_model(dataset, batch_size=256, n_epochs=10, lr=0.03)

    print('pre-training 1')
    model.train_model(dataset, batch_size=256, n_epochs=n_epochs, lr=0.01)

    print('pre-training 2')
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print('pre-training 3')
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)

    model.apply_model_in_batches(reco_dataset, batch_size=128)
    model.to('cpu')


if __name__ == "__main__":

    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]

    # Load the input df
    simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)

    reco_dataset = ReconstructionDataset(simulation_df)
    reco_model = Reconstruction(*reco_dataset.shape)

    n_epochs_pre = 5
    n_epochs_main = 50

    pre_train(reco_model, reco_dataset, n_epochs_pre)

    # Reconstruction:
    reco_model.to('cuda')
    reco_model.train_model(reco_dataset, batch_size=256, n_epochs=n_epochs_main // 4, lr=0.003)
    reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
    reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
    reco_result, reco_loss = reco_model.apply_model_in_batches(reco_dataset, batch_size=128)
    reco_result = reco_result.detach().cpu().numpy()

    reco_result = reco_result * reco_dataset.stds[2] + reco_dataset.means[2]
    reconstructed_df = pd.DataFrame(reco_result, columns=reco_dataset.df["Targets"].columns)
    reconstructed_df = pd.concat({"Reconstructed": reconstructed_df}, axis=1)
    output_df: pd.DataFrame = pd.concat([reco_dataset.df, reconstructed_df], axis=1)
    output_df.to_parquet(output_df_path)
