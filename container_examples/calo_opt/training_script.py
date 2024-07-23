import sys
import os
from typing import Union
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
from reconstruction import ReconstructionDataset, Reconstruction
from surrogate import SurrogateDataset, Surrogate
from optimizer import Optimizer


def pre_train(model: Union[Reconstruction, Surrogate], dataset: Dataset, n_epochs: int):
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


with open(sys.argv[1], "r") as file:
    reco_file_paths_dict = json.load(file)

input_df_path = reco_file_paths_dict["reco_input_df"]
output_df_path = reco_file_paths_dict["reco_output_df"]
parameter_dict_input_path = reco_file_paths_dict["current_parameter_dict"]
parameter_dict_output_path = reco_file_paths_dict["updated_parameter_dict"]
surrogate_model_previous_path = reco_file_paths_dict["surrogate_model_previous_path"]
optimizer_model_previous_path = reco_file_paths_dict["optimizer_model_previous_path"]
surrogate_save_path = reco_file_paths_dict["surrogate_model_save_path"]
optimizer_save_path = reco_file_paths_dict["optimizer_model_save_path"]

# Load the input df
simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)

with open(parameter_dict_input_path, "r") as file:
    parameter_dict: dict = json.load(file)

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

# Surrogate:
print("Surrogate training")
surrogate_dataset = SurrogateDataset(reco_dataset, reco_result)

if os.path.isfile(surrogate_model_previous_path):
    surrogate_model = torch.load(surrogate_model_previous_path)
else:
    surrogate_model = Surrogate(*surrogate_dataset.shape)

surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
surrogate_loss = surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main, lr=0.0003)

best_surrogate_loss = 1e10

while surrogate_loss < 4.0 * best_surrogate_loss:

    if surrogate_loss < best_surrogate_loss:
        break

    else:
        print("Surrogate re-training")
        pre_train(surrogate_model, surrogate_dataset, n_epochs_pre)
        surrogate_model.train_model(surrogate_dataset, batch_size=256, n_epochs=n_epochs_main // 5, lr=0.005)
        surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.005)
        surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
        sl = surrogate_model.train_model(surrogate_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0001)

surr_out, reco_out, true_in = surrogate_model.apply_model_in_batches(surrogate_dataset, batch_size=512)
surr_out = surr_out * surrogate_dataset.stds[1] + surrogate_dataset.means[1]
surrogate_df = pd.DataFrame(surr_out, columns=reco_dataset.df["Targets"].columns)
surrogate_df = pd.concat({"Surrogate": surrogate_df}, axis=1)
output_df: pd.DataFrame = pd.concat([surrogate_dataset.df, surrogate_df], axis=1)
output_df.to_parquet(output_df_path, index=range(len(output_df)))

# Optimization
if os.path.isfile(optimizer_model_previous_path):
    optimizer = torch.load(optimizer_model_previous_path)
else:
    optimizer = Optimizer(surrogate_model, reco_model, parameter_dict)

updated_parameter_dict, is_optimal, o_loss = optimizer.optimize(
    surrogate_dataset,
    batch_size=512,
    n_epochs=40,
    lr=0.02,
    add_constraints=True
)
if not is_optimal:
    raise RuntimeError

with open(parameter_dict_output_path, "w") as file:
    json.dump(updated_parameter_dict, file)

torch.save(surrogate_model, surrogate_save_path)
torch.save(optimizer, optimizer_save_path)