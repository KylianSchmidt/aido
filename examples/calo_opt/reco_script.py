import os
import sys

import pandas as pd
import torch
from reconstruction import Reconstruction, ReconstructionDataset
from torch.utils.data import Dataset
from reconstruction_validation import ReconstructionValidation
import matplotlib.pyplot as plt
import numpy as np
import datetime

def pre_train(model: Reconstruction, dataset: Dataset, n_epochs: int):
    """ Pre-train the  a given model

    TODO Reconstruction results are normalized. In the future only expose the un-normalised ones,
    but also requires adjustments to the surrogate dataset
    """
    
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    print("Reconstruction: Pre-training 0")
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print("Reconstruction: Pre-training 1")
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)
    model.to('cpu')


if __name__ == "__main__":

    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]
    isVal = sys.argv[3].strip().lower() == "true"
    
    results_dir = sys.argv[4]
    
    if not isVal:
        
        n_epochs_pre = 24
        n_epochs_main = 40

        # Load the input df
        simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)
        reco_model_previous_path = os.path.join(results_dir, "reco_model")

        if os.path.exists(reco_model_previous_path):
            reco_model: Reconstruction = torch.load(reco_model_previous_path)
            reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)
        else:
            reco_dataset = ReconstructionDataset(simulation_df)
            reco_model = Reconstruction(*reco_dataset.shape, reco_dataset.means, reco_dataset.stds)
            pre_train(reco_model, reco_dataset, n_epochs_pre)

        # Reconstruction:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        reco_model.to(dev)
        reco_model.train_model(reco_dataset, batch_size=256, n_epochs=n_epochs_main // 4, lr=0.003)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
        reco_result, reco_loss, _ = reco_model.apply_model_in_batches(reco_dataset, batch_size=128)

        reconstructed_df = pd.DataFrame({"true_energy": reco_result})
        reconstructed_df = pd.concat({"Reconstructed": reconstructed_df}, axis=1)
        loss_df = pd.DataFrame({"Reco_loss": reco_loss.tolist()})
        loss_df = pd.concat({"Loss": loss_df}, axis=1)
        output_df: pd.DataFrame = pd.concat([reco_dataset.df, reconstructed_df, loss_df], axis=1)
        output_df.to_parquet(output_df_path)

        torch.save(reco_model, reco_model_previous_path)
        
        # validation
        validator = ReconstructionValidation(reco_model)
        output_df_val = validator.validate(reco_dataset)

        fig_savepath = os.path.join(results_dir, "plots","validation","reco_model","on_trainingData")
        validator.plot(output_df_val,fig_savepath)
        
    else:
        
        reco_model_previous_path = os.path.join(results_dir, "reco_model")
        reco_model: Reconstruction = torch.load(reco_model_previous_path,weights_only=False)
        
        # Load the input df
        simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)
        reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)
        
        # validation
        validator = ReconstructionValidation(reco_model)
        output_df_val = validator.validate(reco_dataset)
        output_df_val.to_parquet(output_df_path)  

        fig_savepath = os.path.join(results_dir, "plots","validation","reco_model","on_validationData")
        validator.plot(output_df_val,fig_savepath)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # val_result, val_loss, _ = reco_model.apply_model_in_batches(reco_dataset, batch_size=512)
        # final_validation_df = pd.DataFrame({"true_energy": val_result})
        # final_validation_df = pd.concat({"Reconstructed": final_validation_df}, axis=1)
        # loss_df_val = pd.DataFrame({"Reco_loss": val_loss.tolist()})
        # loss_df_val = pd.concat({"Loss": loss_df_val}, axis=1)
        # output_df_val: pd.DataFrame = pd.concat([reco_dataset.df, final_validation_df, loss_df_val], axis=1)
        # output_df_val.to_parquet(output_df_path)
        
        # fig_savepath = os.path.join(results_dir, "plots","validation","reco_model")
        
        # bins = np.linspace(0, 20, 100 + 1)
        # val_reco = output_df_val["Reconstructed"]["true_energy"]
        # reco_out = output_df_val["Targets"]

        # plt.hist(val_reco, bins=bins, label="Validation", histtype="step")
        # plt.hist(reco_out, bins=bins, label="Reconstruction", histtype="step")
        # plt.xlim(bins[0], bins[-1])
        # plt.xlabel("Predicted Energy")
        # plt.ylabel(f"Counts / {(bins[1] - bins[0]):.2f}")
        # plt.legend()
        # plt.savefig(os.path.join(fig_savepath, f"validation_loss_{datetime.datetime.now()}.png"))
        # plt.close()

        # bins = np.linspace(-10, 10, 100 + 1)
        
        # print(type(val_reco),type(reco_out))
        # print(val_reco - reco_out)
        # plt.hist((val_reco - reco_out),bins=bins)
        # plt.xlabel("Reco_model Accuracy")
        # plt.xlim(bins[0], bins[-1])
        # plt.savefig(os.path.join(fig_savepath, f"validation_accuracy_{datetime.datetime.now()}.png"))
        # plt.close()
        # print("Validation Plots Saved")

    # n_epochs_pre = 24
    # n_epochs_main = 40

    # # Load the input df
    # simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)

    # reco_model_previous_path = os.path.join(results_dir, "reco_model.pt")

    # if os.path.exists(reco_model_previous_path):
    #     reco_model: Reconstruction = torch.load(reco_model_previous_path)
    #     reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)
    # else:
    #     reco_dataset = ReconstructionDataset(simulation_df)
    #     reco_model = Reconstruction(*reco_dataset.shape, reco_dataset.means, reco_dataset.stds)
    #     pre_train(reco_model, reco_dataset, n_epochs_pre)

    # # Reconstruction:
    # dev = "cuda" if torch.cuda.is_available() else "cpu"
    # reco_model.to(dev)
    # reco_model.train_model(reco_dataset, batch_size=256, n_epochs=n_epochs_main // 4, lr=0.003)
    # reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
    # reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)
    # reco_result, reco_loss, _ = reco_model.apply_model_in_batches(reco_dataset, batch_size=128)
    # print(f"DEBUG nominal reco loss {reco_loss[0:400].mean()}")

    # reconstructed_df = pd.DataFrame({"true_energy": reco_result})
    # reconstructed_df = pd.concat({"Reconstructed": reconstructed_df}, axis=1)
    # loss_df = pd.DataFrame({"Reco_loss": reco_loss.tolist()})
    # loss_df = pd.concat({"Loss": loss_df}, axis=1)
    # output_df: pd.DataFrame = pd.concat([reco_dataset.df, reconstructed_df, loss_df], axis=1)
    # output_df.to_parquet(output_df_path)

    # torch.save(reco_model, reco_model_previous_path)
