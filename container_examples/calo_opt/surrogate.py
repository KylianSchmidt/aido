import numpy as np
import pandas as pd
import torch
from typing import Tuple
from torch.utils.data import DataLoader
from reconstruction import ReconstructionDataset


def ddpm_schedules(beta1: float, beta2: float, T: int):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in ]0, 1["

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class NoiseAdder(torch.nn.Module):

    def __init__(self, n_time_steps: int, betas=(1e-4, 0.02)):
        super().__init__()

        self.n_time_steps = n_time_steps

        for k, v in ddpm_schedules(*betas, n_time_steps).items():
            self.register_buffer(k, v)

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B, 1)
        z: (B, C, H, W)
        x_t: (B, C, H, W)
        """
        z = torch.randn_like(x)  # eps ~ N(0, 1)
        x_t = self.sqrtab[t, None] * x + self.sqrtmab[t, None] * z
        return x_t, z
    

class SurrogateDataset(ReconstructionDataset):
    """This dataset requires an existing ReconstructionDataset instance. This dataset
    adds the output of the reconstruction model.

    Args:
    ----
        reco_dataset (ReconstructionDataset): An existing ReconstructionDataset instance.
        reconstructed_array (np.ndarray): The reconstructed array to be added to the dataset. Must not be normailized.

    Attributes:
    ----------
        df (pd.DataFrame): The concatenated dataframe containing the original dataset and the reconstructed array.

        parameters (np.ndarray): The parameters from the original dataset.
        targets (np.ndarray): The targets from the original dataset.
        context (np.ndarray): The context from the original dataset.
        reconstructed (np.ndarray): The reconstructed array. Columns are identical to 'targets'. MUST REMAIN NORMALISED

        shape (tuple): The shape of the dataset: [parameters, targets, context, reconstructed].

        means (list): The means of the original dataset and the reconstructed array.
        stds (list): The standard deviations of the original dataset and the reconstructed array.
        c_means (list): The means converted to torch tensors and moved to the 'cuda' device.
        c_stds (list): The standard deviations converted to torch tensors and moved to the 'cuda' device.

    Methods:
    -------
        __getitem__(self, idx: int): [parameters, targets, context, reconstructed] at the given index

    TODO: Means and Stds are ordered differently to the Reconstruction Dataset
    """

    def __init__(
            self,
            reco_dataset: ReconstructionDataset,
            reconstructed_array: np.ndarray
            ):
        reconstructed_array = reconstructed_array * reco_dataset.stds[2] + reco_dataset.means[2]
        reconstructed_df = pd.DataFrame(reconstructed_array, columns=reco_dataset.df["Targets"].columns)
        reconstructed_df = pd.concat({"Reconstructed": reconstructed_df}, axis=1)
        self.df: pd.DataFrame = pd.concat([reco_dataset.df, reconstructed_df], axis=1)
        
        self.parameters = reco_dataset.parameters
        self.targets = reco_dataset.targets
        self.context = reco_dataset.context
        self.reconstructed = self.df["Reconstructed"].to_numpy("float32")

        self.shape = (
            self.parameters.shape[1],
            self.targets.shape[1],
            self.context.shape[1],
            self.reconstructed.shape[1]
        )
        self.means = [
            reco_dataset.means[0],
            reco_dataset.means[2],
            reco_dataset.means[3],
            reco_dataset.means[2],
        ]
        self.stds = [
            reco_dataset.stds[0],
            reco_dataset.stds[2],
            reco_dataset.stds[3],
            reco_dataset.stds[2],
        ]

        self.c_means = [torch.tensor(a).to('cuda') for a in self.means]
        self.c_stds = [torch.tensor(a).to('cuda') for a in self.stds]
        self.df = self.filter_infs_and_nans(self.df)

    def __getitem__(self, idx):
        return self.parameters[idx], self.targets[idx], self.context[idx], self.reconstructed[idx]
    
    def __len__(self):
        return len(self.reconstructed)


class Surrogate(torch.nn.Module):
    """ Surrogate model class
    and the surrogate model training function, given a dataset consisting of
    events. for each event, the entries are
    - reco_energy
    - detector parameters and true_energy for conditioning as numpy array

    the surrogate model itself can be very very simple. It is just a feed forward model but used as a diffusion model
    the training loop is also in here
    """

    def __init__(
            self,
            num_parameters: int,
            num_targets: int,
            num_context: int,
            num_reconstructed: int,
            n_time_steps=100,
            betas: Tuple[float] = (1e-4, 0.02)
            ):
        super().__init__()

        self.num_parameters = num_parameters
        self.num_targets = num_targets
        self.num_context = num_context
        self.num_reconstructed = num_reconstructed
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.num_parameters + self.num_targets + self.num_context + self.num_reconstructed + 1, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, num_reconstructed),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.n_time_steps = n_time_steps

        for k, v in ddpm_schedules(betas[0], betas[1], n_time_steps).items():
            self.register_buffer(k, v)

        self.loss_mse = torch.nn.MSELoss()
        self.device = torch.device('cuda')
        self.t_is = torch.tensor([i / self.n_time_steps for i in range(self.n_time_steps + 1)]).to(self.device)

    def forward(
            self,
            parameters: torch.Tensor,
            targets: torch.Tensor,
            context: torch.Tensor,
            reconstructed: torch.Tensor,
            time_step: torch.Tensor
            ):
        """ Concatenate the detector parameters and the input
        print all dimensions for debugging

        In case the detector parameters don't have batch dimension:
        tile them here to have same first dimension as 'targets'
        """
        time_step = time_step.view(-1, 1)

        x = torch.cat([parameters, targets, context, reconstructed, time_step], dim=1)
        return self.layers(x)
    
    def to(self, device=None):
        if device is None:
            device = self.device

        super().to(device)
        self.device = device
        self.sqrtab = self.sqrtab.to(device)
        self.sqrtmab = self.sqrtmab.to(device)
        return self
    
    def create_noisy_input(self, x: torch.Tensor):
        """ Add noise to a tensor
        """
        _ts = torch.randint(1, self.n_time_steps + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_time_steps)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrtab[_ts, None] * x + self.sqrtmab[_ts, None] * noise
        return x_t, noise, _ts

    def sample_forward(
            self,
            parameters: torch.Tensor,
            targets: torch.Tensor,
            context: torch.Tensor,
            ):
        n_sample = targets.shape[0]
        x_i = torch.randn(n_sample, 1).to(self.device)  # x_0 ~ N(0, 1)
        
        for i in range(self.n_time_steps, 0, -1):
            t_is = self.t_is[i]
            t_is = t_is.repeat(n_sample, 1)
            z = torch.randn(n_sample, 1).to(self.device) if i > 1 else 0

            # Split predictions and compute weighting
            eps = self(parameters, targets, context, x_i, t_is)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            )
    
        return x_i
    
    def train_model(
            self,
            surrogate_dataset: SurrogateDataset,
            batch_size: int,
            n_epochs: int,
            lr: float
            ):
        """ Train the Surrogate Diffusion model. The training loop includes the added
        noise.
        """

        train_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        self.optimizer.lr = lr
        self.to(self.device)
        self.train()

        for epoch in range(n_epochs):

            for batch_idx, (parameters, targets, context, reco_result) in enumerate(train_loader):
                parameters = parameters.to(self.device)
                targets = targets.to(self.device)
                reconstructed = reco_result.to(self.device)
                context = context.to(self.device)

                x_t, noise, _ts = self.create_noisy_input(reconstructed)
                model_out = self(parameters, targets, context, x_t, _ts / self.n_time_steps)

                loss = self.loss_mse(noise, model_out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Surrogate Epoch: {epoch} \tLoss: {loss.item():.8f}')

        self.eval()
        return loss.item()

    def apply_model_in_batches(self, dataset: SurrogateDataset, batch_size: int, oversample=1):
        self.to()
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros(oversample * len(dataset), self.num_reconstructed).to('cpu')
        reco = torch.zeros(oversample * len(dataset), self.num_reconstructed).to('cpu')
        true = torch.zeros(oversample * len(dataset), self.num_reconstructed).to('cpu')

        for i_o in range(oversample):

            for batch_idx, (parameters, targets, context, reconstructed) in enumerate(data_loader):
                
                print(f'Surrogate batch: {batch_idx} / {len(data_loader)}', end='\r')
                parameters = parameters.to(self.device)
                targets = targets.to(self.device)
                context = context.to(self.device)
                reconstructed = reconstructed.to(self.device)
                reco_surrogate = self.sample_forward(parameters, targets, context)

                # Unnormalise all to physical values
                reco_surrogate = dataset.unnormalise_target(reco_surrogate)
                reconstructed = dataset.unnormalise_target(reconstructed)
                targets = dataset.unnormalise_target(targets)

                # Store the results
                start_inject_index = i_o * len(dataset) + batch_idx * batch_size
                end_inject_index = i_o * len(dataset) + (batch_idx + 1) * batch_size
                results[start_inject_index: end_inject_index] = reco_surrogate.detach().to('cpu')
                reco[start_inject_index: end_inject_index] = reconstructed.detach().to('cpu')
                true[start_inject_index: end_inject_index] = targets.detach().to('cpu')

        return results, reco, true
