from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def ddpm_schedules(beta1: float, beta2: float, n_time_steps: int) -> dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert 0.0 < beta1 < beta2 < 1.0, "Condition 0.0 < 'beta 1' < 'beta 2' < 1.0 not fulfilled"

    beta_t = (beta2 - beta1) * torch.arange(0, n_time_steps + 1, dtype=torch.float32) / n_time_steps + beta1
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
    

class SurrogateDataset(Dataset):
    """ Dataset class for the Surrogate model

    Args:
    ----
        df (pd.DataFrame): A DataFrame containing the following keys:
        
            ["Parameters", "Context", "Loss"]

    TODO: Accomodate for discrete parameters
    """
    def __init__(
            self,
            input_df: pd.DataFrame,
            parameter_key: str = "Parameters",
            context_key: str = "Context",
            reco_loss_key: str = "Loss",
            device: str = "cuda",
            norm_reco_loss: bool = True
            ):
        """
        Initializes the Surrogate model with the provided DataFrame and keys. All inputs must be
        unnormalized and will be kept as such. Normalization for training is left to the caller.

        Args:
        ----
            input_df (pd.DataFrame): The input DataFrame containing the data.
            parameter_key (str, optional): The key for the parameters column in the DataFrame.
                Defaults to "Parameters".
            context_key (str, optional): The key for the context column in the DataFrame.
                Defaults to "Context".
            reco_loss_key (str, optional): The key for the reconstruction loss column in the DataFrame.
                Defaults to "Loss".
        """
        self.df = input_df
        self.parameters = self.df[parameter_key].to_numpy(np.float32)
        self.context = self.df[context_key].to_numpy(np.float32)
        self.reconstructed = self.df[reco_loss_key].to_numpy(np.float32)
        self.norm_reco_loss = norm_reco_loss

        assert np.all(self.reconstructed >= 0.0), "Reconstruction Loss must only have positive entries"

        self.shape: List[int] = (
            self.parameters.shape[1],
            self.context.shape[1],
            self.reconstructed.shape[1]
        )
        self.means: List[np.float32] = [
            np.mean(self.parameters, axis=0),
            np.mean(self.context, axis=0),
            np.mean(self.reconstructed, axis=0)
        ]
        self.stds: List[np.float32] = [
            np.std(self.parameters, axis=0) + 1e-10,
            np.std(self.context, axis=0) + 1e-10,
            np.std(self.reconstructed, axis=0) + 1e-10
        ]
        self.c_means = [torch.tensor(a).to(device) for a in self.means]
        self.c_stds = [torch.tensor(a).to(device) for a in self.stds]

        if self.norm_reco_loss is True:
            self.reconstructed = self.normalise_reconstructed(self.reconstructed)

        self.df = self.filter_infs_and_nans(self.df)

    def filter_infs_and_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def unnormalise_features(self, target: torch.Tensor, index: int) -> torch.Tensor:
        ''' Return the physically meaningful target from the normalised target
        Index:
            0 -> Parameters
            1 -> Context
            2 -> Reconstructed
        '''
        assert index in [0, 1, 2]
        if index == 2:
            return self.unnormalise_reconstructed(target)
        elif index == 0 or index == 1:
            return target * self.c_stds[index] + self.c_means[index]

    def normalise_features(self, target: torch.Tensor, index: int) -> torch.Tensor:
        ''' Normalize a feature
        Index:
            0 -> Parameters
            1 -> Context
            2 -> Reconstructed
        '''
        assert index in [0, 1, 2]
        if index == 2:
            return self.normalise_reconstructed(target)
        elif index == 0 or index == 1:
            return (target - self.c_means[index]) / self.c_stds[index]

    def normalise_reconstructed(
            self,
            target: torch.Tensor | np.ndarray
            ) -> torch.Tensor | np.ndarray:
        """ Normalize the Reconstruction Loss. Important if the Loss is highly negatively skewed.
        Currently, no normalization is applied
        """
        if isinstance(target, torch.Tensor):
            return torch.log(target + 1e-10)
        elif isinstance(target, np.ndarray):
            return np.log(target + 1e-10)

    def unnormalise_reconstructed(
            self,
            target: torch.Tensor | np.ndarray
            ) -> torch.Tensor | np.ndarray:
        if isinstance(target, torch.Tensor):
            return torch.exp(target)
        elif isinstance(target, np.ndarray):
            return np.exp(target)

    def __getitem__(self, idx: int):
        return self.parameters[idx], self.context[idx], self.reconstructed[idx]

    def __len__(self) -> int:
        return len(self.reconstructed)


class Surrogate(torch.nn.Module):
    """ Surrogate model class and the surrogate model training function, given a dataset consisting of events.
    The surrogate model itself can be very simple. It is just a feed-forward model but used as a diffusion model.

    Attributes:
        betas (Tuple[float]): Tuple containing the start and end beta values for the diffusion process.
        t_is (torch.Tensor): Tensor containing time steps normalized by the number of time steps.

    Methods:
        forward(parameters, context, reconstructed, time_step):
            Forward pass of the model. Concatenates the input features and passes them through the network.
        to(device=None):
            Moves the model and its buffers to the specified device.
        create_noisy_input(x):
            Adds noise to a tensor for the diffusion process.
        sample_forward(parameters, context):
            Samples from the model in a forward pass using the diffusion process.
        train_model(surrogate_dataset, batch_size, n_epochs, lr):
            Trains the surrogate diffusion model using the provided dataset.
        apply_model_in_batches(dataset, batch_size, oversample=1):
            Applies the model to the dataset in batches and returns the results.
    """

    def __init__(
            self,
            num_parameters: int,
            num_context: int,
            num_reconstructed: int,
            n_time_steps: int = 100,
            betas: Tuple[float] = (1e-4, 0.02)
            ):
        """
        Initializes the surrogate model.
        Args:
            num_parameters (int): Number of input parameters.
            num_context (int): Number of context variables.
            num_reconstructed (int): Number of reconstructed variables.
            n_time_steps (int, optional): Number of time steps for the DDPM schedule. Defaults to 100.
            betas (Tuple[float], optional): Tuple containing the start and end values for the beta schedule.
                Defaults to (1e-4, 0.02).
        """
        super().__init__()

        self.num_parameters = num_parameters
        self.num_context = num_context
        self.num_reconstructed = num_reconstructed
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.num_parameters + self.num_context + self.num_reconstructed + 1, 200),
            torch.nn.ELU(),
            torch.nn.Linear(200, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, num_reconstructed),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.surrogate_loss = []
        self.n_time_steps = n_time_steps
        self.device = torch.device('cuda')
        self.t_is = torch.tensor([i / self.n_time_steps for i in range(self.n_time_steps + 1)]).to(self.device)

        for k, v in ddpm_schedules(*betas, n_time_steps).items():
            self.register_buffer(k, v)

    def forward(
            self,
            parameters: torch.Tensor,
            context: torch.Tensor,
            reconstructed: torch.Tensor,
            time_step: torch.Tensor
            ):
        """ When sampling forward, 'parameters' has only one entry, therefore it is broadcast to
        the shape of 'context'.
        """
        assert (
            context.shape[0] == reconstructed.shape[0]
        ), "Context and Reconstructed inputs have unequal lengths"

        if parameters.shape[0] == 1:
            parameters = parameters.repeat(context.shape[0], 1)

        return self.layers(torch.cat([parameters, context, reconstructed, time_step.view(-1, 1)], dim=1))

    def to(self, device: str = None):
        if device is None:
            device = self.device

        super().to(device)
        self.device = device
        self.sqrtab = self.sqrtab.to(device)
        self.sqrtmab = self.sqrtmab.to(device)
        return self

    def create_noisy_input(
            self,
            x: torch.Tensor,
            scale: float = 1.0
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Add gaussian noise to a tensor. Scale the noise with 'scale', by default the noise is N(0, 1).

        Args:
            x (torch.Tensor): The input tensor to which noise will be added.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The noisy tensor.
                - torch.Tensor: The noise added to the input tensor.
                - torch.Tensor: The time steps used for generating the noise.
        """
        _ts = torch.randint(1, self.n_time_steps + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_time_steps)
        noise = scale * torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrtab[_ts, None] * x + self.sqrtmab[_ts, None] * noise
        return x_t, noise, _ts

    def sample_forward(
            self,
            parameters: torch.Tensor,
            context: torch.Tensor,
            ) -> torch.Tensor:

        n_sample = context.shape[0]
        predicted_reco = torch.randn(n_sample, 1).to(self.device)  # x_0 ~ N(0, 1)

        for i in range(self.n_time_steps, 0, -1):
            t_is = self.t_is[i]
            t_is = t_is.repeat(n_sample, 1)
            z = torch.randn(n_sample, 1).to(self.device) if i > 1 else 0

            # Split predictions and compute weighting
            eps = self(parameters, context, predicted_reco, t_is)
            predicted_reco = (
                self.oneover_sqrta[i] * (predicted_reco - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
            )
        return predicted_reco

    def train_model(
            self,
            surrogate_dataset: SurrogateDataset,
            batch_size: int,
            n_epochs: int,
            lr: float
            ) -> float:
        """ Train the Surrogate Diffusion model. The training loop includes the added
        noise.
        """
        train_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        self.optimizer.lr = lr
        self.to(self.device)
        self.train()
        self.loss_mse = torch.nn.MSELoss()

        for epoch in range(n_epochs):

            for batch_idx, (parameters, context, reconstructed) in enumerate(train_loader):
                parameters: torch.Tensor = parameters.to(self.device)
                reconstructed: torch.Tensor = reconstructed.to(self.device)
                context: torch.Tensor = context.to(self.device)

                reco_noisy, noise, time_step = self.create_noisy_input(reconstructed)
                model_out: torch.Tensor = self(parameters, context, reco_noisy, time_step / self.n_time_steps)

                loss: torch.Tensor = self.loss_mse(noise, model_out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(
                f"Surrogate Epoch: {epoch}",
                f"Loss: {loss.item():.5f}",
                f"Prediction: {self.sample_forward(parameters, context).mean().item():.5f}",
                f"Reconstructed: {reconstructed.mean().item():.5f}",
                sep="\t"
            )
            self.surrogate_loss.append(loss.item())

        self.eval()
        return loss.item()

    def apply_model_in_batches(
            self,
            dataset: SurrogateDataset,
            batch_size: int,
            oversample: int = 1,
            unnormalise_results: bool = True
            ):
        """
        Applies the model to the given dataset in batches and returns the results.

        Args:
            dataset (SurrogateDataset): The dataset to apply the model to.
            batch_size (int): The size of each batch.
            oversample (int, optional): The number of times to oversample the dataset. Default is 1.

        Returns:
            tuple: A tuple containing three elements:
                - results (torch.Tensor): The surrogate model's predictions.

        Remarks: In most cases the resulting Tensor with sampled Data is not of importance, only the
            model weights.
        """
        self.to()
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros(oversample * len(dataset), self.num_reconstructed).to('cpu')

        for i_o in range(oversample):

            for batch_idx, (parameters, context, _reconstructed) in enumerate(data_loader):
                print(f'Surrogate batch: {batch_idx} / {len(data_loader)}', end='\r')
                parameters = parameters.to(self.device)
                context = context.to(self.device)

                reco_surrogate = self.sample_forward(parameters, context)

                start_inject_index = i_o * len(dataset) + batch_idx * batch_size
                end_inject_index = i_o * len(dataset) + (batch_idx + 1) * batch_size
                results[start_inject_index: end_inject_index] = reco_surrogate.detach().to('cpu')

        if unnormalise_results:
            results = dataset.unnormalise_reconstructed(results)
        return results
