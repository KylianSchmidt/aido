from typing import List, Tuple, Self

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from aido.logger import logger


def ddpm_schedules(beta1: float, beta2: float, n_time_steps: int) -> dict[str, torch.Tensor]:
    """
    :no-index:

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

    TODO: Accommodate for discrete parameters
    """
    def __init__(
            self,
            input_df: pd.DataFrame,
            parameter_key: str = "Parameters",
            context_key: str = "Context",
            target_key: str = "Targets",
            reconstructed_key: str = "Reconstructed",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            means: List[np.float32] | None = None,
            stds: List[np.float32] | None = None,
            normalize_parameters: bool = False,
            ):
        """
        Initializes the Surrogate model with the provided DataFrame and keys. All inputs must be
        unnormalized and will be normalized internally.

        Args:
            input_df (pd.DataFrame): The input DataFrame containing the data.
            parameter_key (str, optional): The key for the parameters column in the DataFrame.
                Defaults to "Parameters".
            context_key (str, optional): The key for the context column in the DataFrame.
                Defaults to "Context".
            target_key (str, optional): The key for the target column in the DataFrame.
                Defaults to "Targets".
            reconstructed_key (str, optional): The key for the reconstruction loss column in the DataFrame.
                Defaults to "Loss".
            device (str): Torch device. Defaults to 'cuda' if available, else 'cpu'.
            means (List[np.float32], optional): Predefined means for normalization. Defaults to None.
            stds (List[np.float32], optional): Predefined standard deviations for normalization. Defaults to None.
            normalize_parameters (bool, optional): Whether to normalize parameters. Defaults to False.
        """
        self.df = input_df
        self.parameters = self.df[parameter_key].to_numpy(np.float32)
        self.context = self.df[context_key].to_numpy(np.float32)
        self.targets = self.df[target_key].to_numpy(np.float32)
        self.reconstructed = self.df[reconstructed_key].to_numpy(np.float32)
        self.normalize_parameters = normalize_parameters

        self.shape: List[int] = (
            self.parameters.shape[1],
            self.context.shape[1],
            self.targets.shape[1],
            self.reconstructed.shape[1]
        )
        if means is None:
            self.means: List[np.float32] = [
                np.mean(self.parameters, axis=0),
                np.mean(self.context, axis=0),
                np.mean(self.targets, axis=0),
            ]
        else:
            self.means = means

        if stds is None:
            self.stds: List[np.float32] = [
                np.std(self.parameters, axis=0) + 1e-10,
                np.std(self.context, axis=0) + 1e-10,
                np.std(self.targets, axis=0) + 1e-10,
            ]
        else:
            self.stds = stds

        self.c_means = [torch.tensor(a).to(device) for a in self.means]
        self.c_stds = [torch.tensor(a).to(device) for a in self.stds]

        if self.normalize_parameters:
            logger.info("Normalized parameters")
            self.parameters = self.normalize_features(self.parameters, index=0)

        self.context = self.normalize_features(self.context, index=1)
        self.targets = self.normalize_features(self.targets, index=2)
        self.reconstructed = self.normalize_features(self.reconstructed, index=2)
        self.df = self.filter_infs_and_nans(self.df)

    def filter_infs_and_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def unnormalize_features(
            self,
            target: torch.Tensor | np.ndarray,
            index: int
            ) -> torch.Tensor | np.ndarray:
        """Convert normalized features back to their original scale.
        
        Parameters
        ----------
        target : torch.Tensor or np.ndarray
            The normalized features to convert back.
        index : int
            Index indicating the feature type:
            - 0: Parameters
            - 1: Context
            - 2: Targets

        Returns
        -------
        torch.Tensor or np.ndarray
            The unnormalized features in their original scale.
        """
        if isinstance(target, torch.Tensor):
            return target * self.c_stds[index] + self.c_means[index]
        elif isinstance(target, np.ndarray):
            return target * self.stds[index] + self.means[index]

    def normalize_features(
            self,
            target: torch.Tensor | np.ndarray,
            index: int,
            ) -> torch.Tensor | np.ndarray:
        """Normalize a feature using stored means and standard deviations.

        Parameters
        ----------
        target : torch.Tensor or np.ndarray
            The feature to normalize.
        index : int
            Index indicating the feature type:
            - 0: Parameters
            - 1: Context
            - 2: Targets

        Returns
        -------
        torch.Tensor or np.ndarray
            The normalized feature.
        """
        if isinstance(target, torch.Tensor):
            return (target - self.c_means[index]) / self.c_stds[index]
        elif isinstance(target, np.ndarray):
            return (target - self.means[index]) / self.stds[index]

    def __getitem__(self, idx: int):
        return self.parameters[idx], self.context[idx], self.targets[idx], self.reconstructed[idx]

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
            num_targets: int,
            num_reconstructed: int,
            initial_means: List[np.float32],
            initial_stds: List[np.float32],
            n_time_steps: int = 50,
            betas: Tuple[float] = (1e-4, 0.02),
            ):
        """
        Initializes the surrogate model.

        Args:
            num_parameters (int): Number of input parameters.
            num_context (int): Number of context variables.
            num_reconstructed (int): Number of reconstructed variables.
            n_time_steps (int, optional): Number of time steps for the DDPM schedule. Defaults to 50. Setting
                it higher might lead to divergence towards infinity which will register as NaN when reaching
                float32 accuracy ~= 2e9.
            betas (Tuple[float], optional): Tuple containing the start and end values for the beta schedule.
                Defaults to (1e-4, 0.02).
        """
        super().__init__()

        self.num_parameters = num_parameters
        self.num_context = num_context
        self.num_targets = num_targets
        self.num_reconstructed = num_reconstructed
        self.means = initial_means
        self.stds = initial_stds
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(
                self.num_parameters
                + self.num_context
                + self. num_targets
                + self.num_reconstructed
                + 1, 100
            ),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, num_reconstructed),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_mse = torch.nn.MSELoss()
        self.surrogate_loss = []
        self.n_time_steps = n_time_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_is = torch.tensor([i / self.n_time_steps for i in range(self.n_time_steps + 1)]).to(self.device)
        self.best_surrogate_loss = 1e10

        for k, v in ddpm_schedules(*betas, n_time_steps).items():
            self.register_buffer(k, v)

    def forward(
            self,
            parameters: torch.Tensor,
            context: torch.Tensor,
            targets: torch.Tensor,
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

        return self.layers(torch.cat([
            parameters, context, targets, reconstructed, time_step.view(-1, 1)
        ], dim=1))

    def to(self, device: str | None = None) -> Self:
        """Move whole model and data to device

        Args:
            device (str | None). Name of the device, for example "cuda" or "cpu". Default to None

        Returns:
            Self        
        """
        if device is None:
            device = self.device

        super().to(device)
        self.device = device
        self.sqrtab = self.sqrtab.to(device)
        self.sqrtmab = self.sqrtmab.to(device)
        return self

    def update_best_surrogate_loss(self, loss: torch.Tensor) -> bool:
        if loss < self.best_surrogate_loss:
            self.best_surrogate_loss = loss
            return True

        return loss < 4.0 * self.best_surrogate_loss

    def create_noisy_input(
            self,
            x: torch.Tensor,
            scale: float = 1.0
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add gaussian noise to a tensor.
        
        Scale the noise with 'scale', by default the noise is N(0, 1).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to which noise will be added.
        scale : float, default=1.0
            The scale factor for the noise.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - The noisy tensor
            - The noise added to the input tensor
            - The time steps used for generating the noise
        """
        _ts = torch.randint(1, self.n_time_steps + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_time_steps)
        noise = scale * torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = self.sqrtab[_ts, None] * x + self.sqrtmab[_ts, None] * noise
        return x_t, noise, _ts

    def sample_forward(
            self,
            parameters: torch.Tensor,
            context: torch.Tensor,
            targets: torch.Tensor,
            ) -> torch.Tensor:

        n_sample = context.shape[0]
        predicted_reco = torch.randn(n_sample, 1).to(self.device)  # x_0 ~ N(0, 1)

        for i in range(self.n_time_steps, 0, -1):
            t_is = self.t_is[i]
            t_is = t_is.repeat(n_sample, 1)
            z = torch.randn(n_sample, 1).to(self.device) if i > 1 else 0

            # Split predictions and compute weighting
            eps = self(parameters, context, targets, predicted_reco, t_is)
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
        """Train the Surrogate Diffusion model.
        
        The training loop includes noise addition as part of the diffusion process.
        
        Parameters
        ----------
        surrogate_dataset : SurrogateDataset
            The dataset containing the training data.
        batch_size : int
            The size of each training batch.
        n_epochs : int
            The number of training epochs.
        lr : float
            The learning rate for the optimizer.
            
        Returns
        -------
        float
            The final training loss value.
        """
        train_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.to(self.device)
        self.train()

        for epoch in range(n_epochs):

            for batch_idx, (parameters, context, targets, reconstructed) in enumerate(train_loader):
                parameters: torch.Tensor = parameters.to(self.device)
                context: torch.Tensor = context.to(self.device)
                targets: torch.Tensor = targets.to(self.device)
                reconstructed: torch.Tensor = reconstructed.to(self.device)

                reco_noisy, noise, time_step = self.create_noisy_input(reconstructed)
                model_out: torch.Tensor = self(parameters, context, targets, reco_noisy, time_step / self.n_time_steps)

                loss: torch.Tensor = self.loss_mse(noise, model_out)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            logger.info(
                f"Surrogate Epoch: {epoch}\t"
                f"Loss: {loss.item():.5f}\t"
                f"Prediction: {self.sample_forward(parameters, context, targets).mean().item():.5f}\t"
                f"Reconstructed: {reconstructed.mean().item():.5f}",
            )
            self.surrogate_loss.append(loss.item())

        self.eval()
        return loss.item()

    def apply_model_in_batches(
            self,
            dataset: SurrogateDataset,
            batch_size: int,
            oversample: int = 1,
            ) -> torch.Tensor:
        """Apply the model to the given dataset in batches.

        Parameters
        ----------
        dataset : SurrogateDataset
            The dataset to apply the model to.
        batch_size : int
            The size of each batch.
        oversample : int, optional
            The number of times to oversample the dataset, by default 1.

        Returns
        -------
        torch.Tensor
            The surrogate model's predictions.

        Notes
        -----
        In most cases, the resulting tensor with sampled data is not of importance,
        as the main value lies in the trained model weights.
        """
        self.to()
        self.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros(oversample * len(dataset), self.num_reconstructed).to('cpu')

        for i_o in range(oversample):

            for batch_idx, (parameters, context, targets, _reconstructed) in enumerate(data_loader):
                logger.info(f'Surrogate batch: {batch_idx} / {len(data_loader)}')
                parameters: torch.Tensor = parameters.to(self.device)
                context: torch.Tensor = context.to(self.device)
                targets: torch.Tensor = targets.to(self.device)

                reco_surrogate = self.sample_forward(parameters, context, targets)
                reco_surrogate = dataset.unnormalize_features(reco_surrogate, index=2)

                start_inject_index = i_o * len(dataset) + batch_idx * batch_size
                end_inject_index = i_o * len(dataset) + (batch_idx + 1) * batch_size
                results[start_inject_index: end_inject_index] = reco_surrogate.detach().to('cpu')

        return results
