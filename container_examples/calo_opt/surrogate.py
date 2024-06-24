import torch
from generator import CaloDataset
from torch.utils.data import DataLoader


class SurrogateDataset(CaloDataset):
    '''
    This dataset is initialised with an existing CaloDataset and copies most of its behaviour and data.
    However, it adds an additional array that contains the output of the reconstruction model for each event.

    Returns in iterator:
    - detector parameters
    - true inputs
    - reco output
    '''
    def __init__(self, calo_dataset, reco_output):
        super().__init__(
            calo_dataset.sensor_array,
            calo_dataset.detector_array,
            calo_dataset.target_array,
            calo_dataset.context_array
        )

        self.reco_output = reco_output
        self.c_means = calo_dataset.c_means
        self.c_stds = calo_dataset.c_stds
        assert len(self.reco_output) == len(self.sensor_array), "Reco output and sensor array have different lengths!"

    def __getitem__(self, idx):
        return self.detector_array[idx], self.target_array[idx], self.context_array[idx], self.reco_output[idx]
    

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

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

    def __init__(self, n_time_steps, betas=(1e-4, 0.02)):
        super().__init__()

        self.n_time_steps = n_time_steps

        for k, v in ddpm_schedules(betas[0], betas[1], n_time_steps).items():
            self.register_buffer(k, v)

    def forward(self, x, t):
        """
        x: (B, C, H, W)
        t: (B, 1)
        """
        # x: (B, C, H, W)
        # t: (B, 1)
        # z: (B, C, H, W)
        z = torch.randn_like(x)  # eps ~ N(0, 1)
        # x_t: (B, C, H, W)
        x_t = self.sqrtab[t, None] * x + self.sqrtmab[t, None] * z
        return x_t, z


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
            n_detector_parameters,
            n_true_context_inputs,
            n_reco_parameters,
            n_time_steps,
            betas=(1e-4, 0.02)
            ):
        super(Surrogate, self).__init__()

        self.n_detector_parameters = n_detector_parameters

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_detector_parameters + n_true_context_inputs + n_reco_parameters + 1, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, 100),
            torch.nn.ELU(),
            torch.nn.Linear(100, n_reco_parameters),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.n_reco_parameters = n_reco_parameters
        self.n_time_steps = n_time_steps

        for k, v in ddpm_schedules(betas[0], betas[1], n_time_steps).items():
            self.register_buffer(k, v)

        self.loss_mse = torch.nn.MSELoss()
        self.device = torch.device('cuda')
        self.t_is = torch.tensor([i / self.n_time_steps for i in range(self.n_time_steps + 1)]).to(self.device)

    def forward(self, detector_parameters, true_inputs, true_context, reco_step_inputs, time_step):
        """ Concatenate the detector parameters and the input
        print all dimensions for debugging

        In case the detector parameters don't have batch dimension:
        tile them here to have same first dimension as 'true_inputs'
        """
        
        time_step = time_step.view(-1, 1)

        if len(detector_parameters.shape) == 1:
            detector_parameters = detector_parameters.repeat(true_inputs.shape[0], 1)

        x = torch.cat([detector_parameters, true_inputs, true_context, reco_step_inputs, time_step], dim=1)
        return self.layers(x)
    
    def to(self, device=None):
        if device is None:
            device = self.device

        super().to(device)
        self.device = device
        self.sqrtab = self.sqrtab.to(device)
        self.sqrtmab = self.sqrtmab.to(device)
        return self
    
    def create_noisy_input(self, reco_step_inputs):
        x = reco_step_inputs
        _ts = torch.randint(1, self.n_time_steps + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_time_steps)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # add the noise
        x_t = self.sqrtab[_ts, None] * x + self.sqrtmab[_ts, None] * noise
        return x_t, noise, _ts

    def sample_forward(self, detector_parameters, true_inputs, true_context):
        n_sample = true_inputs.shape[0]
        x_i = torch.randn(n_sample, 1).to(self.device)  # x_0 ~ N(0, 1)
        
        for i in range(self.n_time_steps, 0, -1):
            t_is = self.t_is[i]
            t_is = t_is.repeat(n_sample, 1)
            z = torch.randn(n_sample, 1).to(self.device) if i > 1 else 0

            # Split predictions and compute weighting
            eps = self(detector_parameters, true_inputs, true_context, x_i, t_is)
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

        train_loader = DataLoader(surrogate_dataset, batch_size=batch_size, shuffle=True)
        self.optimizer.lr = lr
        self.to(self.device)
        
        self.train()

        for epoch in range(n_epochs):
            for batch_idx, (detector_parameters, true_inputs, true_context, reco_result) in enumerate(train_loader):
                # this needs to be adapted since it is a diffusion model. so the noise loop needs to be in here
                # the noise loop is the same as in the generator
                
                detector_parameters = detector_parameters.to(self.device)
                true_inputs = true_inputs.to(self.device)
                reco_step_inputs = reco_result.to(self.device)
                true_context = true_context.to(self.device)
                # the noise loop
                x_t, noise, _ts = self.create_noisy_input(reco_step_inputs)
                # apply the model
                model_out = self(detector_parameters, true_inputs, true_context, x_t, _ts / self.n_time_steps)

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
        results = torch.zeros(oversample * len(dataset), self.n_reco_parameters).to('cpu')
        reco = torch.zeros(oversample * len(dataset), self.n_reco_parameters).to('cpu')
        true = torch.zeros(oversample * len(dataset), self.n_reco_parameters).to('cpu')

        for i_o in range(oversample):
            for batch_idx, (detector_parameters, true_inputs, true_context, reco_inputs) in enumerate(data_loader):  # the reco is not needed as it is generated here
                
                print(f'batch {batch_idx} of {len(data_loader)}', end='\r')
                detector_parameters = detector_parameters.to(self.device)
                true_inputs = true_inputs.to(self.device)
                true_context = true_context.to(self.device)
                reco_inputs = reco_inputs.to(self.device)
                # apply the model
                reco_surrogate = self.sample_forward(detector_parameters, true_inputs, true_context)

                # un_normalise all to physical values
                reco_surrogate = dataset.unnormalise_target(reco_surrogate)
                reco_inputs = dataset.unnormalise_target(reco_inputs)
                true_inputs = dataset.unnormalise_target(true_inputs)

                # store the results
                start_inject_index = i_o * len(dataset) + batch_idx * batch_size
                end_inject_index = i_o * len(dataset) + (batch_idx + 1) * batch_size
                results[start_inject_index: end_inject_index] = reco_surrogate.detach().to('cpu')
                reco[start_inject_index: end_inject_index] = reco_inputs.detach().to('cpu')
                true[start_inject_index: end_inject_index] = true_inputs.detach().to('cpu')

        return results, reco, true
