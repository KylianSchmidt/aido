import torch
import numpy as np
from torch.utils.data import DataLoader
from surrogate import Surrogate, SurrogateDataset
from typing import Dict


class OneHotEncoder(torch.nn.Module):
    """
    OneHotEncoder is a module that performs one-hot encoding on discrete values.

    Attributes:
        logits (torch.Tensor): A set of unnormalized, real-valued scores for each category. These logits
            represent the model's confidence in each category prior to normalization. They can take any
            real value, including negatives, and are not probabilities themselves. Use the to_probabilities()
            method to convert the logits to probabilities.
    """
    def __init__(self, parameter: dict, temperature: float = 1.0):
        """
        Args:
            parameter (dict): A dictionary containing the parameter information.
            temperature (float, optional): The temperature parameter for gumbel softmax. Defaults to 1.0.
        """
        super().__init__()
        self.discrete_values: list = parameter["discrete_values"]
        self.starting_value = torch.tensor(self.discrete_values.index(parameter["current_value"]))
        self.logits = torch.nn.Parameter(torch.nn.functional.one_hot(self.starting_value, len(self.discrete_values)).float())
        self.temperature = temperature

    def forward(self):
        res = torch.nn.functional.gumbel_softmax(self.logits, tau=self.temperature, hard=True)
        print(f"DEBUG Called forward with parameters {torch.argmax(res)}")
        return res

    @property
    def current_value(self):
        return torch.argmax(self.logits)
    
    @property
    def physical_value(self):
        return torch.tensor(self.discrete_values[self.current_value.item()])

    @property
    def probabilities(self):
        return torch.nn.functional.softmax(self.logits, dim=0)


class ContinuousParameter(torch.nn.Module):
    def __init__(self, parameter: dict):
        super().__init__()
        self.starting_value = torch.tensor(parameter["current_value"])
        self.parameter = torch.nn.Parameter(self.starting_value.clone(), requires_grad=True)
        self.min_value = np.nan_to_num(parameter["min_value"], nan=-10E10)
        self.max_value = np.nan_to_num(parameter["max_value"], nan=10E10)
        self.boundaries = torch.tensor(np.array([self.min_value, self.max_value], dtype="float32"))
        self.sigma = torch.tensor(parameter["sigma"])

    def forward(self):
        return self.parameter

    @property
    def current_value(self):  # TODO Normalization
        return self.parameter
    
    @property
    def physical_value(self):
        return self.current_value  # TODO Keep without normalization


class Optimizer(torch.nn.Module):
    '''
    The optimizer uses the surrogate model to optimise the detector parameters in batches.
    It is also linked to a generator object, to check if the parameters are still in bounds using
    the function is_local(parameters) of the generator.

    Once the parameters are not local anymore, the optimizer will return the last parameters that were local and stop.
    For this purpose, the surrogate model will need to be applied using fixed weights.
    Then the reconstruction model loss will be applied based on the surrogate model output.
    The gradient w.r.t. the detector parameters will be calculated and the parameters will be updated.
    '''
    def __init__(
            self,
            surrogate_model: Surrogate,
            starting_parameter_dict: Dict,
            lr=0.001,
            batch_size=128,
            ):
        
        super().__init__()
        self.surrogate_model = surrogate_model
        self.starting_parameter_dict = starting_parameter_dict
        self.parameter_dict = {k: v for k, v in self.starting_parameter_dict.items() if v.get("optimizable")}
        self.n_time_steps = surrogate_model.n_time_steps
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda"

        self.parameters_all, self.parameters_discrete, self.parameters_continuous = self.parameter_dict_to_cuda()
        self.starting_parameters_continuous = self.module_dict_to_tensor(self.parameters_continuous)
        self.parameter_box = self.parameter_constraints_to_cuda(self.parameters_continuous)
        self.covariance = self.get_covariance_matrix()

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters_all.parameters(), lr=self.lr)

    def parameter_constraints_to_cuda(self, parameters_continuous: torch.nn.ModuleDict):
        # TODO make this part of the Continuous parameter class?
        tensor_list = [parameter.boundaries for parameter in parameters_continuous.values()]
        if tensor_list == []:
            res = torch.tensor([])
        else:
            res = torch.stack(tensor_list)
        return res.to(self.device)

    def module_dict_to_tensor(self, module_dict: torch.nn.ModuleDict):
        # TODO Make this also a part of Continuous parameter class!
        tensor_list = [parameter.current_value for parameter in module_dict.values()]

        if tensor_list == []:
            return torch.tensor([])
        else:
            return torch.stack(tensor_list)

    def parameter_dict_to_cuda(self):
        """ Parameter list as cuda tensor
        Shape: (Parameter, 1)
        """
        parameters_discrete = torch.nn.ModuleDict()
        parameters_continuous = torch.nn.ModuleDict()
        # NOTE Need to consider continuous and discrete parameters seperately then add their Loss

        for name, parameter in self.parameter_dict.items():
            if parameter["discrete_values"] is not None:
                parameters_discrete[name] = OneHotEncoder(parameter)
            else:
                parameters_continuous[name] = ContinuousParameter(parameter)

        parameters_all = torch.nn.ModuleDict()
        parameters_all.update(parameters_discrete)
        parameters_all.update(parameters_continuous)
        return parameters_all, parameters_discrete, parameters_continuous

    def to(self, device: str):
        self.device = device
        self.surrogate_model.to(device)
        self.starting_parameters_continuous.to(device)
        self.parameter_box.to(device)
        super().to(device)
        return self
    
    def get_covariance_matrix(self):
        covariance_matrix = []

        for parameter in self.parameters_continuous.values():
            covariance_matrix.append(parameter.sigma.item())

        return np.diag(np.array(covariance_matrix, dtype="float32"))

    def other_constraints(self, constraints: Dict = {}):
        """ Keep parameters such that within the box size of the generator, there are always some positive values even
        if the central parameters are negative.
        TODO Improve doc string
        TODO Total detector length is an example of possible additional constraints. Very specific use now, must be 
        added in the UserInterface class later on.
        TODO Fix module dict for continuous parameters!
        """
        self.constraints = {key: torch.tensor(value) for key, value in constraints.items()}
        parameters_continuous_tensor = self.module_dict_to_tensor(self.parameters_continuous)

        loss = (
            torch.mean(100. * torch.nn.ReLU()(self.parameter_box[:, 0] - parameters_continuous_tensor)) +
            torch.mean(100. * torch.nn.ReLU()(- self.parameter_box[:, 1] + parameters_continuous_tensor))
        )

        if "length" in self.constraints.keys():
            detector_length = torch.sum(parameters_continuous_tensor)
            loss += torch.mean(100. * torch.nn.ReLU()(detector_length - self.constraints["length"])**2)

        return loss

    def adjust_covariance(self, direction: torch.Tensor, min_scale=2.0):
        """ Stretches the box_covariance of the generator in the directon specified as input.
        Direction is a vector in parameter space
        """
        parameter_direction_vector = direction.detach().cpu().numpy()
        parameter_direction_length = np.linalg.norm(parameter_direction_vector)

        scaling_factor = min_scale * np.max([1., 4. * parameter_direction_length])
        # Create the scaling adjustment matrix
        parameter_direction_normed = parameter_direction_vector / parameter_direction_length
        M_scaled = (scaling_factor - 1) * np.outer(parameter_direction_normed, parameter_direction_normed)
        # Adjust the original covariance matrix
        self.covariance = np.diag(self.covariance**2) + M_scaled
        return np.diag(self.covariance)

    def check_parameter_are_local(self, updated_parameters: torch.Tensor, scale=1.0) -> bool:
        """ Assure that the predicted parameters by the optimizer are within the bounds of the covariance
        matrix spanned by the 'sigma' of each parameter.
        """
        diff = updated_parameters - self.module_dict_to_tensor(self.parameters_continuous)
        diff = diff.detach().cpu().numpy()

        if self.covariance.ndim == 1:
            self.covariance = np.diag(self.covariance)

        return np.dot(diff, np.dot(np.linalg.inv(self.covariance), diff)) < scale

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """ Loss function for the optimizer. TODO Should be the same loss as the Reconstruction model
        but since they are in different containers, that will be tricky to implement.
        """
        y_true = torch.where(torch.isnan(y_pred), torch.zeros_like(y_true) + 1., y_true)
        y_true = torch.where(torch.isinf(y_pred), torch.zeros_like(y_true) + 1., y_true)
        y_pred = torch.where(torch.isnan(y_pred), torch.zeros_like(y_pred), y_pred)
        y_pred = torch.where(torch.isinf(y_pred), torch.zeros_like(y_pred), y_pred)
        return ((y_pred - y_true)**2 / (torch.abs(y_true) + 1.)).mean()

    def optimize(
            self,
            dataset: SurrogateDataset,
            batch_size: int,
            n_epochs: int,
            lr: float,
            add_constraints=True,
            ):
        """ Keep Surrogate model fixed, train only the detector parameters (self.detector_start_parameters)
        TODO Improve documentation of this method.
        """
        self.optimizer.lr = lr
        self.surrogate_model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer_loss = []
        self.constraints_loss = []

        for epoch in range(n_epochs):
            epoch_loss = 0
            stop_epoch = False

            for batch_idx, (_parameters, targets, true_context, reco_result) in enumerate(data_loader):
                targets = targets.to(self.device)
                true_context = true_context.to(self.device)
                reco_result = reco_result.to(self.device)

                ohe = self.parameters_all["num_blocks"]()
                #print("DEBUG OHE value in training loop: ", torch.argmax(ohe))
                reco_surrogate = self.surrogate_model.sample_forward(
                    self.module_dict_to_tensor(self.parameters_all),
                    targets,
                    true_context
                )  # TODO module dict, normalize parameters??
                reco_surrogate = reco_surrogate * dataset.c_stds[1] + dataset.c_means[1]
                targets = targets * dataset.c_stds[1] + dataset.c_means[1]
                loss = self.loss(reco_surrogate, targets)

                if add_constraints:
                    loss += self.other_constraints()

                self.optimizer.zero_grad()
                loss.backward()

                if np.isnan(loss.item()):
                    # Save parameters, reset the optimizer as if it made a step but without updating the parameters
                    print("Optimizer: NaN loss, exiting.")
                    prev_parameters = self.module_dict_to_tensor(self.parameters_all)
                    self.optimizer.step()

                    for i, parameter in enumerate(self.parameters_all):
                        parameter.data = prev_parameters[i].to(self.device)

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = float(prev_parameters[index])

                    return self.parameter_dict, False, epoch_loss / (batch_idx + 1)

                self.optimizer.step()
                #print(f"DEBUG OHE parameter after step(): {self.parameters_all["num_blocks"].logits}")
                #print(f"DEBUG float parameter after step(): {self.parameters_all["thickness_scintillator"].current_value}")    
            
                epoch_loss += loss.item()

                if not self.check_parameter_are_local(self.module_dict_to_tensor(self.parameters_continuous)):
                    stop_epoch = True
                    break

                if batch_idx % 20 == 0:
                    
                    parameter_array = self.module_dict_to_tensor(self.parameters_all).cpu().detach().numpy()

                    for index, key in enumerate(self.parameter_dict):
                        self.parameter_dict[key] = float(parameter_array[index])  # TODO correct dtype

            print(
                f"Optimizer Epoch: {epoch} \tLoss: {(self.loss(reco_surrogate, targets)):.5f} (reco)\t"
                #f"+ {(self.other_constraints()):.5f} (constraints)\t = {loss.item():.5f} (total)"
                f"   |    Parameter Dict {self.module_dict_to_tensor(self.parameters_all)}\t"
            )
            epoch_loss /= batch_idx + 1
            self.optimizer_loss.append(epoch_loss)
            #self.constraints_loss.append(self.other_constraints().detach().cpu().numpy())

            if stop_epoch:
                break

        self.covariance = self.adjust_covariance(
            self.module_dict_to_tensor(self.parameters_continuous).to(self.device) - self.starting_parameters_continuous.to(self.device)
            )
        return self.parameter_dict, True

    def get_optimum(self):
        return self.parameter_dict
