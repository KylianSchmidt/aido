import torch
import numpy as np
from torch.utils.data import DataLoader
from surrogate import Surrogate, SurrogateDataset
from reconstruction import Reconstruction
from typing import Dict


class Optimizer(object):
    '''
    The optimizer uses the surrogate model to optimise the detector parameters in batches.
    It is also linked to a generator object, to check if the parameters are still in bounds using the function is_local(parameters)
    of the generator.

    Once the parameters are not local anymore, the optimizer will return the last parameters that were local and stop.
    For this purpose, the surrogate model will need to be applied using fixed weights.
    Then the reconstruction model loss will be applied based on the surrogate model output.
    The gradient w.r.t. the detector parameters will be calculated and the parameters will be updated.
    '''
    def __init__(
            self,
            surrogate_model: Surrogate,
            reconstruction_model: Reconstruction,
            parameter_dict: Dict,
            lr=0.001,
            batch_size=128,
            constraints: Dict = None
            ):
        
        self.surrogate_model = surrogate_model
        self.reconstruction_model = reconstruction_model
        self.n_time_steps = surrogate_model.n_time_steps
        self.lr = lr
        self.batch_size = batch_size
        self.constraints = constraints
        self.device = torch.device('cuda')
        self.cu_box = torch.tensor(self.generator.box_size, dtype=torch.float32).to(self.device)  # TODO
        self.detector_parameters = torch.nn.Parameter(torch.tensor(, dtype='float32')).to(self.device), requires_grad=True)  # TODO
        self.optimizer = torch.optim.Adam([self.detector_parameters], lr=self.lr)
        self.to(self.device)

    def to(self, device: str):
        self.device = device
        self.surrogate_model.to(device)
        self.cu_box.to(device)

    def other_constraints(self, dataset: SurrogateDataset):
        """ Constrain length of detector to 25cm.
        Now keep parameters such that within the box size of the generator, there are always some positive values even if the 
        central parameters are negative. Both box size and raw_detector_parameters are in non-normalised space, so this is straight forward
        the generator will have to provide the box size
        this will avoid mode collapse
        """
        raw_detector_parameters = self.detector_parameters
        detector_length = torch.sum(raw_detector_parameters)

        if self.constraints is not None:

            if 'length' in self.constraints:
                detector_length = torch.sum(raw_detector_parameters)
                total_length_loss = torch.mean(100.*torch.nn.ReLU()(detector_length - self.constraints['length'])**2)

        box = self.cu_box # the box has same ordering as raw_detector_parameters
        lower_para_bound = -box / 1.1
        bloss = torch.mean(100.*torch.nn.ReLU()(lower_para_bound - raw_detector_parameters)**2)
        return total_length_loss + bloss

    def adjust_generator_covariance(self, direction, min_scale=2.0):
        #stretches the box_covariance of the generator in the directon specified as input
        #direction is a vector in parameter space
        v = direction
        v_length = np.linalg.norm(v)
        v_norm = v / v_length

        s = min_scale *np.max([1., 4.*v_length])  # scale factor at least by a factor of two, if not more

        # Create the scaling adjustment matrix
        M_scaled = (s - 1) * np.outer(v_norm, v_norm)        

        # Adjust the original covariance matrix
        self.generator.box_covariance = np.diag(self.generator.box_size**2) + M_scaled  # TODO
        print('new box_covariance', self.generator.box_covariance)


    def optimize(self, dataset: SurrogateDataset, batch_size, n_epochs, lr, add_constraints = False):
        '''
        keep both models fixed, train only the detector parameters (self.detector_start_parameters)
        using the reconstruction model loss
        '''
        # set the optimizer
        self.optimizer.lr = lr

        self.surrogate_model.eval()
        self.reconstruction_model.eval()

        # create a dataloader for the dataset, this is the surrogate dataloader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # save the initial parameters
        initial_parameters = self.detector_parameters.detach().cpu().numpy()

        # loop over the batches
        mean_loss = 0
        for epoch in range(n_epochs):
            mean_loss = 0 #only use last epoch
            stop_epoch = False
            for batch_idx, (_, true_inputs, true_context, reco_result) in enumerate(data_loader):

                # in principle this could also be sampled from the correct distributions; but the distributions are not known in all cases (mostly for context)
                # keep in mind for an extension
                true_inputs = true_inputs.to(self.device)
                true_context = true_context.to(self.device)
                reco_result = reco_result.to(self.device)
                # apply the model
                n_detector_paras  = dataset.normalise_detector(self.detector_parameters)
                reco_surrogate = self.surrogate_model.sample_forward(n_detector_paras, 
                                                                     true_inputs, 
                                                                     true_context)
                # calculate the loss
                loss = self.reconstruction_model.loss(dataset.unnormalise_target(reco_surrogate), dataset.unnormalise_target(true_inputs))
                if add_constraints:
                    loss += self.other_constraints(dataset)
                #
                self.optimizer.zero_grad()
                loss.backward()
                #print('gradient',self.detector_parameters.grad, 'should have', self.detector_parameters.requires_grad)

                #if loss is nan, stop
                if np.isnan(loss.item()):
                    print("NaN loss, exiting.")
                    # save parameters, reset the optimiser as if it made a step but without updating the parameters
                    prev_parameters = self.detector_parameters.detach().cpu().numpy()
                    self.optimizer.step()
                    self.detector_parameters.data = torch.tensor(prev_parameters).to(self.device)
                    # return 
                    return self.detector_parameters.detach().cpu().numpy(), False, mean_loss / (batch_idx+1)
                
                self.optimizer.step()
                mean_loss += loss.item()

                #record steps

                #check if the parameters are still local otherwise stop
                if not self.generator.is_local(self.detector_parameters.detach().cpu().numpy(),0.8):#a bit smaller box size to be safe
                    stop_epoch = True
                    break
                
                if batch_idx % 20 == 0:
                    nppars = self.detector_parameters.detach().cpu().numpy()
                    pdct = self.generator.translate_parameters(nppars)
                    print('current parameters: ')
                    for k in pdct.keys():
                        print(k, pdct[k])

            print('Optimizer Epoch: {} \tLoss: {:.8f}'.format(
                        epoch, loss.item()))
            if stop_epoch:
                break
            
        self.clamp_parameters()
        mean_loss /= batch_idx+1
        self.adjust_generator_covariance( self.detector_parameters.detach().cpu().numpy() - initial_parameters )
        return self.detector_parameters.detach().cpu().numpy(), True, mean_loss

    def get_optimum(self):
        return self.detector_parameters.detach().cpu().numpy()









    

