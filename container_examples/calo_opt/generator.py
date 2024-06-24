

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import time
import multiprocessing
import os
from typing import Type
from  G4Calo import G4System, GeometryDescriptor

        
class CaloDataset(Dataset):
    def __init__(
            self,
            simulation_parameters_array: np.ndarray,
            simulation_output_array: np.ndarray,
            target_array: np.ndarray,
            context_array: np.ndarray,
            means=None,
            stds=None
            ):
        
        assert (
            isinstance(simulation_output_array, np.ndarray) and isinstance(simulation_parameters_array, np.ndarray) and isinstance(target_array, np.ndarray)
        ), "Arrays are not numpy arrays!"

        self.simulation_output_array = simulation_output_array
        self.simulation_parameters_array = simulation_parameters_array
        self.target_array = target_array
        self.context_array = context_array

        if means is not None:
            self.simulation_parameters_array = (self.simulation_parameters_array - means[0]) / stds[0]
            self.simulation_output_array = (self.simulation_output_array - means[1]) / stds[1]

            # A normalised target is important for the surrogate to work given the scheduling we have here
            self.target_array = (self.target_array - means[2]) / stds[2]
            self.context_array = (self.context_array - means[3]) / stds[3]

            self.c_means = [torch.tensor(a).to('cuda') for a in means]
            self.c_stds = [torch.tensor(a).to('cuda') for a in stds]

        self.filter_infs_and_nans()

    def filter_infs_and_nans(self):
        '''
        Removes all events that contain infs or nans.
        '''
        mask = np.ones(len(self.simulation_output_array), dtype=bool)

        for i in range(len(self.simulation_output_array)):
            if np.any(np.isinf(self.simulation_output_array[i])) or np.any(np.isnan(self.simulation_output_array[i])):
                mask[i] = False
    
        self.simulation_output_array = self.simulation_output_array[mask]
        self.simulation_parameters_array = self.simulation_parameters_array[mask]
        self.target_array = self.target_array[mask]
        self.context_array = self.context_array[mask]

    def unnormalise_target(self, target):
        '''
        receives back the physically meaningful target from the normalised target
        '''
        return target * self.c_stds[2] + self.c_means[2]
    
    def normalise_target(self, target):
        '''
        normalises the target
        '''
        return (target - self.c_means[2]) / self.c_stds[2]
    
    def unnormalise_detector(self, detector):
        '''
        receives back the physically meaningful detector from the normalised detector
        '''
        return detector * self.c_stds[1] + self.c_means[1]
    
    def normalise_detector(self, detector):
        '''
        normalises the detector
        '''
        return (detector - self.c_means[1]) / self.c_stds[1]
        
    def __len__(self):
        return len(self.simulation_output_array)
    
    def __getitem__(self, idx):
        return self.simulation_parameters_array[idx], self.simulation_output_array[idx], self.target_array[idx]
    

class Generator(object):
    def __init__(
            self,
            box_size: np.ndarray,
            par_desc: GeometryDescriptor,  # nominal parameters
            n_vars=10,
            n_events_per_var=1000,
            particles=[['gamma', 0.22], ['pi+', 0.211]],
            energy_range=[1, 20]
            ):
        '''
        The box size describes the range of the parameter variations that are performed in 
        one iteration of random_generate. The box size is an array of the same length as the
        parameter list. The box size is added to the nominal parameter value to get the upper
        limit of the parameter range. The lower limit is the nominal parameter value minus the
        box size.

        workflow:
        while not converged:
            set parmeters
            generate
            train reco and surrogate
            while is_local:
                (update parameters)
                is_local?

        '''
        assert box_size.shape == par_desc.parameters.shape, "Box size and parameters have different shapes!"
        self.box_size = box_size
        # make the box actually a covariance matrix
        self.box_covariance = np.diag(box_size**2)

        self.nominal_parameters = None
        self.par_desc = par_desc

        self.n_vars = n_vars
        self.n_events_per_var = n_events_per_var
        self.energy_range = energy_range

        #from the generator hard coded either way
        self.sensor_parameters = ['sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z', 'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer']
        self.target_parameters = ['true_energy']
        self.context_parameters = ['true_pid']
        self.detector_parameters = ['detector_parameters']
        self.particles = particles

        # will be roughly determined from the first
        self.means = None
        self.stds = None

    def reset(self):
        self.nominal_parameters = None
        self.means = None
        self.stds = None

    @property
    def n_parameters(self):
        return len(self.par_desc.parameters)

    @property
    def n_sensors(self):
        return self.n_parameters // 2

    @property
    def n_parameters_per_sensor(self):
        return len(self.sensor_parameters)

    @property
    def n_target_parameters(self):
        return len(self.target_parameters)

    @property
    def n_context_parameters(self):
        return len(self.context_parameters)

    @property
    def parameters(self):
        return self.par_desc.parameters

    def is_local(self, parameters: np.ndarray, scaler=1.):
        '''
        Returns true if the parameters are within the box size of the nominal parameters.
        '''
        if self.nominal_parameters is None:
            return False
        # check if the parmeters are within one sigma of the multivariate gaussian defined by box_covariance
        # do not use box_size here but really the covariance
        diff = parameters - self.nominal_parameters
        return np.dot(diff, np.dot(np.linalg.inv(self.box_covariance), diff)) < scaler

    def random_parameters(self, parameters: np.ndarray, scaler=1.):
        '''
        Generates a random set of parameters around the nominal parameters.
        use a gaussian distribution for now, defined by box_covariance
        '''
        return np.random.multivariate_normal(parameters, self.box_covariance * scaler)        

    def random_energy(self, scaler=1.):
        '''
        Generates a random energy within the energy range, uniform distribution
        '''
        return np.random.uniform(self.energy_range[0], self.energy_range[1])

    def generate(self, parameters: np.ndarray, scaler=1.):
        '''
        generates a dataframe with different variations of the parameters within the box size
        sets central parameters to parameters
        '''
        assert len(parameters) == len(self.par_desc.parameters), "Wrong number of parameters!"
        self.nominal_parameters = parameters
        # custom made process handling because of Geant4 hang-ups
        processes = []
        queues = []
        results = []

        for i in range(self.n_vars):
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.generate_single, args=(self.random_parameters(parameters, scaler), i, queue))
            time.sleep(.1)
            p.start()
            processes.append(p)
            queues.append(queue)

        done = [False for i in range(self.n_vars)]
        done_after = [-1. for _ in range(self.n_vars)]
        start_time = time.time()
        avg_time_per_job = 1e6

        while not all(done):
            sth_happened = False
            for i, process in enumerate(processes):
                queue = queues[i]
                try:
                    # Wait for the process to put data in the queue
                    result = queue.get(timeout=.1)
                    results.append(result)
                    time.sleep(0.1)
                    process.terminate() #just kill it because of geant being weird
                    done[i] = True
                    done_after[i] = time.time() - start_time
                    sth_happened = True

                except multiprocessing.queues.Empty:
                    pass

                if any(done):
                    avg_time_per_job = np.mean(np.array(done_after)[np.array(done_after) > 0.])

                #if one job takes longer than 10 times the average, kill it
                if not done[i] and time.time() - start_time > 10. * avg_time_per_job:
                    print('killing job', i)
                    process.terminate()
                    done[i] = True
                    sth_happened = True

            if sth_happened:
                print('done', np.sum(np.array(done)), done, done_after)
            time.sleep(.1)

        #dangerous but keep it for now
        os.system("rm -f *_t0.pkl")#no problem with race here are not needed
        os.system("rm -f *_t0.root")#no problem with race here are not needed
        # concatenate the dataframes
        # only select results with valid done_after time
        results = [results[i] for i in range(len(results)) if done_after[i] > 0.]
        return pd.concat(results)

    def generate_single(self, parameters=None, index=0, output_queue=None):
        '''
        generates a dataframe with one set of parameters.
        this should probably be moved to the Parameters class, given that there are explicit mapping dependencies
        '''
        # import only in fork

        time.sleep(index/10) #this is because the random seed in geant is based on the time in ms
        if parameters is None:
            parameters = self.parameters
        cw = self.par_desc.get_descriptor(parameters, GeometryDescriptor)
        
        G4System.init(cw)

        #make it more quiet; doesn't seem to work...
        G4System.applyUICommand("/control/verbose 0")
        G4System.applyUICommand("/run/verbose 0")
        G4System.applyUICommand("/event/verbose 0")
        G4System.applyUICommand("/tracking/verbose 0")
        G4System.applyUICommand("/process/verbose 0")
        G4System.applyUICommand("/run/quiet true")
        
        dfs = []

        for p in self.particles:
            df = G4System.run_batch(self.n_events_per_var, p[0], 1., 20)
            df = df.assign(true_pid=np.array( len(df) * [p[1]], dtype='float32'))
            dfs.append(df)
        df = pd.concat(dfs)
        tiled_pars = len(df)*[np.array(parameters) ] # np.tile(pars[np.newaxis, ...], [len(df), 1])
        # add the array as columns to the dataframe
        df = df.assign(detector_parameters=tiled_pars)
        if output_queue is None:
            return df

        output_queue.put(df)
        print('done with generating #', index)
        del G4System # this is important to force a reload and make it strictly local
        return True
    


    def create_torch_dataset(self, parameters=None, random_scaler = 1., means=None, stds = None):
        # create a torch data loader from the dataframe
        # the dataframe has the following columns:
        # - true_energy [a scalar per event]
        # - sensor_energy [a vector per event]
        # - sensor_x,y,z [each a vector per event]
        # - sensor_dx,dy,dz [each a vector per event]
        # - sensor_layer [each a vector per event]
    
        # the data loader should return a tuple of input to the model and truth target
        # the input is one vector of length n_detector_parameters + n_input_parameters, and should always contain the detector parameters
        # as the first entries
        # the values for n_detector_parameters and n_input_parameters can be inferred from the dataframe
        # n_input_parameters is simply the number of sensors times the parameters per sensor (here 8)
    
        # infer the lengths
        if parameters is None:
            parameters = self.parameters #original parmeters
        
        df = self.generate(parameters, random_scaler)
        print('done with generating')
        
        # create simple array data out of dataframe (will sit in memory, can be improved at some point)
        # this is the input to the data loader
    
    
        simulation_output_array = np.array([df[par].to_numpy() for par in self.sensor_parameters], dtype='float32') #.transpose()
        #transpose axis 0 and 1 
        simulation_output_array = np.swapaxes(simulation_output_array, 0, 1)
        #flatt to nevents x -1
        simulation_output_array = np.reshape(simulation_output_array, (len(simulation_output_array), -1))

        target_array = np.array([df[par].to_numpy() for par in self.target_parameters], dtype='float32').transpose()

        #this is kinda ugly. would be better to find a smarter way to add it to the dataframe in the first place
        simulation_parameters_array = np.array([df[par].to_numpy() for par in self.detector_parameters])[0]
        simulation_parameters_array = np.concatenate(simulation_parameters_array,axis=0)
        simulation_parameters_array = np.array(simulation_parameters_array, dtype='float32')
        simulation_parameters_array = np.reshape(simulation_parameters_array, (len(target_array), -1))

        context_array = np.array([df[par].to_numpy() for par in self.context_parameters], dtype='float32').transpose()
        
        if means is None:
            self.means = [np.mean(simulation_output_array, axis=0), np.mean(simulation_parameters_array, axis=0), np.mean(target_array, axis=0), np.mean(context_array, axis=0)]
        else:
            self.means = means
        if stds is None:
            self.stds = [np.std(simulation_output_array, axis=0)+1e-3, np.std(simulation_parameters_array, axis=0)+1e-3, np.std(target_array, axis=0)+1e-3, np.std(context_array, axis=0)+1e-3]
        else:
            self.stds = stds
        # data ready, now create the torch Dataset
        
        ds = CaloDataset(simulation_output_array, simulation_parameters_array, target_array, context_array, self.means, self.stds)
        return ds
