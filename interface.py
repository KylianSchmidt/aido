<<<<<<< HEAD
from typing import List, Any
from abc import ABC, abstractmethod
=======
import os
from typing import List, Any
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from simulation.conversion import convert_sim_to_reco
>>>>>>> main


class AIDOUserInterface(ABC):

    @abstractmethod
    def simulate(self, parameter_dict_path: str, sim_output_path: str) -> None:
        """ Starts the simulation process. We recommend starting a container and passing the arguments
        from the command line.
        
        Open the parameter dict using:
            
            parameter_dict = json.load(parameter_dict_path)

        Access its items by name and the key 'current_value':

            foo_value = parameter_dict["foo"]["current_value]

        The simulation should output exactly one file, which must be saved at 'sim_output_path'. You
        are free to choose the output format of the simulation (e.g. root file)

        Args:
            parameter_dict_path (str): The path to the parameter dictionary file.
            sim_output_path (str): The path to save the simulation output.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(
            self,
            parameter_dict_file_paths: List[str],
            simulation_file_paths: List[str],
            reco_input_path: str
            ) -> None:
        """ This method must be implemented

        This method must merge the parameter dicts and the simulation outputs into a single file.
        Its file path will be passed by the scheduler to the 'reconstruct' method as the first
        argument ('reco_input_path'). You are free to choose the file format of 'reco_input_path'.

        Args:
            parameter_dict_file_paths (List[str]): List of the simulation parameter dictionary paths
            simulation_file_paths (List[str]): List of the simulation output paths
            reco_input_path (str): Path for the merged file created by this method.
        """
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, reco_input_path: str, reco_output_path: str) -> None:
        """ This method must be implemented

        Start your reconstruction algorithm here. We recommend using a container and starting the
        reconstruction from the command line.

        Args:
            reco_input_path (str): Path of the input file for your reconstruction process. It is the same
                path as the output of the 'merge' method.
            reco_output_path (str): Path of the output file generated by your reconstruction process. Since
                this file interfaces with the AIDO Optimizer, it must have a specific format detailled in the
                following.

        Output file format (IMPORTANT):
            The output file generated by this method must be a parquet file of a pandas.DataFrame.
        """
        raise NotImplementedError

    def loss(self, y_pred: Any, y_true: Any) -> float:
        """ Important method used by the Optimizer later on

        TODO Find how to pass this to the optimizer (container problems could arise)
        """
        raise NotImplementedError
<<<<<<< HEAD
=======


class AIDOUserInterfaceExample(AIDOUserInterface):
    """ This class is an example of how to implement the 'AIDOUserInterface' class.
    """

    def simulate(self, parameter_dict_path: str, sim_output_path: str):
        os.system(
            f"singularity exec -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/simulation.py {parameter_dict_path} {sim_output_path}"
        )
        return None

    def merge(self, parameter_dict_file_paths, simulation_file_paths, reco_input_path):
        """ Combines parameter dicts and pd.DataFrames into a large pd.DataFrame which is subsequently saved
        to parquet format.
        """
        df_list: List[pd.DataFrame] = []

        for simulation_output_path in list(zip(parameter_dict_file_paths, simulation_file_paths)):
            df_list.append(
                convert_sim_to_reco(
                    *simulation_output_path,
                    input_keys=[
                        'sensor_energy', 'sensor_x', 'sensor_y', 'sensor_z',
                        'sensor_dx', 'sensor_dy', 'sensor_dz', 'sensor_layer'
                    ],
                    target_keys=["true_energy"],
                    context_keys=["true_pid"]
                )
            )

        df: pd.DataFrame = pd.concat(df_list, axis=0, ignore_index=True)
        df.to_parquet(reco_input_path, index=range(len(df)))
        return None

    def reconstruct(self, reco_input_path: str, reco_output_path: str):
        """ Start your reconstruction algorithm from a local container.

        TODO Change to the dockerhub version when deploying to production.
        """
        os.system(
            f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/reco_script.py {reco_input_path} {reco_output_path}"
        )
        os.system("rm *.pkl")
        return None

    def loss(self, y_pred: np.array, y_true: np.array) -> float:
        """ Calculate the loss for the optimizer. Easiest way is to rewrite the loss with numpy

        TODO This will not work if the loss requires non-numpy classes and functions.
        """
        return np.mean((y_pred - y_true)**2 / (np.abs(y_true) + 1.))
>>>>>>> main
