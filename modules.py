import os
from typing import List, Dict
import numpy as np
import pandas as pd
from simulation.conversion import convert_sim_to_reco


class Reconstruction:

    def merge(
            self,
            parameter_dict_file_paths: List[str],
            simulation_file_paths: List[str],
            reco_input_path: str
            ) -> None:
        """ This method must be implemented
        """
        raise NotImplementedError

    def run(self, reco_file_paths_dict: Dict[str, str]) -> None:
        """ This method must be implemented

        Start your reconstruction algorithm here. We recommend using a container and starting the
        reconstruction from the command line.
        """
        raise NotImplementedError
    
    def loss(self, y_pred, y_true) -> float:
        """ Important method used by the Optimizer later on

        TODO Find how to pass this to the optimizer (container problems could arise)
        """
        raise NotImplementedError


class ReconstructionExample(Reconstruction):

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

    def run(self, reco_file_paths_dict: Dict[str, str]):
        """ Start your reconstruction algorithm from a local container.

        TODO Change to the dockerhub version when deploying to production.
        """
        os.system(
            f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/reco_script.py {reco_file_paths_dict["own_path"]}"
        )

    def loss(self, y_pred: np.array, y_true: np.array) -> float:
        """ Calculate the loss for the optimizer. Easiest way is to rewrite the loss with numpy

        TODO This will not work if the loss requires non-numpy classes and functions.
        """
        np.mean((y_pred - y_true)**2 / (np.abs(y_true) + 1.))
