import os
from typing import List, Dict
import pandas as pd
from simulation.conversion import convert_sim_to_reco


class Reconstruction:

    def merge(self, parameter_dict_file_paths: List[str], simulation_file_paths: List[str], reco_input_path: str):
        """ This method must be implemented
        """
        raise NotImplementedError

    def run(self, reco_file_paths_dict: Dict[str, str]):
        """ This method must be implemented

        Start your reconstruction algorithm here. We recommend using a container and starting the
        reconstruction from the command line, as in the following example.
        """
        raise NotImplementedError


class ReconstructionExample(Reconstruction):

    def merge(self, parameter_dict_file_paths, simulation_file_paths, reco_input_path):
        """ This method must be implemented
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

    def run(self, reco_file_paths_dict: Dict):
        """ Start your reconstruction algorithm here. We recommend using a container and starting the
        reconstruction from the command line, as in the following example.
        """
        os.system(
            f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
            container_examples/calo_opt/training_script.py {reco_file_paths_dict["own_path"]}"
        )
