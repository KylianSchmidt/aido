import os
from typing import Dict, Iterable, List

import pandas as pd
import torch
from calo_opt.reconstruction.model import Reconstruction

import aido


class CaloOptInterface(aido.UserInterfaceBase):
    """ This class is an example of how to implement the 'AIDOUserInterface' class.

    We use the following container:

        https://hub.docker.com/r/jkiesele/minicalosim

    Download the container and add the file path to the variable 'container_path'
    """

    htc_global_settings = {}
    container_path: str = ...  # place the path for the example container here
    container_extra_flags: str = ""  # place extra flags for singularity here

    def simulate(self, parameter_dict_path: str, sim_output_path: str):
        os.system(
            f"singularity exec {self.container_extra_flags} {self.container_path} python3 \
            examples/calo_opt/simulation.py {parameter_dict_path} {sim_output_path} > /dev/null 2>&1"
        )
        return None

    def convert_sim_to_reco(
            parameter_dict_path: Dict | str,
            simulation_output_df: pd.DataFrame | str,
            input_keys: List[str],
            target_keys: List[str],
            context_keys: List[str] | None = None
            ):
        """
        This is a helper function specific to the CaloOpt example. Converts the files from the simulation
        to a pandas dataframe.

        Args:
            parameter_dict (dict or file path str): Instance of or file path to Parameter Dictionary.
            simulation_output_df (pd.DataFrame or file path str): Instance of or file path to pd.DataFrame
            input_keys (list of keys in df): Keys of input features to be used by the model.
            target_keys (list of keys in df): Keys of target features of the reconstruction model.
            context_keys (list of keys in df): (Optional) Keys of additional information for each
                event.

        Returns:
            pd.DataFrame:A DataFrame containing the simulation parameter list, input features, and
            target features, context features.
        """

        def expand_columns(df: pd.DataFrame) -> pd.DataFrame:
            """ Check if columns in df are lists and flatten them by replacing those
            columns with <column_name>_{i} for i in index of the list.
            """
            for column in df.columns:
                item = df[column][0]

                if isinstance(item, Iterable):
                    column_list = df[column].tolist()
                    expanded_df = pd.DataFrame(column_list, index=df.index)
                    expanded_df.columns = [f'{column}_{i}' for i in expanded_df.columns]
                    df = pd.concat([df.drop(columns=column), expanded_df], axis=1)

            return df

        if isinstance(simulation_output_df, str):
            input_df: pd.DataFrame = pd.read_parquet(simulation_output_df)

        parameter_dict = aido.SimulationParameterDictionary.from_json(parameter_dict_path)

        df_combined_dict = {
            "Parameters": parameter_dict.to_df(len(input_df), display_discrete="as_one_hot"),
            "Inputs": expand_columns(input_df[input_keys]),
            "Targets": expand_columns(input_df[target_keys]),
            "Context": expand_columns(input_df[context_keys])
        }
        df: pd.DataFrame = pd.concat(
            df_combined_dict.values(),
            keys=df_combined_dict.keys(),
            axis=1
        )
        return df

    def merge(
            self,
            parameter_dict_file_paths: List[str],
            simulation_file_paths: List[str],
            reco_input_path: str
            ):
        """ Combines parameter dicts and pd.DataFrames into a large pd.DataFrame which is subsequently saved
        to parquet format.
        """
        df_list: List[pd.DataFrame] = []

        for simulation_output_path in list(zip(parameter_dict_file_paths, simulation_file_paths)):
            df_list.append(
                type(self).convert_sim_to_reco(
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
        df = df.fillna(0)
        df = df.reset_index(drop=True)
        df.to_parquet(reco_input_path, index=range(len(df)))
        return None

    def reconstruct(self, reco_input_path: str, reco_output_path: str, is_validation: bool):
        """ Start your reconstruction algorithm from a local container.
        """
        os.system(
            f"singularity exec --nv {self.container_extra_flags} {self.container_path} \
            python3 examples/calo_opt/train.py \
            {reco_input_path} {reco_output_path} {is_validation} {self.results_dir}"
        )
        os.system("rm -f *.pkl")
        return None

    def loss(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return Reconstruction.loss(y, y_pred)
