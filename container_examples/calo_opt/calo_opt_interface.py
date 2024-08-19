import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Iterable
from interface import AIDOUserInterface


def convert_sim_to_reco(
        parameter_dict: Dict | str,
        simulation_output_df: pd.DataFrame | str,
        input_keys: List[str],
        target_keys: List[str],
        context_keys: List[str] | None = None
        ):
    """Convert the files from the simulation to simple lists.

    Args:
        parameter_dict (dict or file path str): Instance of or file path to Parameter Dictionary.
        simulation_output_df (pd.DataFrame or file path str): Instance of or file path to pd.DataFrame
        input_keys (list of keys in df): Keys of input features to be used by the model.
        target_keys (list of keys in df): Keys of target features of the reconstruction model.
        context_keys (list of keys in df): (Optional) Keys of additional information for each 
            event.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List]: A tuple containing the 
            simulation parameter list, input features, and target features, context features and the shape
    TODO Return everything as lists or as dicts (relevant for 'create_torch_dataset' function)
    """

    def to_df(parameter_dict: Dict | str, df_length: int) -> pd.DataFrame:
        """ Create parameter dict from file if path given. Remove all parameters that are not
        optimizable and also only keep current values. Output is a df of length 'df_length', so
        that it can be concatenated with the other df's.
        """
        if isinstance(parameter_dict, str):
            with open(parameter_dict, "r") as file:
                parameter_dict: Dict = json.load(file)

        parameter_dict_only_optimizables = {}

        for parameter in parameter_dict.values():
            if parameter["optimizable"] is True:
                parameter_dict_only_optimizables[parameter["name"]] = parameter["current_value"]

        return pd.DataFrame(parameter_dict_only_optimizables, index=range(df_length))

    def expand_columns(df: pd.DataFrame) -> pd.DataFrame:
        """ Check if columns in df are lists and flatten them by replacing those
        columns with <column_name>_{i} for i in index of the list.
        """
        for column in df.columns:
            item = df[column][0]

            if isinstance(item, Iterable):
                expanded_df = pd.DataFrame(df[column].tolist(), index=df.index)
                expanded_df.columns = [f'{column}_{i}' for i in expanded_df.columns]
                df = pd.concat([df.drop(columns=column), expanded_df], axis=1)

        return df

    if isinstance(simulation_output_df, str):
        input_df: pd.DataFrame = pd.read_parquet(simulation_output_df)

    parameter_df = to_df(parameter_dict, len(input_df))
    df_combined_dict = {
        "Parameters": parameter_df,
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


class AIDOUserInterfaceExample(AIDOUserInterface):
    """ This class is an example of how to implement the 'AIDOUserInterface' class.
    """

    def simulate(self, parameter_dict_path: str, sim_output_path: str):
        os.system(
            f"singularity exec -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base.sif python3 \
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
            f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base.sif python3 \
            container_examples/calo_opt/reco_script.py {reco_input_path} {reco_output_path}"
        )
        os.system("rm *.pkl")
        return None

    def loss(self, y_pred: np.array, y_true: np.array) -> float:
        """ Calculate the loss for the optimizer. Easiest way is to rewrite the loss with numpy

        TODO This will not work if the loss requires non-numpy classes and functions.
        """
        return np.mean((y_pred - y_true)**2 / (np.abs(y_true) + 1.))
