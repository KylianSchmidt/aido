import numpy as np
import pandas as pd
import json
from typing import Dict, List


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
    # 1. Simulation Parameter list
    if isinstance(parameter_dict, str):
        with open(parameter_dict, "r") as file:
            parameter_dict: Dict = json.load(file)

    # Remove parameters if they are not optimizable TODO write to function
    parameter_dict_only_optimizables = []

    for parameter in parameter_dict.values():
        if parameter["optimizable"] is True:
            parameter_dict_only_optimizables.append(parameter["current_value"])

    parameter_dict["Parameters"] = parameter_dict_only_optimizables
    print("DEBUG", parameter_dict)
    parameter_df = pd.DataFrame(parameter_dict)

    # 2. Simulation output (pd.DataFrame -> linear array)
    if isinstance(simulation_output_df, str):
        input_df: pd.DataFrame = pd.read_parquet(simulation_output_df)

    input_features_df = input_df[input_keys]

    # 3. Reconstruction targets
    target_features = input_df[target_keys].to_numpy(dtype='float32')

    # 4. Context information from simulation
    context_information = input_df[context_keys].to_numpy(dtype='float32')

    # Concat df
    df: pd.DataFrame = pd.concat(
        [parameter_df, input_features_df, target_features, context_information],
        axis=1
    )
    
    for column in df.columns:
        item = df[column][0]

        if isinstance(item, list):
            expanded_df = pd.DataFrame(
                df[column].tolist(),
                index=df.index
            )

        expanded_df.columns = [f'{column}_{i}' for i in expanded_df.columns]
        df = pd.concat(df.drop(columns=column), expanded_df, axis=1)

    return df
