import numpy as np
import pandas as pd
import json


def convert_sim_to_reco(
        parameter_dict: Dict | str,
        simulation_output_df: pd.DataFrame | str,
        input_features_keys: List[str],
        target_features_keys: List[str],
        context_information_keys: List[str] | None = None
        ):
    """Convert the files from the simulation to simple lists.

    Args:
        parameter_dict (dict or file path str): Instance of or file path to Parameter Dictionary.
        simulation_output_df (pd.DataFrame or file path str): Instance of or file path to pd.DataFrame
        input_features_keys (list of keys in df): Keys of input features to be used by the model.
        target_features_keys (list of keys in df): Keys of target features of the reconstruction model.
        context_information (list of keys in df): (Optional) Keys of additional information for each 
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

    simulation_parameter_list = []

    for parameter in parameter_dict.values():
        if parameter["optimizable"] is True:
            simulation_parameter_list.append(parameter["current_value"])

    simulation_parameters = np.array(simulation_parameter_list, dtype='float32')

    # 2. Simulation output (pd.DataFrame -> linear array)
    if isinstance(simulation_output_df, str):
        simulation_output_df: pd.DataFrame = pd.read_pickle(simulation_output_df)

    input_features = np.array([simulation_output_df[par].to_numpy() for par in input_features_keys], dtype='float32')
    input_features = np.swapaxes(input_features, 0, 1)
    input_features = np.reshape(input_features, (len(input_features), -1))

    # 3. Reconstruction targets
    target_features = simulation_output_df[target_features_keys].to_numpy(dtype='float32')

    # 4. Context information from simulation
    context_information = simulation_output_df[context_information_keys].to_numpy(dtype='float32')

    # Reshape parameters to (N, num_parameters)
    simulation_parameters = np.repeat([simulation_parameters], len(target_features), axis=0)

    shape = (simulation_parameters.shape[1], input_features.shape[1], target_features.shape[1], context_information.shape[1])
    return simulation_parameters, input_features, target_features, context_information, shape
