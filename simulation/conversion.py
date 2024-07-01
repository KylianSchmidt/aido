import pandas as pd
import json
from typing import Dict, List, Iterable


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

    def convert_parameters_to_df(parameter_dict: Dict | str, df_length: int) -> pd.DataFrame:
        """ Create parameter dict from file if path given. Remove all parameters that are not
        optimizable and also only keep current values. Output is a df of length 'length', so
        that it can be concatenated with the other df's.
        """
        if isinstance(parameter_dict, str):
            with open(parameter_dict, "r") as file:
                parameter_dict: Dict = json.load(file)

        parameter_dict_only_optimizables = {}

        for parameter in parameter_dict.values():
            if parameter["optimizable"] is True:
                parameter_dict_only_optimizables[parameter["name"]] = parameter["current_value"]

        print(
            "Parameter Dict (optimizables, only current values):\n",
            json.dumps(parameter_dict_only_optimizables, indent=4)
        )
        return pd.DataFrame(parameter_dict_only_optimizables, index=range(df_length))
    
    def expand_columns(df: pd.DataFrame) -> pd.DataFrame:
        """ Check if columns in df are lists and flatten them by replacing those
        columns with <column_name>_{i} for i in index of the list.
        """
        for column in df.columns:
            item = df[column][0]

            if isinstance(item, Iterable):
                expanded_df = pd.DataFrame(
                    df[column].tolist(),
                    index=df.index
                )

                expanded_df.columns = [f'{column}_{i}' for i in expanded_df.columns]
                df = pd.concat([df.drop(columns=column), expanded_df], axis=1)

        return df

    if isinstance(simulation_output_df, str):
        input_df: pd.DataFrame = pd.read_parquet(simulation_output_df)

    parameter_df = convert_parameters_to_df(parameter_dict, len(input_df))
    df_combined_dict = {
        "Parameters": parameter_df,
        "Inputs": input_df[input_keys],
        "Targets": input_df[target_keys],
        "Context": input_df[context_keys]
    }
    df: pd.DataFrame = pd.concat(
        df_combined_dict.values(),
        keys=df_combined_dict.keys(),
        axis=1
    )
    df = expand_columns(df)
    return df
