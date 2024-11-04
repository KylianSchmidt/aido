import os
from typing import Dict, Iterable, List

import pandas as pd

import aido


class AIDOUserInterfaceExample(aido.AIDOBaseUserInterface):
    """ This class is an example of how to implement the 'AIDOUserInterface' class.
    """

    htc_global_settings = {}

    def simulate(self, parameter_dict_path: str, sim_output_path: str):
        os.system(
            f"singularity exec -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base.sif python3 \
            examples/calo_opt/simulation.py {parameter_dict_path} {sim_output_path}"
        )
        return None
    
    def convert_sim_to_reco(
            parameter_dict_path: Dict | str,
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
            "Parameters": parameter_dict.to_df(len(input_df), one_hot=True),
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

    def merge(self, parameter_dict_file_paths, simulation_file_paths, reco_input_path):
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
        df.to_parquet(reco_input_path, index=range(len(df)))
        return None

    def reconstruct(self, reco_input_path: str, reco_output_path: str):
        """ Start your reconstruction algorithm from a local container.

        TODO Change to the dockerhub version when deploying to production.
        """
        os.system(
            f"singularity exec --nv -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base.sif python3 \
            examples/calo_opt/reco_script.py {reco_input_path} {reco_output_path}"
        )
        os.system("rm *.pkl")
        return None
    
    def constraints(self, parameter_dict: aido.SimulationParameterDictionary) -> float:
        return 0.0


if __name__ == "__main__":
    sim_param_cheap = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber", 1.0, units="cm", max_value=50.0, min_value=0.1, sigma=0.5),
        aido.SimulationParameter("thickness_scintillator", 0.5, units="cm", max_value=10.0, min_value=0.01, sigma=0.5),
        aido.SimulationParameter("absorber_material", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[1.3, 0.092]),
        aido.SimulationParameter(
            "scintillator_material", "G4_PbWO4", discrete_values=["G4_PbWO4", "G4_Fe"], cost=[1.5, 1.0]
        ),
        aido.SimulationParameter("num_blocks", 10, optimizable=False),
        aido.SimulationParameter("num_events", 400, optimizable=False)
    ])
    sim_param_expensive = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber", 1.0, units="cm", max_value=50.0, min_value=0.1, sigma=0.5),
        aido.SimulationParameter("thickness_scintillator", 0.5, units="cm", max_value=10.0, min_value=0.01, sigma=0.5),
        aido.SimulationParameter("absorber_material", "G4_Pb", discrete_values=["G4_Pb", "G4_Fe"], cost=[1.3, 0.092]),
        aido.SimulationParameter(
            "scintillator_material", "G4_PbWO4", discrete_values=["G4_PbWO4", "G4_Fe"], cost=[1.5, 1.0]
        ),
        aido.SimulationParameter("num_blocks", 10, optimizable=False),
        aido.SimulationParameter("num_events", 400, optimizable=False)
    ])
    print(f"{AIDOUserInterfaceExample().constraints(sim_param_cheap)=}")
