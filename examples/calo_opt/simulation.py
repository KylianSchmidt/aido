import json
import os
import sys

import numpy as np
import pandas as pd
from G4Calo import GeometryDescriptor, run_batch


class Simulation():
    def __init__(
            self,
            parameter_dict: dict
            ):
        self.parameter_dict = parameter_dict

        if "num_events" in parameter_dict:
            self.n_events_per_var = parameter_dict["num_events"]["current_value"]
        else:
            self.n_events_per_var = 100

        self.cw = GeometryDescriptor()

        for i in range(3):
            self.cw.addLayer(
                max(parameter_dict[f"thickness_absorber_{i}"]["current_value"], 1e-3),
                parameter_dict[f"material_absorber_{i}"]["current_value"],
                False,
                1
            )
            self.cw.addLayer(
                max(parameter_dict[f"thickness_scintillator_{i}"]["current_value"], 1e-3),
                parameter_dict[f"material_scintillator_{i}"]["current_value"],
                True,
                1
            )

    def run_simulation(self) -> pd.DataFrame:
        dfs = []
        particles = {"pi+": 0.211, "gamma": 0.22}

        for particle in particles.items():
            name, pid = particle
            df: pd.DataFrame = run_batch(
                gd=self.cw,
                nEvents=int(self.n_events_per_var / len(particles)),
                particleSpec=name,
                minEnergy_GeV=1.,
                maxEnergy_GeV=20.,
                no_mp=True,
                manual_seed=self.parameter_dict["metadata"]["rng_seed"]
            )
            df = df.assign(true_pid=np.full(len(df), pid, dtype='float32'))
            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    @classmethod
    def produce_descriptor(cls, parameter_dict: dict):
        ''' Returns a GeometryDescriptor from the given parameters.
        Strictly takes a dict as input to ensure that the parameter names are consistent.
        Current parameters:
        - layer_thickness, alternating between absorber and scintillator

        If materials etc are added, the mapping from parameter_dict to material name has to be added here.
        '''
        cw = GeometryDescriptor()

        for name, value in parameter_dict.items():
            if name.startswith("thickness_absorber"):
                cw.addLayer(value["current_value"], "G4_Pb", False, 1)
            elif name.startswith("thickness_scintillator"):
                cw.addLayer(value["current_value"], "G4_PbWO4", True, 1)
        return cw


if __name__ == "__main__":
    parameter_dict_file_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(parameter_dict_file_path, "r") as file:
        parameter_dict = json.load(file)

    generator = Simulation(parameter_dict)
    df = generator.run_simulation()

    for column in df.columns:
        if df[column].dtype == "awkward":
            df[column] = df[column].to_list()

    df.to_parquet(output_path)
    os.system("rm -f ./*.pkl")
