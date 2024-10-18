import json
import os
import sys

import numpy as np
import pandas as pd
from G4Calo import G4System, GeometryDescriptor


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

        if "absorber_material" in parameter_dict:
            absorber_material = parameter_dict["absorber_material"]["current_value"]
        else:
            absorber_material = "G4_Pb"

        if "scintillator_material" in parameter_dict:
            scintillator_material = parameter_dict["scintillator_material"]["current_value"]
        else:
            scintillator_material = "G4_PbWO4"

        if "num_blocks" in parameter_dict:  # Case for discrete number of absorber/scintillator blocks
            self.cw = GeometryDescriptor()

            for _ in range(parameter_dict["num_blocks"]["current_value"]):
                self.cw.addLayer(
                    parameter_dict["thickness_absorber"]["current_value"], absorber_material, False, 1
                )
                self.cw.addLayer(
                    parameter_dict["thickness_scintillator"]["current_value"], scintillator_material, True, 1
                )
        elif "simple_setup" in parameter_dict:
            self.cw = self.produce_descriptor(parameter_dict)
        elif "full_calorimeter" in parameter_dict:
            self.cw = GeometryDescriptor()

            for i in range(3):
                self.cw.addLayer(
                    parameter_dict[f"thickness_absorber_{i}"]["current_value"],
                    parameter_dict[f"material_absorber_{i}"]["current_value"],
                    False,
                    1
                )
                self.cw.addLayer(
                    parameter_dict[f"thickness_scintillator_{i}"]["current_value"],
                    parameter_dict[f"material_scintillator_{i}"]["current_value"],
                    True,
                    1
                )

    def run_simulation(self) -> pd.DataFrame:
        G4System.init(self.cw)
        G4System.applyUICommand("/control/verbose 0")
        G4System.applyUICommand("/run/verbose 0")
        G4System.applyUICommand("/event/verbose 0")
        G4System.applyUICommand("/tracking/verbose 0")
        G4System.applyUICommand("/process/verbose 0")
        G4System.applyUICommand("/run/quiet true")

        dfs = []
        particles = {'pi+': 0.211, 'gamma': 0.22}

        for particle in particles.items():
            name, pid = particle
            df: pd.DataFrame = G4System.run_batch(int(self.n_events_per_var / len(particles)), name, 1., 20.)
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
os.system("rm ./*.pkl")
