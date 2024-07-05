import numpy as np
import pandas as pd
import sys
import json
import awkward
from  G4Calo import G4System, GeometryDescriptor


class Simulation():
    def __init__(
            self,
            parameter_dict: dict
            ):
        self.n_events_per_var = parameter_dict["num_events"]["current_value"]
        self.parameter_dict = parameter_dict
        self.cw = produce_descriptor(self.parameter_dict)

    def run_simulation(self) -> pd.DataFrame:
        G4System.init(self.cw)
        G4System.applyUICommand("/control/verbose 0")
        G4System.applyUICommand("/run/verbose 0")
        G4System.applyUICommand("/event/verbose 0")
        G4System.applyUICommand("/tracking/verbose 0")
        G4System.applyUICommand("/process/verbose 0")
        G4System.applyUICommand("/run/quiet true")
        
        dfs = []
        particles = [['gamma', 0.22], ['pi+', 0.211]]

        for p in particles:
            df: pd.DataFrame = G4System.run_batch(self.n_events_per_var, p[0], 1., 20)
            df = df.assign(true_pid=np.array(len(df) * [p[1]], dtype='float32'))
            dfs.append(df)
            
        return pd.concat(dfs, axis=0, ignore_index=True)


def produce_descriptor(parameter_dict: dict):
    ''' Returns a GeometryDescriptor from the given parameters.
    Strictly takes a dict as input to ensure that the parameter names are consistent.
    Current parameters:
     - layer_thickness, alternating between absorber and scintillator

    If materials etc are added, the mapping from parameter_dict to material name has to be added here.
    '''

    cw = GeometryDescriptor()

    for name, value in parameter_dict.items():
        if name.startswith("thickness_absorber"):
            cw.addLayer(max([value["min_value"], value["current_value"]]), "G4_Pb", False, 1)
        elif name.startswith("thickness_scintillator"):
            cw.addLayer(max([value["min_value"], value["current_value"]]), "G4_PbWO4", True, 1)
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
