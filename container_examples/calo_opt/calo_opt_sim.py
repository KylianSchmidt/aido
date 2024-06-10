import numpy as np
import pandas as pd
import sys
import json
from generator import produce_descriptor
from  G4Calo import G4System, GeometryDescriptor


class Simulation():
    def __init__(
            self,
            parameter_dict: dict
            ):
        self.n_events_per_var = 100
        self.parameter_dict = parameter_dict
        self.cw = produce_descriptor(self.parameter_dict)

    def run_simulation(self):
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
            df = G4System.run_batch(self.n_events_per_var, p[0], 1., 20)
            df = df.assign(true_pid=np.array(len(df) * [p[1]], dtype='float32'))
            dfs.append(df)
        return pd.concat(dfs)


parameter_dict_file_path = sys.argv[1]
output_path = sys.argv[2]

with open(parameter_dict_file_path, "r") as file:
    parameter_dict = json.load(file)
print(" PARAMETER DICT", parameter_dict)

generator = Simulation(parameter_dict)
df = generator.run_simulation()
print(df)
result_array = np.linspace(0, 1, 10)
np.save(output_path, result_array)
