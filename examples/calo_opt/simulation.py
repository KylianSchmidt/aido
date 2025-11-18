import json
import os
import sys

import numpy as np
from G4Calo import GeometryDescriptor, run_batch
from minipandas import MiniFrame, concat


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

    def run_simulation(self) -> MiniFrame:
        mfs = []
        particles = {"pi+": 0.211, "gamma": 0.22}

        pids = []
        for particle in particles.items():
            name, pid = particle
            mf: MiniFrame = run_batch(
                gd=self.cw,
                nEvents=int(self.n_events_per_var / len(particles)),
                particleSpec=name,
                minEnergy_GeV=1.,
                maxEnergy_GeV=20.,
                no_mp=True,
                manual_seed=self.parameter_dict["metadata"]["rng_seed"]
            )
            pids.append(np.full(len(mf), pid, dtype='float32'))
            mfs.append(mf)
        df = concat(mfs, axis=0, ignore_index=True).to_pandas(indiv_cols=False)
        df = df.assign(true_pid=np.concatenate(pids, axis=0))
        return df


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
