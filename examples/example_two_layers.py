import os

from calo_opt.plotting import CaloOptPlotting

import aido
from calo_opt.interface import CaloOptInterface  # Import your derived class


class UIFullCalorimeter(CaloOptInterface):

    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:
        CaloOptPlotting(self.results_dir).plot()
        return None


if __name__ == "__main__":

    aido.set_config("simulation.sigma", 1.5)
    min_value = 0.0
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 1.0, min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_0", 1.0, min_value=min_value),
        aido.SimulationParameter("material_absorber_0", "G4_Fe", optimizable=False),
        aido.SimulationParameter("material_scintillator_0", "G4_PbWO4", optimizable=False),
        aido.SimulationParameter("num_events", 400, optimizable=False),
    ])

    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=20,
        max_iterations=50,
        threads=20,
        results_dir="./results/two_layers/",
        description="Example with two layers"
    )
    os.system("rm *.root")
