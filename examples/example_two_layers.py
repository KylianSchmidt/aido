import os

import matplotlib.pyplot as plt
from calo_opt.interface_simple import AIDOUserInterfaceExample  # Import your derived class
from calo_opt.plotting import CaloOptPlotting

import aido


class UIFullCalorimeter(AIDOUserInterfaceExample):
    
    def plot(self, parameter_dict: aido.SimulationParameterDictionary) -> None:
        calo_opt_plotter = CaloOptPlotting(self.results_dir)
        calo_opt_plotter.plot_energy_resolution_all()
        calo_opt_plotter.plot_energy_resolution_first_and_last()
        calo_opt_plotter.plot_energy_resolution_evolution()
        calo_opt_plotter.plot_calorimeter_sideview()
        plt.close("all")
        return None


if __name__ == "__main__":

    aido.set_config("simulation.sigma", 1.5)
    min_value = 0.001
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 1.0, min_value=min_value),
        aido.SimulationParameter("thickness_scintillator_0", 1.0, min_value=min_value),
        aido.SimulationParameter("material_absorber_0", -1, optimizable=False),
        aido.SimulationParameter("material_scintillator_0", 1, optimizable=False),
        aido.SimulationParameter("num_events", 400, optimizable=False),
        aido.SimulationParameter("two_layers", True, optimizable=False)
    ])

    aido.optimize(
        parameters=parameters,
        user_interface=UIFullCalorimeter,
        simulation_tasks=20,
        max_iterations=200,
        threads=20,
        results_dir="/work/kschmidt/aido/results_two_layers/results_20241213",
        description="""
            Full Calorimeter with cost and length constraints.
            Improved normalization of reconstructed array in Surrogate Model
            Using boosted parameter dict output by optimizer
            Reduced sigma
            One-Hot parameters in Reco and Surrogate
            Made reco results 1d (temporary!)
            Normalized reco loss in surrogate
            Separetely decrease the learning of discrete parameters
            Set discrete learning rate a bit higher (1e-4)
            With correct gradients for the constraints
            Replaced empty events with bad reco loss (fixed)
            Penalties for empty sensors
            Longer Surrogate training
            Add true energy to context and removed penalties from loss
            Add deposited energy to Context
            Increased sigma
            Add validation Tasks
            Changed Optimizer to compute reco loss itself
            Improvements to the Surrogate model training
            Actually implemented covariance box correctly
            Save reco model between iterations
            Discrete LR = 0.001, gradients clamped at 0.01
            Fixed material choice
            Removed constraints
            Only gamma
            Remove empty events
        """
    )
    os.system("rm *.root")
