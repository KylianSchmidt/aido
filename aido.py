import os
from SimulationWrapper import AIDO
from simulation.SimulationHelpers import SimulationParameter, SimulationParameterDictionary
from modules import ReconstructionExample


def simulation(output_parameter_dict_path: str, output_path: str):
    os.system(
        f"singularity exec -B /work,/ceph /ceph/kschmidt/singularity_cache/ml_base python3 \
        container_examples/calo_opt/simulation.py {output_parameter_dict_path} {output_path}"
    )
    return None


if __name__ == "__main__":
    os.system("rm ./results -rf")

    sim_param_dict = SimulationParameterDictionary([
        SimulationParameter('thickness_absorber_0', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
        SimulationParameter('thickness_absorber_1', 1.0, min_value=1E-3, max_value=5.0, sigma=0.2),
        SimulationParameter('thickness_scintillator_0', 0.5, min_value=1E-3, max_value=1.0, sigma=0.2),
        SimulationParameter('thickness_scintillator_1', 0.1, min_value=1E-3, max_value=1.0, sigma=0.2),
        SimulationParameter("num_events", 100, optimizable=False)
    ])

    AIDO(
        sim_param_dict,
        simulation_callable=simulation,
        reconstruction_callable=ReconstructionExample(),
        simulation_tasks=4,
        threads=6
    )
