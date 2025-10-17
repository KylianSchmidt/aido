import sys
from pathlib import Path

import pytest

import aido
from examples.calo_opt.interface import CaloOptInterface


@pytest.fixture
def example_simple_parameters() -> aido.SimulationParameterDictionary:
    parameters = aido.SimulationParameterDictionary([
        aido.SimulationParameter("thickness_absorber_0", 1.0, min_value=0.0),
        aido.SimulationParameter("thickness_scintillator_0", 1.0, min_value=0.0),
        aido.SimulationParameter("material_absorber_0", "G4_Fe", optimizable=False),
        aido.SimulationParameter("material_scintillator_0", "G4_PbWO4", optimizable=False),
        aido.SimulationParameter("thickness_absorber_1", 1.0, min_value=0.0),
        aido.SimulationParameter("thickness_scintillator_1", 1.0, min_value=0.0),
        aido.SimulationParameter("material_absorber_1", "G4_Fe", optimizable=False),
        aido.SimulationParameter("material_scintillator_1", "G4_PbWO4", optimizable=False),
        aido.SimulationParameter("thickness_absorber_2", 1.0, min_value=0.0),
        aido.SimulationParameter("thickness_scintillator_2", 1.0, min_value=0.0),
        aido.SimulationParameter("material_absorber_2", "G4_Fe", optimizable=False),
        aido.SimulationParameter("material_scintillator_2", "G4_PbWO4", optimizable=False),
    ])
    return parameters


@pytest.fixture
def example_interface() -> CaloOptInterface:
    interface = CaloOptInterface()
    interface.verbose = True
    interface.container_path = "/ceph/kschmidt/singularity_cache/minicalosim_latest.sif"
    interface.container_extra_flags = "-B /work,/ceph"
    return interface


@pytest.mark.slow
def test_example_single_iteration(
    example_simple_parameters: aido.SimulationParameterDictionary,
    example_interface: CaloOptInterface,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "argv", ["aido_optimize"])

    aido.optimize(
        parameters=example_simple_parameters,
        user_interface=example_interface,
        simulation_tasks=2,
        max_iterations=1,
        threads=2,
        results_dir=tmp_path,
        description="Test example with single iteration"
    )
