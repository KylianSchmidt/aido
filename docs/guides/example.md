# Provided examples

## Calo-opt

This example is based on the existing `calo_opt` project. In order to run these examples, you need to pull the following container:

 - With `docker`: 

    ```bash
    docker pull jkiesele/minicalosim/7829fde_2
    ``` 
 - With `singularity`/ `apptainer` (useful if you lack root access to the machine, do not have docker installed or do not have permissions):

    ```bash
    apptainer pull docker://jkiesele/minicalosim:7829fde_2
    ```

Register the container by adding the path to it to the `CaloOptInterface` class:

```python
interface = CaloOptInterface()
interface.container_path = "</path/to/container.sif>"
interface.container_extra_flags = ""  # Extra apptainer settings
interface.verbose = True  # This flag enables / disables Geant4 logs
```

In essence, the `CaloOptInterface` creates a subprocess in which Geant4 runs:

```python
def simulate(self, parameter_dict_path: str, sim_output_path: str):
    os.system(
        f"singularity exec {self.container_extra_flags} {self.container_path} python3 \
        examples/calo_opt/simulation.py {parameter_dict_path} {sim_output_path} {self.suppress_output}"
    )
```

Finally, run the example by calling:

```python
python3 examples/<example.py>
```

Or add the root directory for AIDO to your `$PYTHONPATH` and call the examples from the `examples` folder directly.

If you get Errors of the type `Unfulfilled dependencies at Runtime`, check if the Simulation was able to run. The first debugging tool is to set `interface.verbose=True` and check the error logs. For example using a wrong path or not mounting a required filesystem into the container can all lead to a failed Task. The scheduler will then simply inform us that the next Task could not run. 

## Sampling calorimeter

In this example we simulate a sampling calorimeter composed of three layers of absorber material and three of active recording material. More detail can be found in https://arxiv.org/abs/2502.02152 Section 6.

### Simulation

The current examples has some hardcoded parts that set up the geometry of the detector.

1. **Configurable parameters**:
    - Thickness of the layers
    - Material of the layers
    - Cost per unit of length
    - Probability of sampling one material over another, e.g. the confidence of the model in that material choice.
w
2. **Currently hardcoded settings** in `simulation.py` but that can be changed in the code of the example:
    - Number of layers: expects exactly six layers, three of them absorber, three scintillators.
    - Names of the layers: either `thickness_absorber_{i}`or `thickness_scintillator_{i}` for $i \in {0, 1, 2}$.
    - Lateral granularity: only one cell per layer perpendicularly to the beam axis
    - Particles: always uses a 50-50 mix of pions and photons and only shoot one particle per event
    - Sampled energy: uniform distribution between 1 and 20 GeV.
    - Seed: have to set the seed for Geant4 manually to avoid duplicate events.

3. **Fixed settings** that cannot be configured at all
    - Shape of the detector: there are only tiles of configurable sizes but no other shapes available
    - Orientation: the beam impacts perpendicularly on the detector

For more information about the Simulation software used, consult the dedicated github page by J. Kieseler https://github.com/jkiesele/minicalosim (Private).