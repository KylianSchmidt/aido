### TODOs

[x] Description in Simulation Parameter Dictionary is missing
[ ] Reco Loss cannot be zero for empty events, otherwise the Optimizer gets rewarded for building a detector with thick absorbers.
[x] Simulate the exact design proposed by the Optimizer at the start of each epoch in order to keep track of it
[ ] Reconstruction Loss is one-dimensional for now, return to a multi-dimensional version in the near future
[x] Line 99 in SimulationParameter is faulty when asigning int to sigma
[x] The boundaries that get enforced by "additional constraints" should be the min and max, not the sigma boundaries. They should remain totally unchecked during Optimization.
[ ] File sizes can add up. Clean up the Simulation Files and Model saves once they are used.
[ ] Allow for an extra validation step for the reconstruction algorithm for the end user. For example a separate Task that resimulates the same detector configurations and passes it to interface.
[x] Use a try-except clause to allocate memory for the surrogate and optimizer models (or check if b2luigi can wait until memory is freed up)
[ ] Argument in scheduler that decides whether to re-raise Exceptions from user-defined plotting (e.g. ignore_plotting_exceptions=True)
[x] Check if the min max values are correctly enforced when calling current_value on SimulationParameter
[x] Probabilities of discrete parameters are not propagated from the ParameterModule to the SimulationParameterDictionary in Optimizer.
[x] Add 'check if all probabilities were changed' in SimulationParameterDictionary
[x] Split merge and reconstruct in ReconstructionTask (to allow for multiple envs)
[ ] Keep the best version of the parameters of the optimizer.