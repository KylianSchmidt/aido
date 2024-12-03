### TODOs

[ ] Description in Simulation Parameter Dictionary is missing
[ ] Reco Loss cannot be zero for empty events, otherwise the Optimizer gets rewarded for building a detector with thick absorbers.
[x] Simulate the exact design proposed by the Optimizer at the start of each epoch in order to keep track of it
[ ] Reconstruction Loss is one-dimensional for now, return to a multi-dimensional version in the near future
[x] Line 99 in SimulationParameter is faulty when asigning int to sigma
[ ] The boundaries that get enforced by "additional constraints" should be the min and max, not the sigma boundaries. They should remain totally unchecked during Optimization.
