### TODOs

1. Description in Simulation Parameter Dictionary is missing
2. Reco Loss cannot be zero for empty events, otherwise the Optimizer gets rewarded for building a detector with thick absorbers.
3. Simulate the exact design proposed by the Optimizer at the start of each epoch in order to keep track of it
4. Reconstruction Loss is one-dimensional for now, return to a multi-dimensional version in the near future