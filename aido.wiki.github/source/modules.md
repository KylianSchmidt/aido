# aido

* [aido package](aido.md)
  * [Submodules](aido.md#submodules)
  * [aido.interface module](aido.md#module-aido.interface)
    * [`UserInterfaceBase`](aido.md#aido.interface.UserInterfaceBase)
      * [`UserInterfaceBase.simulate()`](aido.md#aido.interface.UserInterfaceBase.simulate)
      * [`UserInterfaceBase.merge()`](aido.md#aido.interface.UserInterfaceBase.merge)
      * [`UserInterfaceBase.reconstruct()`](aido.md#aido.interface.UserInterfaceBase.reconstruct)
      * [`UserInterfaceBase.constraints()`](aido.md#aido.interface.UserInterfaceBase.constraints)
      * [`UserInterfaceBase.plot()`](aido.md#aido.interface.UserInterfaceBase.plot)
      * [`UserInterfaceBase.loss()`](aido.md#aido.interface.UserInterfaceBase.loss)
  * [aido.main module](aido.md#module-aido.main)
    * [`optimize()`](aido.md#aido.main.optimize)
    * [`check_results_folder_format()`](aido.md#aido.main.check_results_folder_format)
    * [`set_config()`](aido.md#aido.main.set_config)
    * [`get_config()`](aido.md#aido.main.get_config)
  * [aido.optimization_helpers module](aido.md#module-aido.optimization_helpers)
    * [`OneHotEncoder`](aido.md#aido.optimization_helpers.OneHotEncoder)
      * [`OneHotEncoder.logits`](aido.md#aido.optimization_helpers.OneHotEncoder.logits)
      * [`OneHotEncoder.forward()`](aido.md#aido.optimization_helpers.OneHotEncoder.forward)
      * [`OneHotEncoder.current_value`](aido.md#aido.optimization_helpers.OneHotEncoder.current_value)
      * [`OneHotEncoder.physical_value`](aido.md#aido.optimization_helpers.OneHotEncoder.physical_value)
      * [`OneHotEncoder.probabilities`](aido.md#aido.optimization_helpers.OneHotEncoder.probabilities)
      * [`OneHotEncoder.cost`](aido.md#aido.optimization_helpers.OneHotEncoder.cost)
    * [`ContinuousParameter`](aido.md#aido.optimization_helpers.ContinuousParameter)
      * [`ContinuousParameter.reset()`](aido.md#aido.optimization_helpers.ContinuousParameter.reset)
      * [`ContinuousParameter.forward()`](aido.md#aido.optimization_helpers.ContinuousParameter.forward)
      * [`ContinuousParameter.current_value`](aido.md#aido.optimization_helpers.ContinuousParameter.current_value)
      * [`ContinuousParameter.physical_value`](aido.md#aido.optimization_helpers.ContinuousParameter.physical_value)
      * [`ContinuousParameter.cost`](aido.md#aido.optimization_helpers.ContinuousParameter.cost)
    * [`ParameterModule`](aido.md#aido.optimization_helpers.ParameterModule)
      * [`ParameterModule.items()`](aido.md#aido.optimization_helpers.ParameterModule.items)
      * [`ParameterModule.values()`](aido.md#aido.optimization_helpers.ParameterModule.values)
      * [`ParameterModule.forward()`](aido.md#aido.optimization_helpers.ParameterModule.forward)
      * [`ParameterModule.continuous_tensors()`](aido.md#aido.optimization_helpers.ParameterModule.continuous_tensors)
      * [`ParameterModule.current_values()`](aido.md#aido.optimization_helpers.ParameterModule.current_values)
      * [`ParameterModule.physical_values()`](aido.md#aido.optimization_helpers.ParameterModule.physical_values)
      * [`ParameterModule.probabilities`](aido.md#aido.optimization_helpers.ParameterModule.probabilities)
      * [`ParameterModule.constraints`](aido.md#aido.optimization_helpers.ParameterModule.constraints)
      * [`ParameterModule.cost_loss`](aido.md#aido.optimization_helpers.ParameterModule.cost_loss)
      * [`ParameterModule.covariance`](aido.md#aido.optimization_helpers.ParameterModule.covariance)
      * [`ParameterModule.adjust_covariance()`](aido.md#aido.optimization_helpers.ParameterModule.adjust_covariance)
  * [aido.optimizer module](aido.md#module-aido.optimizer)
    * [`Optimizer`](aido.md#aido.optimizer.Optimizer)
      * [`Optimizer.to()`](aido.md#aido.optimizer.Optimizer.to)
      * [`Optimizer.check_parameters_are_local()`](aido.md#aido.optimizer.Optimizer.check_parameters_are_local)
      * [`Optimizer.boundaries()`](aido.md#aido.optimizer.Optimizer.boundaries)
      * [`Optimizer.other_constraints()`](aido.md#aido.optimizer.Optimizer.other_constraints)
      * [`Optimizer.save_parameters()`](aido.md#aido.optimizer.Optimizer.save_parameters)
      * [`Optimizer.print_grads()`](aido.md#aido.optimizer.Optimizer.print_grads)
      * [`Optimizer.optimize()`](aido.md#aido.optimizer.Optimizer.optimize)
      * [`Optimizer.boosted_parameter_dict`](aido.md#aido.optimizer.Optimizer.boosted_parameter_dict)
  * [aido.plotting module](aido.md#module-aido.plotting)
    * [`percentage_type()`](aido.md#aido.plotting.percentage_type)
    * [`Plotting`](aido.md#aido.plotting.Plotting)
      * [`Plotting.plot()`](aido.md#aido.plotting.Plotting.plot)
      * [`Plotting.parameter_evolution()`](aido.md#aido.plotting.Plotting.parameter_evolution)
      * [`Plotting.optimizer_loss()`](aido.md#aido.plotting.Plotting.optimizer_loss)
      * [`Plotting.simulation_samples()`](aido.md#aido.plotting.Plotting.simulation_samples)
      * [`Plotting.probability_evolution()`](aido.md#aido.plotting.Plotting.probability_evolution)
      * [`Plotting.fwhm()`](aido.md#aido.plotting.Plotting.fwhm)
  * [aido.scheduler module](aido.md#module-aido.scheduler)
    * [`SimulationTask`](aido.md#aido.scheduler.SimulationTask)
      * [`SimulationTask.iteration`](aido.md#aido.scheduler.SimulationTask.iteration)
      * [`SimulationTask.simulation_task_id`](aido.md#aido.scheduler.SimulationTask.simulation_task_id)
      * [`SimulationTask.num_simulation_tasks`](aido.md#aido.scheduler.SimulationTask.num_simulation_tasks)
      * [`SimulationTask.num_validation_tasks`](aido.md#aido.scheduler.SimulationTask.num_validation_tasks)
      * [`SimulationTask.start_param_dict_filepath`](aido.md#aido.scheduler.SimulationTask.start_param_dict_filepath)
      * [`SimulationTask.results_dir`](aido.md#aido.scheduler.SimulationTask.results_dir)
      * [`SimulationTask.requires()`](aido.md#aido.scheduler.SimulationTask.requires)
      * [`SimulationTask.output()`](aido.md#aido.scheduler.SimulationTask.output)
      * [`SimulationTask.run()`](aido.md#aido.scheduler.SimulationTask.run)
    * [`ReconstructionTask`](aido.md#aido.scheduler.ReconstructionTask)
      * [`ReconstructionTask.iteration`](aido.md#aido.scheduler.ReconstructionTask.iteration)
      * [`ReconstructionTask.is_validation`](aido.md#aido.scheduler.ReconstructionTask.is_validation)
      * [`ReconstructionTask.num_simulation_tasks`](aido.md#aido.scheduler.ReconstructionTask.num_simulation_tasks)
      * [`ReconstructionTask.num_validation_tasks`](aido.md#aido.scheduler.ReconstructionTask.num_validation_tasks)
      * [`ReconstructionTask.start_param_dict_filepath`](aido.md#aido.scheduler.ReconstructionTask.start_param_dict_filepath)
      * [`ReconstructionTask.results_dir`](aido.md#aido.scheduler.ReconstructionTask.results_dir)
      * [`ReconstructionTask.requires()`](aido.md#aido.scheduler.ReconstructionTask.requires)
      * [`ReconstructionTask.output()`](aido.md#aido.scheduler.ReconstructionTask.output)
      * [`ReconstructionTask.run()`](aido.md#aido.scheduler.ReconstructionTask.run)
    * [`OptimizationTask`](aido.md#aido.scheduler.OptimizationTask)
      * [`OptimizationTask.iteration`](aido.md#aido.scheduler.OptimizationTask.iteration)
      * [`OptimizationTask.num_simulation_tasks`](aido.md#aido.scheduler.OptimizationTask.num_simulation_tasks)
      * [`OptimizationTask.num_validation_tasks`](aido.md#aido.scheduler.OptimizationTask.num_validation_tasks)
      * [`OptimizationTask.results_dir`](aido.md#aido.scheduler.OptimizationTask.results_dir)
      * [`OptimizationTask.output()`](aido.md#aido.scheduler.OptimizationTask.output)
      * [`OptimizationTask.requires()`](aido.md#aido.scheduler.OptimizationTask.requires)
      * [`OptimizationTask.create_reco_path_dict()`](aido.md#aido.scheduler.OptimizationTask.create_reco_path_dict)
      * [`OptimizationTask.run()`](aido.md#aido.scheduler.OptimizationTask.run)
    * [`start_scheduler()`](aido.md#aido.scheduler.start_scheduler)
  * [aido.simulation_helpers module](aido.md#module-aido.simulation_helpers)
    * [`SimulationParameter`](aido.md#aido.simulation_helpers.SimulationParameter)
      * [`SimulationParameter.to_dict()`](aido.md#aido.simulation_helpers.SimulationParameter.to_dict)
      * [`SimulationParameter.from_dict()`](aido.md#aido.simulation_helpers.SimulationParameter.from_dict)
      * [`SimulationParameter.current_value`](aido.md#aido.simulation_helpers.SimulationParameter.current_value)
      * [`SimulationParameter.optimizable`](aido.md#aido.simulation_helpers.SimulationParameter.optimizable)
      * [`SimulationParameter.sigma`](aido.md#aido.simulation_helpers.SimulationParameter.sigma)
      * [`SimulationParameter.probabilities`](aido.md#aido.simulation_helpers.SimulationParameter.probabilities)
      * [`SimulationParameter.weighted_cost`](aido.md#aido.simulation_helpers.SimulationParameter.weighted_cost)
    * [`SimulationParameterDictionary`](aido.md#aido.simulation_helpers.SimulationParameterDictionary)
      * [`SimulationParameterDictionary.to_dict()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.to_dict)
      * [`SimulationParameterDictionary.to_json()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.to_json)
      * [`SimulationParameterDictionary.to_df()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.to_df)
      * [`SimulationParameterDictionary.get_current_values()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.get_current_values)
      * [`SimulationParameterDictionary.get_probabilities()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.get_probabilities)
      * [`SimulationParameterDictionary.update_current_values()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.update_current_values)
      * [`SimulationParameterDictionary.update_probabilities()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.update_probabilities)
      * [`SimulationParameterDictionary.sigma_array`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.sigma_array)
      * [`SimulationParameterDictionary.covariance`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.covariance)
      * [`SimulationParameterDictionary.metadata`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.metadata)
      * [`SimulationParameterDictionary.from_dict()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.from_dict)
      * [`SimulationParameterDictionary.from_json()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.from_json)
      * [`SimulationParameterDictionary.generate_new()`](aido.md#aido.simulation_helpers.SimulationParameterDictionary.generate_new)
  * [aido.surrogate module](aido.md#module-aido.surrogate)
    * [`ddpm_schedules()`](aido.md#aido.surrogate.ddpm_schedules)
    * [`NoiseAdder`](aido.md#aido.surrogate.NoiseAdder)
      * [`NoiseAdder.forward()`](aido.md#aido.surrogate.NoiseAdder.forward)
    * [`SurrogateDataset`](aido.md#aido.surrogate.SurrogateDataset)
      * [`SurrogateDataset.filter_infs_and_nans()`](aido.md#aido.surrogate.SurrogateDataset.filter_infs_and_nans)
      * [`SurrogateDataset.unnormalise_features()`](aido.md#aido.surrogate.SurrogateDataset.unnormalise_features)
      * [`SurrogateDataset.normalise_features()`](aido.md#aido.surrogate.SurrogateDataset.normalise_features)
    * [`Surrogate`](aido.md#aido.surrogate.Surrogate)
      * [`Surrogate.betas`](aido.md#aido.surrogate.Surrogate.betas)
      * [`Surrogate.t_is`](aido.md#aido.surrogate.Surrogate.t_is)
      * [`Surrogate.forward()`](aido.md#aido.surrogate.Surrogate.forward)
      * [`Surrogate.to()`](aido.md#aido.surrogate.Surrogate.to)
      * [`Surrogate.create_noisy_input()`](aido.md#aido.surrogate.Surrogate.create_noisy_input)
      * [`Surrogate.sample_forward()`](aido.md#aido.surrogate.Surrogate.sample_forward)
      * [`Surrogate.train_model()`](aido.md#aido.surrogate.Surrogate.train_model)
      * [`Surrogate.apply_model_in_batches()`](aido.md#aido.surrogate.Surrogate.apply_model_in_batches)
      * [`Surrogate.forward()`](aido.md#id0)
      * [`Surrogate.to()`](aido.md#id2)
      * [`Surrogate.update_best_surrogate_loss()`](aido.md#aido.surrogate.Surrogate.update_best_surrogate_loss)
      * [`Surrogate.create_noisy_input()`](aido.md#id3)
      * [`Surrogate.sample_forward()`](aido.md#id4)
      * [`Surrogate.train_model()`](aido.md#id5)
      * [`Surrogate.apply_model_in_batches()`](aido.md#id6)
  * [aido.surrogate_validation module](aido.md#aido-surrogate-validation-module)
  * [aido.training module](aido.md#module-aido.training)
    * [`pre_train()`](aido.md#aido.training.pre_train)
    * [`training_loop()`](aido.md#aido.training.training_loop)
  * [Module contents](aido.md#module-aido)
    * [`optimize()`](aido.md#aido.optimize)
    * [`SimulationParameter`](aido.md#aido.SimulationParameter)
      * [`SimulationParameter.to_dict()`](aido.md#aido.SimulationParameter.to_dict)
      * [`SimulationParameter.from_dict()`](aido.md#aido.SimulationParameter.from_dict)
      * [`SimulationParameter.current_value`](aido.md#aido.SimulationParameter.current_value)
      * [`SimulationParameter.optimizable`](aido.md#aido.SimulationParameter.optimizable)
      * [`SimulationParameter.sigma`](aido.md#aido.SimulationParameter.sigma)
      * [`SimulationParameter.probabilities`](aido.md#aido.SimulationParameter.probabilities)
      * [`SimulationParameter.weighted_cost`](aido.md#aido.SimulationParameter.weighted_cost)
    * [`SimulationParameterDictionary`](aido.md#aido.SimulationParameterDictionary)
      * [`SimulationParameterDictionary.to_dict()`](aido.md#aido.SimulationParameterDictionary.to_dict)
      * [`SimulationParameterDictionary.to_json()`](aido.md#aido.SimulationParameterDictionary.to_json)
      * [`SimulationParameterDictionary.to_df()`](aido.md#aido.SimulationParameterDictionary.to_df)
      * [`SimulationParameterDictionary.get_current_values()`](aido.md#aido.SimulationParameterDictionary.get_current_values)
      * [`SimulationParameterDictionary.get_probabilities()`](aido.md#aido.SimulationParameterDictionary.get_probabilities)
      * [`SimulationParameterDictionary.update_current_values()`](aido.md#aido.SimulationParameterDictionary.update_current_values)
      * [`SimulationParameterDictionary.update_probabilities()`](aido.md#aido.SimulationParameterDictionary.update_probabilities)
      * [`SimulationParameterDictionary.sigma_array`](aido.md#aido.SimulationParameterDictionary.sigma_array)
      * [`SimulationParameterDictionary.covariance`](aido.md#aido.SimulationParameterDictionary.covariance)
      * [`SimulationParameterDictionary.metadata`](aido.md#aido.SimulationParameterDictionary.metadata)
      * [`SimulationParameterDictionary.from_dict()`](aido.md#aido.SimulationParameterDictionary.from_dict)
      * [`SimulationParameterDictionary.from_json()`](aido.md#aido.SimulationParameterDictionary.from_json)
      * [`SimulationParameterDictionary.generate_new()`](aido.md#aido.SimulationParameterDictionary.generate_new)
    * [`check_results_folder_format()`](aido.md#aido.check_results_folder_format)
    * [`set_config()`](aido.md#aido.set_config)
    * [`get_config()`](aido.md#aido.get_config)
    * [`UserInterfaceBase`](aido.md#aido.UserInterfaceBase)
      * [`UserInterfaceBase.simulate()`](aido.md#aido.UserInterfaceBase.simulate)
      * [`UserInterfaceBase.merge()`](aido.md#aido.UserInterfaceBase.merge)
      * [`UserInterfaceBase.reconstruct()`](aido.md#aido.UserInterfaceBase.reconstruct)
      * [`UserInterfaceBase.constraints()`](aido.md#aido.UserInterfaceBase.constraints)
      * [`UserInterfaceBase.plot()`](aido.md#aido.UserInterfaceBase.plot)
      * [`UserInterfaceBase.loss()`](aido.md#aido.UserInterfaceBase.loss)
    * [`Plotting`](aido.md#aido.Plotting)
      * [`Plotting.plot()`](aido.md#aido.Plotting.plot)
      * [`Plotting.parameter_evolution()`](aido.md#aido.Plotting.parameter_evolution)
      * [`Plotting.optimizer_loss()`](aido.md#aido.Plotting.optimizer_loss)
      * [`Plotting.simulation_samples()`](aido.md#aido.Plotting.simulation_samples)
      * [`Plotting.probability_evolution()`](aido.md#aido.Plotting.probability_evolution)
      * [`Plotting.fwhm()`](aido.md#aido.Plotting.fwhm)
    * [`Surrogate`](aido.md#aido.Surrogate)
      * [`Surrogate.betas`](aido.md#aido.Surrogate.betas)
      * [`Surrogate.t_is`](aido.md#aido.Surrogate.t_is)
      * [`Surrogate.forward()`](aido.md#aido.Surrogate.forward)
      * [`Surrogate.to()`](aido.md#aido.Surrogate.to)
      * [`Surrogate.create_noisy_input()`](aido.md#aido.Surrogate.create_noisy_input)
      * [`Surrogate.sample_forward()`](aido.md#aido.Surrogate.sample_forward)
      * [`Surrogate.train_model()`](aido.md#aido.Surrogate.train_model)
      * [`Surrogate.apply_model_in_batches()`](aido.md#aido.Surrogate.apply_model_in_batches)
      * [`Surrogate.forward()`](aido.md#id8)
      * [`Surrogate.to()`](aido.md#id9)
      * [`Surrogate.update_best_surrogate_loss()`](aido.md#aido.Surrogate.update_best_surrogate_loss)
      * [`Surrogate.create_noisy_input()`](aido.md#id10)
      * [`Surrogate.sample_forward()`](aido.md#id11)
      * [`Surrogate.train_model()`](aido.md#id12)
      * [`Surrogate.apply_model_in_batches()`](aido.md#id13)
    * [`SurrogateDataset`](aido.md#aido.SurrogateDataset)
      * [`SurrogateDataset.filter_infs_and_nans()`](aido.md#aido.SurrogateDataset.filter_infs_and_nans)
      * [`SurrogateDataset.unnormalise_features()`](aido.md#aido.SurrogateDataset.unnormalise_features)
      * [`SurrogateDataset.normalise_features()`](aido.md#aido.SurrogateDataset.normalise_features)
