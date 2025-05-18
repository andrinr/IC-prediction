# JAX 
import jax
from jax.lib import xla_bridge
from nvidia.dali.plugin.jax import DALIGenericIterator
# equinox
import equinox as eqx
# other
import sys
import numpy as np
# Local
import nn
import data
from config import load_config
from datetime import datetime
import os

def main(argv) -> None:
    # Memory and performance optimizations
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".85"  # Use 85% of available GPU memory
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=4"  # Maximum autotuning
   
    config = load_config(argv[0])
    print(config)

    # JAX Settings / Device Info
    devices = jax.devices()
    print(f"Available devices: {devices}")
    print("Jax is using %s" % xla_bridge.get_backend().platform)
    jax.config.update("jax_enable_x64", False)  # Use 32-bit for better performance
    jax.config.update("jax_disable_jit", False)
    # Enable memory defragmentation
    jax.config.update("jax_debug_nans", False)
    jax.config.update("jax_debug_infs", False)

    dataset_params = {
        "grid_size" : config.input_grid_size,
        "grid_directory" : config.grid_dir,
        "start" : config.file_index_start,
        "steps" : config.file_index_steps,
        "stride" : config.file_index_stride,
        "normalizing_function" : config.normalizing_function,
        "flip" : config.flip,
        "type" : "train"}
    
    # Optimize data pipeline with prefetch and parallel processing
    train_dataset = data.DirectorySequence(**dataset_params)
    train_data_pipeline = data.directory_sequence_pipe(
        train_dataset, 
        config.grid_size,
        num_threads=4,  # Increase number of CPU threads for data loading
        prefetch_queue_depth=2  # Enable prefetching
    )
    train_data_iterator = DALIGenericIterator(train_data_pipeline, ["data", "attributes"])

    dataset_params["type"] = "val"

    val_dataset = data.DirectorySequence(**dataset_params)
    val_data_pipeline = data.directory_sequence_pipe(
        val_dataset, 
        config.grid_size,
        num_threads=4,
        prefetch_queue_depth=2
    )
    val_data_iterator = DALIGenericIterator(val_data_pipeline, ["data", "attributes"])

    # dummy_train_dataset = data.CubeData(
    #     batch_size=100,
    #     steps = config.steps,
    #     grid_size=config.input_grid_size)
    # train_data_pipeline = data.cube_sequence_pipe(dummy_train_dataset, config.grid_size)
    # train_data_iterator = DALIGenericIterator(train_data_pipeline, ["data", "step", "mean"])

    # dummy_val_dataset = data.CubeData(
    #     batch_size=10,
    #     steps = config.steps,
    #     grid_size=config.input_grid_size)
    # val_data_pipeline = data.cube_sequence_pipe(dummy_val_dataset, config.grid_size)
    # val_data_iterator = DALIGenericIterator(val_data_pipeline, ["data", "step", "mean"])


    # Initialize Neural Network
    init_rng = jax.random.key(0)
    unet_hyperparams = {
        "num_spatial_dims" : 3,
        "in_channels" : config.unet_input_channels,
        "out_channels" : config.unet_output_channels,
        "hidden_channels" : config.unet_hidden_channels,
        "num_levels" : config.unet_num_levels,
        "padding" : 'SAME',
        "padding_mode" : 'ZEROS',
        "activation" : config.activation}

    fno_hyperparams = {
        "modes" : config.fno_modes,
        "input_channels" : config.fno_input_channels,
        "hidden_channels" : config.fno_hidden_channels,
        "output_channels" : config.fno_output_channels,
        "n_fourier_layers" : config.fno_n_layers,
        "increasing_modes" : config.fno_increasing_modes,
        "activation" : config.activation}

    model = nn.SequentialModel(
        sequence_length = config.file_index_steps,
        constructor = nn.UNet if config.model_type == "UNet" else nn.FNO,
        parameters = unet_hyperparams if config.model_type == "UNet" else fno_hyperparams,
        unique_networks = config.unique_networks,
        sequential_skip_channels=config.sequential_skip_channels,
        key = init_rng)

    model_params, model_static = eqx.partition(model, eqx.is_array)

    # model = nn.Dummy(
    #     num_spatial_dims=3,
    #     channels=1,
    #     activation=jax.nn.relu,
    #     padding='SAME',
    #     padding_mode='CIRCULAR',
    #     key=init_rng) 

    parameter_count = nn.count_parameters(model)
    print(f'Number of parameters: {parameter_count}')

    # train the model in stepwise mode
    print(f"Stepwise mode training for {config.stepwise_epochs} epochs")
    model_params, metric_step = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,        
        n_epochs = config.stepwise_epochs,
        add_potential = config.include_potential,
        sequential_mode = False,
        single_state_loss = False)
    
    # train the model in mixed mode
    print(f"Mixed mode training for {config.mixed_epochs} epochs")
    model_params, metric_mixed = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,
        n_epochs = config.mixed_epochs,
        add_potential = config.include_potential,
        sequential_mode = True,
        single_state_loss = False)
    
    # train the model in sequential mode
    print(f"Sequential mode training for {config.sequential_epochs} epochs")
    model_params, metric_sequential = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,
        n_epochs = config.sequential_epochs,
        add_potential = config.include_potential,
        sequential_mode = True,
        single_state_loss = True)
    
    training_stats = {
        "metric_sequential" : metric_sequential.to_dict(),
        "metric_mixed" : metric_mixed.to_dict(),
        "metric_step" : metric_step.to_dict()}

    model = eqx.combine(model_params, model_static)

    now = datetime.now()
    datetime_str = now.strftime("%H%M%S")

    if isinstance(config.file_index_stride, list): 
        filename =f"{config.model_dir}/model_{config.file_index_stride[0]:03d}_{config.file_index_start:03d}_{config.file_index_steps:02d}_{datetime_str}.eqx"
    else:
        filename =f"{config.model_dir}/model_{config.file_index_stride:03d}_{config.file_index_start:03d}_{config.file_index_steps:02d}_{datetime_str}.eqx"
        
    # filename =f"{config.model_dir}/model_{config.file_index_stride[0]}_{datetime_str}.eqx"
    nn.save_sequential_model(
        filename, 
        config._asdict(), 
        training_stats, 
        unet_hyperparams if config.model_type == "UNet" else fno_hyperparams, 
        model)

    # Delete Data Pipeline
    del val_data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])