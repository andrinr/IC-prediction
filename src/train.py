# JAX 
import jax
from jax.lib import xla_bridge
from nvidia.dali.plugin.jax import DALIGenericIterator
# equinox
import equinox as eqx
# other
import sys
# Local
import nn
import data
from config import load_config
from datetime import datetime

def main(argv) -> None:
   
    config = load_config(argv[0])

    # JAX Settings / Device Info
    print("Jax is using %s" % xla_bridge.get_backend().platform)
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_disable_jit", False)

    dataset_params = {
        "grid_size" : config.input_grid_size,
        "grid_directory" : config.grid_dir,
        "tipsy_directory" : config.tipsy_dir,
        "start" : config.file_index_start,
        "steps" : config.file_index_steps,
        "stride" : config.file_index_stride,
        "flip" : True,
        "type" : "Test"}
    
    train_dataset = data.DirectorySequence(**dataset_params)
    train_data_pipeline = data.directory_sequence_pipe(train_dataset, config.grid_size)
    train_data_iterator = DALIGenericIterator(train_data_pipeline, ["data", "attributes"])

    dataset_params["type"] = "val"

    val_dataset = data.DirectorySequence(**dataset_params)
    val_data_pipeline = data.directory_sequence_pipe(val_dataset, config.grid_size)
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
        "in_channels" : 1,
        "out_channels" : 1,
        "hidden_channels" : 8,
        "num_levels" : 4,
        "padding" : 'SAME',
        "padding_mode" : 'CIRCULAR'}

    fno_hyperparams = {
        "modes" : 32,
        "input_channels" : 2,
        "hidden_channels" : 8,
        "output_channels" : 2,
        "n_fourier_layers" : 5}

    model = nn.SequentialModel(
        sequence_length = config.file_index_steps,
        constructor = nn.UNet if config.model_type == "UNet" else nn.FNO,
        parameters = unet_hyperparams if config.model_type == "UNet" else fno_hyperparams,
        activation = jax.nn.relu,
        unique_networks = config.unique_networks,
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
    model_params, train_loss, val_loss, time = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,
        n_epochs = config.stepwise_epochs,
        sequential_mode = False,
        single_state_loss = False)
    
    # train the model in sequential mode
    print(f"Mxied mode training for {config.sequential_epochs} epochs")
    model_params, train_loss_mixed, val_loss_mixed, time = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,
        n_epochs = config.mixed_epochs,
        sequential_mode = True,
        single_state_loss = False)
    
    # train the model in mixed mode
    print(f"Sequential mode training for {config.mixed_epochs} epochs")
    model_params, train_loss_sequential, val_loss_sequential, time = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,
        n_epochs = config.sequential_epochs,
        sequential_mode = True,
        single_state_loss = True)

    model = eqx.combine(model_params, model_static)

    training_stats = {
        "stepwise_loss" : train_loss,
        "stepwise_val_loss" : val_loss,
        "sequential_loss" : train_loss_sequential,
        "sequential_val_loss" : val_loss_sequential,
        "time" : time}

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    filename =f"{config.model_dir}/model_{datetime_str}.eqx"
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