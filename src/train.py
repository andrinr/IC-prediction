import sys
# JAX 
import jax
from jax.lib import xla_bridge
from nvidia.dali.plugin.jax import DALIGenericIterator
# equinox
import equinox as eqx
# Local
import nn
import data
from config import load_config

def main(argv) -> None:
   
    config = load_config(argv[0])

    # JAX Settings / Device Info
    print("Jax is using %s" % xla_bridge.get_backend().platform)
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_disable_jit", False)

    dataset_params = {
        "grid_size" : config.input_grid_size,
        "directory" : config.data_dir,
        "start" : 0,
        "steps" : config.stride,
        "stride" : config.stride,
        "flip" : True,
        "type" : "train"}
    
    train_dataset = data.VolumetricSequence(**dataset_params)
    train_data_pipeline = data.volumetric_sequence_pipe(train_dataset, config.grid_size)
    train_data_iterator = DALIGenericIterator(train_data_pipeline, ["sequence", "step", "mean"])

    dataset_params["type"] = "val"

    val_dataset = data.VolumetricSequence(**dataset_params)
    val_data_pipeline = data.volumetric_sequence_pipe(val_dataset, config.grid_size)
    val_data_iterator = DALIGenericIterator(val_data_pipeline, ["sequence", "step", "mean"])

    # Initialize Neural Network
    init_rng = jax.random.key(0)
    # unet_hyperparams = {
    #     "num_spatial_dims" : 3,
    #     "in_channels" : 1,
    #     "out_channels" : 1,
    #     "hidden_channels" : 8,
    #     "num_levels" : 4,
    #     "padding" : 'SAME',
    #     "padding_mode" : 'CIRCULAR'}

    # model = nn.UNet(
    #     activation=jax.nn.relu,
    #     **unet_hyperparams,	
    #     key=init_rng)

    sq_fno_hyperparams = {
        "modes" : 20,
        "hidden_channels" : 3,
        "n_furier_layers" : 3}

    model = nn.FNO(
        activation = jax.nn.relu,
        key = init_rng,
        **sq_fno_hyperparams)

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

    # train the model
    model_params, train_loss, val_loss = nn.train_model(
        model_params,
        model_static, 
        train_data_iterator,
        val_data_iterator,
        config.learning_rate,
        config.n_epochs)

    model = eqx.combine(model_params, model_static)

    nn.save(config.model_params_file, sq_fno_hyperparams, model)

    # Delete Data Pipeline
    del val_data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])