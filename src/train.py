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
        "directory" : config.data_dir,
        "start" : 0,
        "steps" : config.steps,
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
        "modes" : 8,
        "hidden_channels" : 4,
        "n_furier_layers" : 4}

    model_static = 0
    model_params = []
    for i in range(config.steps):
        model = nn.FNO(
            activation = jax.nn.relu,
            key = init_rng,
            **sq_fno_hyperparams)

        model_params_i, model_static = eqx.partition(model, eqx.is_array)
        model_params.append(model_params_i)

    if not config.sequential_mode:
        model_params = model_params[0]

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
    model_params, train_loss, val_loss, time = nn.train_model(
        model_params = model_params,
        model_static = model_static, 
        train_data_iterator = train_data_iterator,
        val_data_iterator = val_data_iterator,
        learning_rate = config.learning_rate,
        n_epochs = config.n_epochs,
        sequential_mode = config.sequential_mode)

    model = eqx.combine(model_params, model_static)

    training_stats = {
        "train_loss" : train_loss,
        "val_loss" : val_loss,
        "time" : time}

    now = datetime.now()
    datetime_str = now.strftime("%Y%m%d_%H%M%S")
    filename =f"{config.model_dir}/model_{datetime_str}.eqx"
    nn.save(filename, config._asdict(), training_stats, sq_fno_hyperparams, model)

    # Delete Data Pipeline
    del val_data_pipeline

if __name__ == "__main__":
    main(sys.argv[1:])