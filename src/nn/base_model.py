import equinox as eqx
from pathlib import Path

class BaseModel(eqx.Model):
    """
    BaseModel mainly for storing and loading paramters.

    Implementation inspired by:
    NeuralOperator: https://github.com/neuraloperator/neuraloperator
    """

    def save_checkpoint(self, save_folder, save_name):
        """Saves the model state and init param in the given folder under the given name
        """
        save_folder = Path(save_folder)
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        state_dict_filepath = save_folder.joinpath(f'{save_name}_state_dict.pt').as_posix()
        torch.save(self.state_dict(), state_dict_filepath)
        metadata_filepath = save_folder.joinpath(f'{save_name}_metadata.pkl').as_posix()
        # Objects (e.g. GeLU) are not serializable by json - find a better solution in the future
        torch.save(self._init_kwargs, metadata_filepath)
        # with open(metadata_filepath, 'w') as f:
        #     json.dump(self._init_kwargs, f)

    def load_checkpoint(self, save_folder, save_name, map_location=None):
        save_folder = Path(save_folder)
        state_dict_filepath = save_folder.joinpath(f'{save_name}_state_dict.pt').as_posix()
        self.load_state_dict(torch.load(state_dict_filepath, map_location=map_location))
    