
import yaml
from .config import Config

def load_config(file_name : str) -> Config:
    with open(file_name, 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)

        return Config(**config_data)