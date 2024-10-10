"""Python wrapper to provide easy config access."""
from pathlib import Path

import yaml


class DotDict(dict):
    """dot.notation access to dictionary attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def make_dotdict_recursive(data: dict) -> DotDict:
    """Transforms a dict into a dotdict, including nested dicts."""
    data = DotDict(data)
    for key in data:
        if type(data[key]) is dict:
            data[key] = DotDict(data[key])

    return data

def load_config():
    """Loads the config.yml file to a python object, where variables are accessible
    via ["KEY"] and .KEY.
    """
    with Path("config.yml").open() as config_file:
        config_dict = yaml.safe_load(config_file)
        return make_dotdict_recursive(config_dict)


config = load_config()
