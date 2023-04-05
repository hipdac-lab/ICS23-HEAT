import glob
import yaml


def load_config(config_path):
    with open(config_path, 'r') as c_f0:
        config_dic = yaml.safe_load(c_f0)

    return config_dic