from __future__ import print_function
from python_generator import DataGenerator
import pandas as pd
import os


def get_generators(params):

    train_file_names = os.path.join(params.data_path, "filenames/train_filenames_filtered.csv")
    validation_file_names = os.path.join(params.data_path, "filenames/validation_filenames_filtered.csv")
    test_file_names = os.path.join(params.data_path, "filenames/test_filenames_filtered.csv")

    train_ids = pd.read_csv(train_file_names)["ids"]
    validation_ids = pd.read_csv(validation_file_names)["ids"]
    test_ids = pd.read_csv(test_file_names)["ids"]

    partition = {'train': train_ids.values.tolist(),
                 'validation': validation_ids.values.tolist(),
                 'test': test_ids.values.tolist()}

    params.contrast_factor = 0.7
    params.brightness_factor = 0.5
    params.n_channels = 3
    params.shuffle = True

    # Generators
    training_generator = DataGenerator(partition['train'], is_training=True, params=params)
    validation_generator = DataGenerator(partition['validation'], is_training=False, params=params)
    test_generator = DataGenerator(partition['test'], is_training=False, params=params)

    return training_generator, validation_generator, test_generator
