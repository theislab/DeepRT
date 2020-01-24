"""General utility functions"""
import keras.backend as K
import json
import matplotlib.pyplot as plt
from PIL import Image
import glob as glob
import pandas as pd
import numpy as np
import os
import shutil

from keras.utils import get_file

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


class Logging():
    def __init__(self, logging_directory, params):
        self.log_dir = logging_directory
        self.model_directory = None
        self.tensorboard_directory = None
        self.params = params

    def __create_dir(self, dir):
        os.makedirs(dir)

    def __create_tensorboard_dir(self, model_dir):

        # set abs path to new dir
        new_dir = os.path.join(model_dir, "tensorboard_dir")

        # create new dir
        self.__create_dir(new_dir)

        # set object instance to new path
        self.tensorboard_directory = new_dir

    def __remove_empty_directories(self):

        # get current directories
        current_directories = glob.glob(self.log_dir + "/*")

        # check for each dir, if weight.hdf5 file is contained
        for current_directory in current_directories:
            if not os.path.isfile(os.path.join(current_directory, "weights.hdf5")):
                # remove directory
                shutil.rmtree(current_directory)

    def create_model_directory(self):
        '''
        :param logging_directory: string, gen directory for logging
        :return: None
        '''

        # remove emtpy directories
        self.__remove_empty_directories()

        # get allready created directories
        existing_ = os.listdir(self.log_dir)

        # if first model iteration, set to zero
        if existing_ == []:
            new = 0
            # save abs path of created dir
            created_dir = os.path.join(self.log_dir, str(new))

            # make new directory
            self.__create_dir(created_dir)

            # create subdir for tensorboard logs
            self.__create_tensorboard_dir(created_dir)

        else:
            # determine the new model directory
            last_ = max(list(map(int, existing_)))
            new = int(last_) + 1

            # save abs path of created dir
            created_dir = os.path.join(self.log_dir, str(new))

            # make new directory
            self.__create_dir(created_dir)

            # create subdir for tensorboard logs
            self.__create_tensorboard_dir(created_dir)

        # set class instancy to hold abs path
        self.model_directory = created_dir

    def save_dict_to_json(self, json_path):
        """Saves dict of floats in json file
        Args:
            d: (dict) of float-castable values (np.float, int, float, etc.)
            json_path: (string) path to json file
        """
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: str(v) for k, v in self.params.dict.items()}
            json.dump(d, f, indent=4)


class TrainOps():
    def __init__(self, params):
        self.params = params

    def lr_schedule(self, epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3

        if epoch > 85:
            lr *= 1e-3
        elif epoch > 80:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def percentual_deviance(self, y_true, y_pred):
        return K.mean(K.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0])) / K.mean(y_true)

    def custom_mae(self,y_true, y_pred):
        return K.mean(K.abs((y_true[:, :, :, 0] - y_pred[:, :, :, 0])))

    def load_models(self, model, weights):
        pre_init = False
        if weights == "imagenet":
            WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                                   'releases/download/v0.2/'
                                   'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

            weights_path = get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir = 'models',
                md5_hash = 'a7b3fe01876f51b976af0dea6bc144eb')

            # get model weights
            model.load_weights(weights_path, by_name = True, skip_mismatch = True)

            print("loaded imagenet model weights")
            pre_init = True

        if weights == "thickness_map":
            weights_path = "./thickness_model_weights/weights.hdf5"
            # get model weights
            model.load_weights(weights_path, by_name = True, skip_mismatch = True)

            print("load thickness map weights")
            pre_init = True

        if not pre_init:
            print("init random weights")

    def plot_examples(self, record, name):
        fig = plt.figure(figsize = (16, 8))
        columns = 2
        rows = 1
        names = ["image", "map"]
        for i in range(1, columns * rows + 1):
            img = record[i - 1]
            fig.add_subplot(rows, columns, i)
            if names[i - 1] == "map":
                plt.imshow(img, cmap = plt.cm.jet)
            if names[i - 1] == "image":
                plt.imshow(img)

            plt.title(names[i - 1])

        plt.savefig(self.params.model_directory + "/exmaple_{}.png".format(name))
        plt.close()


class Evaluation():
    '''
    labels: list, integers
    predictions: list, integers
    history: pandas data frame
    '''

    def __init__(self, labels,predictions, softmax_output, model_dir, filenames, params):
        self.params = params
        self.labels = labels
        self.prediction = predictions
        self.model_dir = model_dir
        self.filenames = filenames
        self.history = self.get_loss_files()
        self.softmax = softmax_output

        self.mae = None
        self.mae_rel = None




