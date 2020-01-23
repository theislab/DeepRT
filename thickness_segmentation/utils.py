"""General utility functions"""

import json
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2
import glob
import shutil
import tensorflow
from PIL import Image
import numpy as np
import matplotlib.gridspec as gridspec
import os
from sklearn.metrics import jaccard_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, TensorBoard


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
        self.learning_rate = None
        self.batch_size = None
        self.num_epochs = None
        self.data_path = None
        self.img_shape = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent = 4)

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

    def __create_main_directory(self):
        '''
        :return: create main log dir if not allready created
        '''
        if not os.path.isdir(self.log_dir):
            print("main logging dir does not exist, creating main logging dir ./logs")
            os.makedirs(self.log_dir)
        else:
            pass

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

        # create main dir if not exist
        self.__create_main_directory()

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
            json.dump(d, f, indent = 4)


class Evaluation():

    def __init__(self, params):
        self.params = params
        self.model_dir = params.model_directory
        self.mode = params.mode

    def plot_examples(self, record, name):
        fig = plt.figure(figsize = (24, 8))
        columns = 3
        rows = 1
        names = ["image", "label", "prediction"]
        for i in range(1, columns * rows + 1):
            img = record[i - 1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.title(names[i - 1])
            plt.savefig(os.path.join(self.params.model_directory,
                                     self.params.mode + "_predictions" +"/exmaple_{}.png".format(name)))
        plt.close()


class TrainOps():
    def __init__(self, params):
        self.params = params

    def plot_examples(self, record, name):
        fig = plt.figure(figsize = (16, 8))
        columns = 2
        rows = 1
        for i in range(1, columns * rows + 1):
            img = record[i - 1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.savefig(self.params.model_directory + "/exmaple_{}.png".format(name))
        plt.close()

    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tensorflow.reshape(y_true, [-1])
        y_pred_f = tensorflow.reshape(y_pred, [-1])
        intersection = tensorflow.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tensorflow.reduce_sum(y_true_f) + tensorflow.reduce_sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss

    def lr_schedule(self, epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = self.params.learning_rate

        if epoch > 85:
            lr *= 1e-3
        elif epoch > 80:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

    def callbacks_(self):
        '''callbacks'''
        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        checkpoint = ModelCheckpoint(filepath = self.params.model_directory + "/weights.hdf5",
                                     monitor = 'val_loss',
                                     save_best_only = True,
                                     verbose = 1,
                                     save_weights_only = True)

        tb = TensorBoard(log_dir = self.params.model_directory + "/tensorboard",
                         histogram_freq = 0,
                         write_graph = True,
                         write_images = True,
                         embeddings_layer_names = None,
                         embeddings_metadata = None)

        csv_logger = CSVLogger(filename = self.params.model_directory + '/history.csv',
                                                          append = True,
                                                          separator = ",")

        return [lr_scheduler, checkpoint, tb, csv_logger]

