"""General utility functions"""

import json
# import matplotlib.pyplot as plt
from PIL import Image
import glob as glob
import pandas as pd
import numpy as np
import os
import shutil
from keras.utils import get_file
from sklearn.metrics import log_loss
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import balanced_accuracy_score
import keras
from deepRT import DeepRT


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
            json.dump(d, f, indent = 4)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


class Evaluation():
    '''
    labels: list, integers
    predictions: list, integers
    history: pandas data frame
    '''

    def __init__(self, labels, predictions, softmax_output, model_dir, filenames, params):
        self.params = params
        self.labels = labels
        self.prediction = predictions
        self.model_dir = model_dir
        self.filenames = filenames
        # self.history = self.get_loss_files()
        self.softmax = softmax_output

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.precision_weighted = None
        self.recall_weighted = None
        self.confusion_matrix = None
        self.cohen_qk = None
        self.weighted_accuracy = None
        self.binary_precision = None
        self.binary_recall = None
        self.binary_labels = None
        self.binary_predictions = None
        self.binary_precision_weighted = None
        self.binary_recall_weighted = None

    def __accuracy(self):
        return (accuracy_score(self.labels, self.prediction))

    def __balanced_accuracy(self):
        return (balanced_accuracy_score(self.labels, self.prediction))

    def __precision(self):
        return (precision_score(self.labels, self.prediction, average = 'macro'))

    def __precision_weighted(self):
        return (precision_score(self.labels, self.prediction, average = 'weighted'))

    def __binary_precision(self):
        return (precision_score(self.binary_labels, self.binary_predictions, average = 'macro'))

    def __binary_precision_weighted(self):
        return (precision_score(self.binary_labels, self.binary_predictions, average = 'weighted'))

    def __recall(self):
        return (recall_score(self.labels, self.prediction, average = 'macro'))

    def __recall_weighted(self):
        return (recall_score(self.labels, self.prediction, average = 'weighted'))

    def __binary_recall(self):
        return (recall_score(self.binary_labels, self.binary_predictions, average = 'macro'))

    def __binary_recall_weighted(self):
        return (recall_score(self.binary_labels, self.binary_predictions, average = 'weighted'))

    def __confusion_matrix(self):
        return (confusion_matrix(self.labels, self.prediction))

    def __log_loss(self):
        return (log_loss(y_true = self.labels, y_pred = self.softmax, labels = [0, 1, 2, 3, 4]))

    def __cohens_qk(self):
        return cohen_kappa_score(y1 = self.labels, y2 = self.prediction, weights = "quadratic")

    def __binarize_labels_prediction(self):

        lbl_pd = pd.DataFrame(self.labels)
        lbl_pd[lbl_pd == 2] = 1
        lbl_pd[lbl_pd == 3] = 1
        lbl_pd[lbl_pd == 4] = 1

        self.binary_labels = lbl_pd[0].tolist()

        pred_pd = pd.DataFrame(self.prediction)
        pred_pd[pred_pd == 2] = 1
        pred_pd[pred_pd == 3] = 1
        pred_pd[pred_pd == 4] = 1

        self.binary_predictions = pred_pd[0].tolist()

    def get_loss_files(self):
        import pandas as pd
        from functools import reduce
        from tensorboard.backend.event_processing import event_accumulator

        # get tensorboard file
        event_file = os.listdir(os.path.join(self.model_dir, "tensorboard_dir"))
        ea = event_accumulator.EventAccumulator(os.path.join(self.model_dir, "tensboard_dir", event_file[0]),
                                                size_guidance = {  # see below regarding this argument
                                                    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
                                                    event_accumulator.IMAGES: 4,
                                                    event_accumulator.AUDIO: 4,
                                                    event_accumulator.SCALARS: 1000,
                                                    event_accumulator.HISTOGRAMS: 1,
                                                })

        ea.Reload()
        loss = pd.DataFrame(ea.Scalars('loss'))
        acc = pd.DataFrame(ea.Scalars('acc'))
        val_loss = pd.DataFrame(ea.Scalars('val_loss'))
        val_acc = pd.DataFrame(ea.Scalars('val_acc'))

        dfs = [loss, acc, val_loss, val_acc]

        df_final = reduce(lambda left, right: pd.merge(left, right, on = 'step'), dfs)

        df_final = df_final.rename(
            columns = {"value_x": "loss", "value_y": "acc", "value_x": "val_loss", "value_y": "val_acc"})

        df_final = df_final.drop(columns = ["wall_time_x", "wall_time_y"])
        return df_final

    def __filenames(self):
        # generate example predictions
        pred_im = pd.DataFrame(self.filenames)
        pred_im_pd = pred_im[0].str.split("/", expand = True)
        pred_im_pd = pred_im_pd.rename(columns = {0: "labels", 1: "id"})
        return (pred_im_pd)

    def __plot_confusion_matrix(self, normalize=True, title=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        y_true = self.labels
        y_pred = self.prediction
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.matshow(cm, cmap = plt.cm.Blues)

        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]), size = "large",
                         horizontalalignment = "center",
                         color = "white" if cm[i, j] > thresh else "black")

        plt.title("confusion matrix")
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # save cm as txt file
        np.savetxt(self.model_dir + "/cm.txt", cm, fmt = '%s')

        plt.savefig(os.path.join(self.model_dir, "confusion_matrix.png"))
        return

    def __plot_history(self):
        plt.rcParams.update({'font.size': 16})
        f, axs = plt.subplots(2, 2, figsize = (10, 10))

        # load loss curves
        statistics_pd = self.history

        if 'lr' in statistics_pd:
            plt.suptitle("Train statistics")
            for i in range(1, 4):
                plt.subplot(3, 1, i)
                if i == 1:
                    plt.plot(statistics_pd["loss"], label = "train loss")
                    plt.plot(statistics_pd["val_loss"], label = "validation loss")
                    plt.xlabel("epochs")
                    plt.ylabel("cross entropy")
                    plt.legend()
                if i == 2:
                    plt.plot(statistics_pd["acc"], label = "train accuracy")
                    plt.plot(statistics_pd["val_acc"], label = "validation accuracy")
                    plt.xlabel("epochs")
                    plt.ylabel("accuracy")
                    plt.legend()
                if i == 3:
                    plt.plot(statistics_pd["lr"], label = "learning rate decay")
                    plt.xlabel("epochs")
                    plt.ylabel("lr")
                    plt.legend()

        # plor without learning rate
        else:
            plt.suptitle("Train statistics")
            for i in range(1, 3):
                plt.subplot(2, 1, i)
                if i == 1:
                    plt.plot(statistics_pd["loss"], label = "train loss")
                    plt.plot(statistics_pd["val_loss"], label = "validation loss")
                    plt.xlabel("epochs")
                    plt.ylabel("cross entropy")
                    plt.legend()
                if i == 2:
                    plt.plot(statistics_pd["acc"], label = "train accuracy")
                    plt.plot(statistics_pd["val_acc"], label = "validation accuracy")
                    plt.xlabel("epochs")
                    plt.ylabel("accuracy")
                    plt.legend()

        plt.grid(True)
        plt.savefig(self.model_dir + "/history.png")

    def __save_example_predictions(self, params):

        # data frame with filenames and labels of test predictions
        pred_im_pd = self.__filenames()

        # only take the names of which we have predictions
        if pred_im_pd.shape[0] > len(self.prediction):
            pred_im_pd = pred_im_pd.iloc[:len(self.prediction)]

        # test prediction added
        pred_im_pd["predictions"] = self.prediction

        # set label levels
        levels = ["0", "1", "2", "3", "4"]

        for level in levels:
            pred_im_class_pd = pred_im_pd[pred_im_pd["labels"] == level]

            # shuffle indices
            pred_im_class_pd = pred_im_class_pd.sample(frac = 1)

            # save ten predictions
            ten_im = pred_im_class_pd.iloc[0:5]

            for im_name in ten_im["id"]:
                pred_class = ten_im[ten_im["id"] == im_name].predictions.values[0]
                im_path = os.path.join(params.data_path, "test", level, im_name)

                # create save directory if does not exist
                if not os.path.exists(os.path.join(self.model_dir, "predictions", level)):
                    os.makedirs(os.path.join(self.model_dir, "predictions", level))

                outcome_string = "__true__" + str(level) + "__pred__" + str(pred_class) + ".jpeg"
                save_example_name = im_name.replace(".jpeg", outcome_string)

                fundus_im = np.array(Image.open(im_path))

                plt.imsave(os.path.join(self.model_dir, "predictions", level, save_example_name), fundus_im)

    def __example_prediction_canvas(self):
        plt.rcParams.update({'font.size': 5})
        example_prediction_paths = glob.glob(self.model_dir + "/predictions/**/*")

        fig = plt.figure(figsize = (10, 10))
        # set figure proportion after number of examples created
        columns = int(len(example_prediction_paths) / 5)
        rows = 5
        for i in range(1, columns * rows + 1):
            img = np.array(Image.open(example_prediction_paths[i - 1]))
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.title(example_prediction_paths[i - 1].split("/")[-1].replace(".jpeg", ""))
            plt.axis('off')

        plt.savefig(os.path.join(self.model_dir, "example_canvas.png"))

    def __main_result(self):

        # create binary labels/predictions
        self.__binarize_labels_prediction()
        '''init all metrics'''
        self.accuracy = self.__accuracy()
        self.precision = self.__precision()
        self.precision_weighted = self.__precision_weighted()
        self.recall_weighted = self.__recall_weighted()
        self.recall = self.__recall()
        self.ce = self.__log_loss()
        self.cohen_qk = self.__cohens_qk()
        self.weighted_accuracy = self.__balanced_accuracy()
        self.binary_precision = self.__binary_precision()
        self.binary_recall = self.__binary_recall()
        self.binary_precision_weighted = self.__binary_precision_weighted()
        self.binary_recall_weighted = self.__binary_recall_weighted()

        # dump all stats in txt file
        result_array = np.array(["accuracy", self.accuracy,
                                 "precision", self.precision,
                                 "recall", self.recall,
                                 "cross_entropy", self.ce,
                                 "cohens_qk", self.cohen_qk,
                                 "weighted_accuracy", self.weighted_accuracy,
                                 "binary_precision", self.binary_precision,
                                 "binary_recall", self.binary_recall,
                                 "recall_weighted", self.recall_weighted,
                                 "precision_weighted", self.precision_weighted,
                                 "binary_recall_weighted", self.binary_recall_weighted,
                                 "binary_precision_weighted", self.binary_precision_weighted])

        np.savetxt(self.model_dir + "/result.txt", result_array, fmt = '%s')

    def write_plot_evaluation(self):
        self.__main_result()
        # self.__plot_confusion_matrix()
        # self.__plot_history()

    def plot_examples(self):
        self.__save_example_predictions(self.params)
        self.__example_prediction_canvas()


class TrainOps():
    def __init__(self, params):
        self.params = params

    def step_decay(self, epoch):
        """Learning Rate Schedule

        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.

        # Arguments
            epoch (int): The number of epochs

        # Returns
            lr (float32): learning rate
        """
        lr = self.params.learning_rate
        if epoch >= int(self.params.num_epochs / 3):
            lr *= 5e-1
        if epoch >= int(self.params.num_epochs / 2):
            lr *= 2e-1
        if epoch >= int(self.params.num_epochs / 3) * 2:
            lr *= 5e-1
        if epoch >= int((self.params.num_epochs / 3) * 2.25):
            lr *= 2e-1
        print('Learning rate: ', lr)

        return lr

    def set_trainable(self, model):
        for l in model.layers[0:-1]:
            l.trainable = False
            print(l.name, l.trainable)

    def load_model_test(self, model, model_dir):

        # save all weights before loading
        preloaded_layers = model.layers.copy()
        preloaded_weights = []
        for pre in preloaded_layers:
            preloaded_weights.append(pre.get_weights())

        model.load_weights(model_dir + "/weights.hdf5", by_name = True, skip_mismatch = True)

        # controll loading of weights
        for layer, pre in zip(model.layers, preloaded_weights):
            weights = layer.get_weights()

            if weights:
                if np.array_equal(weights[0], pre[0]):
                    print('not loaded', layer.name)
                else:
                    print('loaded', layer.name)

    def load_models(self, model, params):

        # save all weights before loading
        preloaded_layers = model.layers.copy()
        preloaded_weights = []
        for pre in preloaded_layers:
            preloaded_weights.append(pre.get_weights())

        if params.weights_init == "random":
            print("init random weights")

        if params.weights_init == "imagenet":
            if params.model_version == "ResNet50":
                print("loading imagenet weights")

                WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                                       'releases/download/v0.2/'
                                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

                weights_path = get_file(
                    'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                    WEIGHTS_PATH_NO_TOP,
                    cache_subdir = 'models',
                    md5_hash = 'a7b3fe01876f51b976af0dea6bc144eb')

                # get model weights
                model.load_weights(weights_path, by_name = True)

                print("loaded imagenet model weights")

        if params.weights_init == "DeepRT":
            print("loading thickness_map")
            # get model weights
            model.load_weights(params.thickness_weights,
                               by_name = True)

            print("load thickness map weights")

        if params.weights_init != 0:

            # controll loading of weights
            for layer, pre in zip(model.layers, preloaded_weights):
                weights = layer.get_weights()

                if weights:
                    if np.array_equal(weights[0], pre[0]):
                        print('not loaded', layer.name)
                    else:
                        print('loaded', layer.name)

    def model(self, params):

        if params.model_version not in ["DeepRT", "ResNet50"]:
            print("Not available model")
            exit()

        """load model"""
        if params.model_version == "ResNet50":
            model = keras.applications.ResNet50(include_top = True,
                                                weights = None,
                                                input_tensor = None,
                                                input_shape = (params.img_shape, params.img_shape, 3),
                                                pooling = "avg",
                                                classes = 5)

        if params.model_version == "DeepRT":
            model = DeepRT(params,
                           n = 2,
                           num_classes = 5)

        '''load model weights'''
        if params.continue_training == 0:
            # get model weights
            self.load_models(model, params)
        else:
            model_dir = "./logs_/1"
            model.load_weights(model_dir + "/weights.hdf5",
                               by_name = True)  # .load_weights(model_dir+"/weights.hdf5")
            print("loaded trained model under configuration")

        return model
