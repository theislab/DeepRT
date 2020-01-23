from __future__ import print_function
from keras.optimizers import SGD
import model as mt
from keras.callbacks import LearningRateScheduler, CSVLogger, TensorBoard, ModelCheckpoint
import pandas as pd
import os
import input as i
from utils import Params, TrainOps, Logging


def main(logging, params, trainops):

    # call model
    model = mt.DeepRT(input_shape = params.img_shape,
                      enc_filters = params.enc_filters,
                      dec_filters = params.dec_filters)

    model.summary()

    # load model
    '''train and save model'''
    save_model_path = os.path.join(logging.model_directory, "weights.hdf5")

    model.compile(loss = trainops.custom_mae,
                  optimizer = SGD(lr = params.learning_rate, momentum = params.momentum),
                  metrics = [trainops.custom_mae, trainops.percentual_deviance])

    '''callbacks'''
    lr_scheduler = LearningRateScheduler(trainops.lr_schedule)

    checkpoint = ModelCheckpoint(filepath = save_model_path, monitor = 'val_custom_mae',
                                 save_best_only = True, verbose = 1, save_weights_only = True)

    tb = TensorBoard(log_dir = logging.tensorboard_directory,
                     histogram_freq = 0,
                     write_graph = True,
                     write_images = True,
                     embeddings_freq = 0,
                     embeddings_layer_names = None,
                     embeddings_metadata = None)

    # continously log loss and metric values
    csv_logger = CSVLogger(filename = logging.model_directory + '/history.csv', append = True, separator = ",")

    # get input generators and statistics
    training_generator, validation_generator, test_generator = i.get_generators(params)

    # example record
    train_sample = training_generator.example_record()
    test_sample = test_generator.example_record()

    # plot record
    trainops.plot_examples(train_sample, name = "train_record")
    trainops.plot_examples(test_sample, name = "test_record")

    # saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory + "/config.json")

    history = model.fit_generator(
        generator = training_generator,
        steps_per_epoch = 1000, #int(len(training_generator.list_IDs) / params.batch_size),
        epochs = params.num_epochs,
        validation_data = validation_generator,
        validation_steps = int(len(validation_generator.list_IDs) / params.batch_size),
        callbacks = [checkpoint, lr_scheduler, tb, csv_logger])

    pd.DataFrame(history.history).to_csv(logging.model_directory + "/loss_files.csv")


# load utils classes
params = Params("params.json")
logging = Logging("./logs", params)

trainops = TrainOps(params)
params.data_path = "./data"
params.architecture = "DeepRT"

# create logging directory
logging.create_model_directory()

params.model_directory = logging.model_directory

main(logging = logging, params = params, trainops=trainops)
