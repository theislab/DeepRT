import os
import model as mt
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
from python_generator import DataGenerator
from utils import Params, Logging, TrainOps, Evaluation

# load utils classes
params = Params("params.json")

params.data_dir = "./data"
params.continuing_training = False
params.batchnorm = True
params.is_training = True
params.shuffle = True

logging = Logging("./logs", params)

# create logging directory
logging.create_model_directory()

# add to params
params.model_directory = logging.model_directory

# saving model config file to model output dir
logging.save_dict_to_json(logging.model_directory + "/config.json")

# get train ops
trainops = TrainOps(params)

# get file names for generator
train_file = os.path.join(params.data_dir, "file_names_complete", "train_new_old_mapping.csv")
val_file = os.path.join(params.data_dir, "file_names_complete", "validation_new_old_mapping.csv")

train_files = pd.read_csv(train_file, index_col = False)["new_id"]
val_files = pd.read_csv(val_file)["new_id"]

partition = {"train": train_files.tolist(), "validation": val_files.tolist()}

num_training_examples = len(partition['train'])
num_val_examples = len(partition['validation'])

# Generators
training_generator = DataGenerator(partition['train'], params)
validation_generator = DataGenerator(partition['validation'], params)

# get model
model = mt.get_deep_unet(params)

'''Compile model'''
model.compile(optimizer = Adam(lr = params.learning_rate),
              loss = trainops.dice_loss,
              metrics = [trainops.dice_coeff])

model.summary()

'''train and save model'''
save_model_path = os.path.join(params.model_directory,
                               "weights.hdf5")

cp = tf.keras.callbacks.ModelCheckpoint(filepath = save_model_path, monitor = "val_loss",
                                        save_best_only = True, verbose = 1, save_weights_only = True)

'''Load models trained weights'''
if params.continuing_training:
    model.load_weights(save_model_path, by_name = True, skip_mismatch = True)

# example record
train_sample = training_generator.example_record()
test_sample = training_generator.example_record()

# plot record
trainops.plot_examples(train_sample, name="train_record")
trainops.plot_examples(test_sample, name="test_record")


# Train model on data set
history = model.fit_generator(generator = training_generator,
                              validation_data = validation_generator,
                              use_multiprocessing = False,
                              steps_per_epoch = int(num_training_examples / params.batch_size),
                              validation_steps = int(num_val_examples / params.batch_size),
                              epochs = params.epochs,
                              verbose = 1,
                              workers = 5,
                              callbacks = [cp])
