import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_random_seed(1)
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import input as i
from utils import *


def main(logging, params, trainops):
    # load model architectire and weights
    model = trainops.model(params)

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=params.learning_rate, momentum=params.momentum),
                  metrics=['accuracy'])

    # print model structure
    model.summary()

    # get standard configured data generators
    train_generator, valid_generator, test_generator = i.create_generators(params.data_path)

    # get data number of samples for training
    num_training_images, num_validation_images, num_test_images = i.get_data_statistics(params.data_path)

    '''callbacks'''
    lr_scheduler = LearningRateScheduler(trainops.step_decay)
    csv_logger = CSVLogger(filename = logging.model_directory+ '/history.csv', 
            append = True, 
            separator = ",")

    print("save directory is", logging.model_directory)
    checkpoint = ModelCheckpoint(filepath=logging.model_directory + "/weights.hdf5",
                                                    monitor='val_acc',
                                                    save_best_only=True,
                                                    verbose=1,
                                                    save_weights_only=True)

    tb = TensorBoard(log_dir=logging.tensorboard_directory,
                                     histogram_freq=0,
                                     write_graph=True,
                                     write_images=True,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)

    rlr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=20, verbose=0, mode='auto',
                                                min_delta=0.0001, cooldown=0, min_lr=0)

    # saving model config file to model output dir
    logging.save_dict_to_json(logging.model_directory + "/config.json")

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=int(num_training_images / (params.batch_size * 1)),
        epochs=params.num_epochs,
        validation_data=valid_generator,
        use_multiprocessing=False,
        workers=8,
        validation_steps=int(num_validation_images / (1)),
        callbacks=[checkpoint, lr_scheduler, tb,csv_logger, rlr])

    pd.DataFrame(history.history).to_csv(logging.model_directory + "/loss_files.csv")

    print("###################### inititing predictions and evaluations ######################")
    pred = model.predict_generator(generator=test_generator,
                                   steps=int(num_test_images / (1)),
                                   verbose=1,
                                   use_multiprocessing=False,
                                   workers=1)

    # get predictions and labels in list format
    preds = np.argmax(pred, axis=1).tolist()
    lbls = test_generator.labels.tolist()[:len(preds)]

    # instantiate the evaluation class
    evaluation = Evaluation(history=pd.DataFrame(history.history),
                            labels=lbls,
                            predictions=preds,
                            softmax_output=pred,
                            model_dir=logging.model_directory,
                            filenames=test_generator.filenames,
                            params=params)

    # get and save 5 example fundus images for each class in "./predictions and assemble to canvas"
    #evaluation.plot_examples()
    evaluation.write_plot_evaluation()

# load utils classes
params = Params("params.json")

# set string attributes of params object
params.thickness_weights = "./pretrained_weights/DeepRT_light/weights.hdf5"
logging = Logging("./logs", params)
trainops = TrainOps(params)


# insert "imagenet", "random", "DeepRT", Note here Imagenet
# weights is only available for
# Resnet50 and DeepRT weights only for DeepRT model.
params.weights_init = "DeepRT"

# choose btw light DeepRT or Resnet 50 inserting "DeepRT", "ResNet50"
params.model_version = "DeepRT"

# insert on which data partition you want to evaluate
params.partition = "512_3"

# compile full data path
params.data_path = os.path.join("./data", params.partition)

# set num epochs such that training steps remain constant
step_factor = int(100 / int(params.partition.split("_")[-1]))
params.num_epochs = params.num_epochs * step_factor

# set model directory for saving all logs and models
model_dir = logging.create_model_directory()
main(logging=logging, params=params, trainops= trainops)
