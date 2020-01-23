from keras.preprocessing.image import ImageDataGenerator
import glob as glob
from utils import Params
import os
import numpy as np

params = Params("params.json")

def get_data_statistics(data_path):
    # number of training images
    num_training_images = len(glob.glob(os.path.join(data_path + "/train", "*", "*.jpeg")))
    num_validation_images = len(glob.glob(os.path.join(data_path + "/validation", "*", "*.jpeg")))
    num_test_images = len(glob.glob(os.path.join(data_path + "/test", "*", "*.jpeg")))

    print("number of train, validation and test images are:",
          num_training_images,
          num_validation_images,
          num_test_images)

    return num_training_images, num_validation_images, num_test_images


def apply_transform(x):
    transform_parameters = {'red_mean': 138.91592015833191,
                            'green_mean': 135.26675259654633,
                            'blue_mean': 133.17857971137104,
                            'var': 1379.760843672262}

    im_processed = np.copy(x).astype(np.float64)

    for k, i in enumerate( ["red_mean", "green_mean", "blue_mean"] ):
        # subtract feature wise means
        im_processed[:, :, k] = im_processed[:, :, k] - transform_parameters[i]

    return im_processed / np.sqrt(np.float(transform_parameters["var"]))


def create_generators(data_path):
    print('Using real-time data augmentation.')


    train_datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.0,
        height_shift_range=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=apply_transform)

    test_datagen = ImageDataGenerator(
        preprocessing_function=apply_transform)


    train_generator = train_datagen.flow_from_directory(
        directory=data_path + "/train",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=params.batch_size,
        shuffle=True,
        seed=1,
        class_mode="categorical")

    print('train_generator created')

    valid_generator = test_datagen.flow_from_directory(
        directory=data_path + "/validation",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

    print('validation_generator created')

    test_generator = test_datagen.flow_from_directory(
        directory=data_path + "/test",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

    print('test_generator created')

    return (train_generator, valid_generator, test_generator)


def create_test_generator(data_path):
    test_datagen = ImageDataGenerator(
        preprocessing_function=apply_transform,
    )

    test_generator = test_datagen.flow_from_directory(
        directory=data_path + "/test",
        target_size=(params.img_shape, params.img_shape),
        color_mode='rgb',
        batch_size=1,
        shuffle=False,
        class_mode="categorical")

    print('test_generator created')

    return test_generator


def get_test_statistics(data_path):
    # number of training images
    num_test_images = len(glob.glob(os.path.join(data_path + "/test", "*", "*.jpeg")))

    print("number of test images are:", num_test_images)

    return (num_test_images)
