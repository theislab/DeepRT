import numpy as np
import keras
import cv2
import os
import random
import skimage as sk
from skimage.transform import AffineTransform, warp
from PIL import Image, ImageOps


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, params):
        'Initialization'
        self.shape = params.img_shape
        self.batch_size = params.batch_size
        self.list_IDs = list_IDs
        self.shuffle = params.shuffle
        self.on_epoch_end()
        self.is_training = params.is_training
        self.data_path = params.data_path

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(int(len(self.list_IDs)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.shape, self.shape, 3))
        y = np.empty((self.batch_size, self.shape, self.shape, 1))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # load samples
            im = cv2.imread(os.path.join(self.data_path, "all_images", str(ID) + '.png'))
            lbl = cv2.imread(os.path.join(self.data_path, "all_labels", str(ID) + '.png'))[:, :, 0]

            # if image is color inverted, we invert it back to dark background
            if str(ID) in ["13", "14", "23", "27", "29", "30", "35", "40", "48", "50", "60", "67",
                      "72", "110", "121", "124", "125", "133", "135", "140", "143", "149", "150"]:

                # if light image, invert pixels
                im = np.array(ImageOps.invert(Image.fromarray(im)))

            # turn to one and zeros
            lbl = lbl / 255

            # resize samples
            im_resized = cv2.resize(im, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST)
            lbl_resized = cv2.resize(lbl, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST)

            # round and cast label to int
            lbl_resized = np.rint(lbl_resized).astype(int)

            # Store sample
            X[i,] = im_resized.reshape((self.shape, self.shape, 3))
            y[i,] = lbl_resized.reshape((self.shape, self.shape, 1))

            X[i,], y[i,] = self.__pre_process(X[i,], y[i,])
        return X, y.astype(np.int32)

    def __pre_process(self, train_im, label_im):

        # scaling
        train_im = np.divide(train_im, 255., dtype = np.float32)

        # label_im = np.nan_to_num(label_im)
        train_im = np.nan_to_num(train_im)
        if self.is_training:
            self.augment(train_im, label_im)
        return train_im.reshape(self.shape, self.shape, 3), label_im.reshape((self.shape, self.shape, 1))

    def augment(self, train_im, label_im):
        # get boolean if rotate
        flip_hor = bool(random.getrandbits(1))
        flip_ver = bool(random.getrandbits(1))
        rot90_ = bool(random.getrandbits(1))
        shift = bool(random.getrandbits(1))
        noise = bool(random.getrandbits(1))
        rotate = bool(random.getrandbits(1))

        if (rotate):
            train_im, label_im = self.random_rotation(train_im, label_im)
        if (shift):
            train_im, label_im = self.random_shift(train_im, label_im)
        if (noise):
            train_im = self.gaussian_noise(train_im)

        # label preserving augmentations
        if rot90_:
            train_im, label_im = self.rot90(train_im, label_im)
        if (flip_hor):
            train_im, label_im = self.flip_horizontally(train_im, label_im)
        if (flip_ver):
            train_im, label_im = self.flip_vertically(train_im, label_im)

        return train_im, label_im

    def random_rotation(self, image_array, label_array):

        (h, w) = image_array.shape[:2]
        center = (w / 2, h / 2)

        random_degree = random.uniform(-50, 50)
        # rotate the image by 180 degrees
        M = cv2.getRotationMatrix2D(center, random_degree, 1.0)

        r_im = cv2.warpAffine(image_array, M, (w, h), cv2.INTER_NEAREST)
        l_im = cv2.warpAffine(label_array, M, (w, h), cv2.INTER_NEAREST)
        return r_im, l_im

    def flip_vertically(self, image_array, label_array):
        flipped_image = np.fliplr(image_array)
        flipped_label = np.fliplr(label_array)
        return flipped_image, flipped_label

    def rot90(self, image_array, label_array):
        rot_image_array = np.rot90(image_array)
        rot_label_array = np.rot90(label_array)
        return rot_image_array, rot_label_array

    def flip_horizontally(self, image_array, label_array):
        flipped_image = np.flipud(image_array)
        flipped_label = np.flipud(label_array)
        return flipped_image, flipped_label

    def random_noise(self, image_array):
        # add random noise to the image
        return sk.util.random_noise(image_array)

    def shift(self, image, vector):
        transform = AffineTransform(translation = vector)
        shifted = warp(image, transform, mode = 'wrap', preserve_range = True)

        shifted = shifted.astype(image.dtype)
        return shifted

    def gaussian_noise(self, image_array):
        '''
        :param image_array: Image onto which gaussian noise is added, numpy array, float
        :return: transformed image array
        '''
        value = random.uniform(0, 1)
        image_array = image_array + np.random.normal(0, value)
        return image_array

    def random_shift(self, image_array, label_array):

        rand_x = random.uniform(-40, 40)
        rand_y = random.uniform(-40, 40)

        image_array = self.shift(image_array, (rand_x, rand_y))
        label_array = self.shift(label_array, (rand_x, rand_y))

        return (image_array.reshape(self.shape, self.shape, 3),
                label_array.reshape(self.shape, self.shape, 1))

    def example_record(self):

        record_idx = random.randint(0, len(self.list_IDs))
        im = cv2.imread(os.path.join(self.data_path, "all_images", str(self.list_IDs[record_idx]) + '.png'))
        lbl = cv2.imread(os.path.join(self.data_path, "all_labels", str(self.list_IDs[record_idx]) + '.png'))[:, :, 0]

        # if image is color inverted, we invert it back to dark background
        if str(id) in ["13", "14", "23", "27", "29", "30", "35", "40", "48", "50", "60", "67",
                       "72", "110", "121", "124", "125", "133", "135", "140", "143", "149", "150"]:
            # if light image, invert pixels
            im = np.array(ImageOps.invert(Image.fromarray(im)))

        # turn to one and zeros
        lbl = lbl / 255

        # resize samples
        im_resized = cv2.resize(im, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST)
        lbl_resized = cv2.resize(lbl, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST)

        # cast label to int
        lbl_resized = np.rint(lbl_resized).astype(int)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis = -1)

        # Store sample
        image = im_resized.reshape(self.shape, self.shape, 3)
        label = lbl_resized.reshape((self.shape, self.shape, 1))

        record = (image, label[:, :, 0])
        return record

    def get_record(self, id):
        im = cv2.imread(os.path.join(self.data_path, "all_images", str(id) + '.png'))
        lbl = cv2.imread(os.path.join(self.data_path, "all_labels", str(id) + '.png'))[:, :, 0]

        # if image is color inverted, we invert it back to dark background
        if str(id) in ["13", "14", "23", "27", "29", "30", "35", "40", "48", "50", "60", "67",
                       "72", "110", "121", "124", "125", "133", "135", "140", "143", "149", "150"]:
            # if light image, invert pixels
            im = np.array(ImageOps.invert(Image.fromarray(im)))

        # turn to one and zeros
        lbl = lbl / 255

        # resize samples
        im_resized = cv2.resize(im, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST)
        lbl_resized = cv2.resize(lbl, (self.shape, self.shape), interpolation = cv2.INTER_NEAREST)

        # cast label to int
        lbl_resized = np.rint(lbl_resized).astype(int)

        # if image grey scale, make 3 channel
        if len(im_resized.shape) == 2:
            im_resized = np.stack((im_resized,) * 3, axis = -1)

        # Store sample
        image = im_resized.reshape(1, self.shape, self.shape, 3)
        label = lbl_resized.reshape((1, self.shape, self.shape, 1))

        # scaling
        image = np.divide(image, 255., dtype = np.float32)

        return image, label
