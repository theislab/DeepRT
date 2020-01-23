import random
from PIL import Image
from PIL import ImageOps
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import os
import random
import skimage as sk
from skimage import transform


import regex as re
from numpy import inf

def random_rotation(image_array, label_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    r_im = sk.transform.rotate(image_array, random_degree, preserve_range = True)
    r_l = sk.transform.rotate(label_array, random_degree, preserve_range = True)
    return r_im.astype(np.float32), r_l.astype(np.uint8)

def elastic_transform(image, mask, alpha=720, sigma=24, alpha_affine=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))


    res_x = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    res_y = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    return res_x, res_y

def get_unique_string(im_name):
    '''
    :param im_name: a string with the image file name
    :return: the unique identifier for that image
    '''
    im_parts = im_name.split("_")

    strings = []
    for parts in im_parts:
        if re.findall('\d+', parts):
            strings.append(re.findall('\d+', parts))

    unique_str = strings[0][0] + "_" + strings[1][0]
    return (unique_str)

def padding_with_zeros(im, orig_shape, new_shape):
    '''
    :param im:
    :param orig_shape:
    :param new_shape:
    :return:
    '''
#    im = im.reshape(orig_shape)
    result = np.zeros([int(new_shape[0]), int(new_shape[1])])
    #print(new_shape[0],new_shape[1], orig_shape[0], orig_shape[1])
    x_offset = int((new_shape[0] - orig_shape[0])/2)  # 0 would be what you wanted
    y_offset = int((new_shape[1] - orig_shape[1])/2)  # 0 in your case
    #print(x_offset, y_offset)
    
    result[x_offset:im.shape[0]+x_offset,y_offset:im.shape[1]+y_offset] = im
    return(result)

def get_clinic_train_data(im_dir, seg_dir,  img_width, img_height,batch_size,iter):
    '''
    :param im_dir: directors for images
    :param seg_dir: dir for groundtruth
    :param batch_size: number of images to load
    :return: returns arrays of  a batch, im are floats and gt are int32
    '''

    # get all files from each dir
    im_names = os.listdir(im_dir)
    seg_names = os.listdir(seg_dir)
    # random int for selecting images
    random_int = np.random.choice(len(im_names),batch_size)
    # random_int = 1
    # set containers holding data
    images = []
    seg_maps = []
    im_id = []
    im_displayed = []
    # set sizes
    size = [img_width, img_height]
    # gather data
    k = 0
    for i in range(batch_size):
        im_name = im_names[iter]
        unique_str = re.findall(r'\d+', im_name)
        im_displayed.append(unique_str)
        # print("Just feeding same image")

        if batch_size > k:
            k += 1
            # retrieve image
            train_im = Image.open(im_dir + im_name).convert('L')
            train_im = np.array(train_im)

            orig_shape = [train_im.shape[0], train_im.shape[1]]
            new_shape = [train_im.shape[0], train_im.shape[1] * np.divide(img_width, img_height)]
            #print(new_shape, orig_shape)
            # print(train_im.shape)
            im_padded = padding_with_zeros(train_im, orig_shape, new_shape)
            # im = np.array(im)
            im_resized = Image.fromarray(im_padded).resize(size, Image.NEAREST)

            # scale image
            im_resized = np.subtract(np.divide(im_resized, 255, dtype=np.float32), 0.5, dtype=np.float32)

            y_path = [s for s in seg_names if unique_str == re.findall(r'\d+', s)]

            seg_im = Image.open(seg_dir + y_path[0])  # .convert('L')
            seg_im = np.array(seg_im)

            seg_padded = padding_with_zeros(seg_im, orig_shape, new_shape)
            # im = np.array(im)
            seg_resized = Image.fromarray(seg_padded).resize(size)

            # get random numbers to determine whether to do data augmentation or not
            float_horizontal_flip = 0#random.uniform(0.0, 1.0)
            float_vertical_flip = 0#random.uniform(0.0, 1.0)
            float_elastic_tranform = 0#random.uniform(0.0, 1.0)
            float_rotation = 0#random.uniform(0.0, 1.0)

            # 50 % chance that both im and seg is flipped
            if float_horizontal_flip > 0.5:
                im_resized = np.fliplr(im_resized)
                seg_resized = np.fliplr(seg_resized)
            im_resized = np.array(im_resized)
            seg_resized = np.array(seg_resized, dtype=np.uint8)
            #
            if float_vertical_flip > 0.5:
                im_resized = np.flip(im_resized, 0)
                seg_resized = np.flip(seg_resized, 0)
            im_resized = np.array(im_resized)
            seg_resized = np.array(seg_resized, dtype=np.uint8)

            if float_elastic_tranform > 0.5:
                im_resized, seg_resized = elastic_transform(im_resized, seg_resized)

            if float_rotation > 0.5:
                im_resized, seg_resized = random_rotation(im_resized, seg_resized)

            #invert the images with reversed colors
            if np.mean(im_resized) > 90:
                im_resized = np.invert(im_resized)

            images.append(im_resized)
            seg_maps.append(seg_resized)

    im_batch = np.reshape(np.asarray(images), (batch_size, img_height, img_width, 1))
    labels_batch = np.reshape(np.asarray(seg_maps, dtype = np.int32), (batch_size, img_height, img_width, 1))

    return (im_batch, labels_batch, im_displayed, new_shape,orig_shape)