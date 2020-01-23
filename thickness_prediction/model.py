import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.utils.data_utils import get_file
from keras.regularizers import l2


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def stage(x, num_filters_in, stage, num_res_blocks):
    for res_block in range(num_res_blocks):
        activation = 'relu'
        batch_normalization = True
        strides = 1
        if stage == 0:
            num_filters_out = num_filters_in * 4
            if res_block == 0:  # first layer and first stage
                activation = None
                batch_normalization = False
        else:
            if stage < 1:
                num_filters_out = num_filters_in * 2
            else:
                num_filters_out = num_filters_in
            if res_block == 0:  # first layer but not first stage
                strides = 2  # downsample

        # bottleneck residual unit
        y = resnet_layer(inputs=x,
                         num_filters=num_filters_in,
                         kernel_size=1,
                         strides=strides,
                         activation=activation,
                         batch_normalization=batch_normalization,
                         conv_first=False)
        y = resnet_layer(inputs=y,
                         num_filters=num_filters_in,
                         conv_first=False)

        y = resnet_layer(inputs=y,
                         num_filters=num_filters_out,
                         kernel_size=1,
                         conv_first=False)

        if res_block == 0:
            # linear projection residual shortcut connection to match
            # changed dims
            x = resnet_layer(inputs=x,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             strides=strides,
                             activation=None,
                             batch_normalization=False)
        x = keras.layers.add([x, y])

    num_filters_in = num_filters_out

    return x, num_filters_in


def DeepRT(input_shape, enc_filters, dec_filters=16, dropout=1.0, batchnorm=True, training=True):

    # Start model definition.
    num_filters_in = enc_filters

    inputs = Input(shape=(input_shape, input_shape, 3))

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    stage1, num_filters_in = stage(x, num_filters_in, stage=0, num_res_blocks=2)
    stage2, num_filters_in = stage(stage1, num_filters_in, stage=1, num_res_blocks=2)
    stage3, num_filters_in = stage(stage2, num_filters_in, stage=2, num_res_blocks=2)
    stage4, num_filters_in = stage(stage3, num_filters_in, stage=3, num_res_blocks=2)
    stage5, num_filters_in = stage(stage4, num_filters_in, stage=4, num_res_blocks=2)
    stage6, num_filters_in = stage(stage5, num_filters_in, stage=4, num_res_blocks=2)

    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(stage6)
    x = Activation('relu')(x)

    # expansive path
    u7 = Conv2DTranspose(dec_filters * 32, (3, 3), strides=(2, 2), padding='same')(x)
    u7 = concatenate([u7, stage5])
    u7 = Dropout(dropout)(u7, training=training)
    c8 = conv2d_block(u7, n_filters=dec_filters * 32, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(dec_filters * 16, (3, 3), strides=(2, 2), padding='same')(c8)
    u8 = concatenate([u8, stage4])
    u8 = Dropout(dropout)(u8, training=training)
    c9 = conv2d_block(u8, n_filters=dec_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(dec_filters * 8, (3, 3), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, stage3])
    u9 = Dropout(dropout)(u9, training=training)
    c10 = conv2d_block(u9, n_filters=dec_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u10 = Conv2DTranspose(dec_filters * 4, (3, 3), strides=(2, 2), padding='same')(c10)
    u10 = concatenate([u10, stage2])
    u10 = Dropout(dropout)(u10, training=training)
    c11 = conv2d_block(u10, n_filters=dec_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u11 = Conv2DTranspose(dec_filters * 2, (3, 3), strides=(2, 2), padding='same')(c11)
    u11 = concatenate([u11, stage1])
    u11 = Dropout(dropout)(u11, training=training)
    c12 = conv2d_block(u11, n_filters=dec_filters * 2, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='linear')(c12)

    model = Model(inputs=inputs, outputs=[outputs])
    return model
