from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate


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


def get_deep_unet(params):
    input_img = Input((params.img_shape, params.img_shape, 3), name='img')

    # contracting path
    c1 = conv2d_block(input_img, n_filters=params.n_filters * 1, kernel_size=3, batchnorm=params.batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(params.dropout)(p1, training=params.is_training)

    c2 = conv2d_block(p1, n_filters=params.n_filters * 2, kernel_size=3, batchnorm=params.batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(params.dropout)(p2, training=params.is_training)

    c3 = conv2d_block(p2, n_filters=params.n_filters * 4, kernel_size=3, batchnorm=params.batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(params.dropout)(p3, training=params.is_training)

    c4 = conv2d_block(p3, n_filters=params.n_filters * 8, kernel_size=3, batchnorm=params.batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(params.dropout)(p4, training=params.is_training)

    c5 = conv2d_block(p4, n_filters=params.n_filters * 8, kernel_size=3, batchnorm=params.batchnorm)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)
    p5 = Dropout(params.dropout)(p5, training=params.is_training)

    c6 = conv2d_block(p5, n_filters=params.n_filters * 16, kernel_size=3, batchnorm=params.batchnorm)

    # expansive path
    u8 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c6)
    u8 = concatenate([u8, c5])
    u8 = Dropout(params.dropout)(u8, training=params.is_training)
    c9 = conv2d_block(u8, n_filters=params.n_filters * 8, kernel_size=3, batchnorm=params.batchnorm)

    u9 = Conv2DTranspose(params.n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c9)
    u9 = concatenate([u9, c4])
    u9 = Dropout(params.dropout)(u9, training=params.is_training)
    c10 = conv2d_block(u9, n_filters=params.n_filters * 8, kernel_size=3, batchnorm=params.batchnorm)

    u10 = Conv2DTranspose(params.n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c10)
    u10 = concatenate([u10, c3])
    u10 = Dropout(params.dropout)(u10, training=params.is_training)
    c11 = conv2d_block(u10, n_filters=params.n_filters * 4, kernel_size=3, batchnorm=params.batchnorm)

    u11 = Conv2DTranspose(params.n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c11)
    u11 = concatenate([u11, c2])
    u11 = Dropout(params.dropout)(u11, training=params.is_training)
    c12 = conv2d_block(u11, n_filters=params.n_filters * 2, kernel_size=3, batchnorm=params.batchnorm)

    u12 = Conv2DTranspose(params.n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c12)
    u12 = concatenate([u12, c1])
    u12 = Dropout(params.dropout)(u12, training=params.is_training)
    c13 = conv2d_block(u12, n_filters=params.n_filters * 1, kernel_size=3, batchnorm=params.batchnorm)

    prediction = Conv2D(1, (1, 1), activation="sigmoid")(c13)

    outputs = prediction

    model = Model(inputs=input_img, outputs=[outputs])
    return model