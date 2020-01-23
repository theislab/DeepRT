import tensorflow as tf
from params import *
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D( filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal",
                padding = "same" )( input_tensor )
    if batchnorm:
        x = BatchNormalization()( x )
    x = Activation( "relu" )( x )
    # second layer
    x = Conv2D( filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal",
                padding = "same" )( x )
    if batchnorm:
        x = BatchNormalization()( x )
    x = Activation( "relu" )( x )
    return x


def get_unet(input_img, n_filters=16, dropout=1.0, batchnorm=True, training=True):
    # contracting path
    c1 = conv2d_block( input_img, n_filters = n_filters * 1, kernel_size = 3, batchnorm = batchnorm )
    p1 = MaxPooling2D( (2, 2) )( c1 )
    p1 = Dropout( dropout )( p1, training = training )

    c2 = conv2d_block( p1, n_filters = n_filters * 2, kernel_size = 3, batchnorm = batchnorm )
    p2 = MaxPooling2D( (2, 2) )( c2 )
    p2 = Dropout( dropout )( p2, training = training )

    c3 = conv2d_block( p2, n_filters = n_filters * 4, kernel_size = 3, batchnorm = batchnorm )
    p3 = MaxPooling2D( (2, 2) )( c3 )
    p3 = Dropout( dropout )( p3, training = training )

    c4 = conv2d_block( p3, n_filters = n_filters * 8, kernel_size = 3, batchnorm = batchnorm )
    p4 = MaxPooling2D( pool_size = (2, 2) )( c4 )
    p4 = Dropout( dropout )( p4, training = training )

    c5 = conv2d_block( p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm )
    p5 = MaxPooling2D( pool_size = (2, 2) )( c5 )
    p5 = Dropout( dropout )( p5, training = training )

    c6 = conv2d_block( p5, n_filters = n_filters * 32, kernel_size = 3, batchnorm = batchnorm )
    p6 = MaxPooling2D( pool_size = (2, 2) )( c6 )
    p6 = Dropout( dropout )( p6, training = training )

    c7 = conv2d_block( p6, n_filters = n_filters * 64, kernel_size = 3, batchnorm = batchnorm )

    # expansive path
    u7 = Conv2DTranspose( n_filters * 32, (3, 3), strides = (2, 2), padding = 'same' )( c7 )
    u7 = concatenate( [u7, c6] )
    u7 = Dropout( dropout )( u7, training = training )
    c8 = conv2d_block( u7, n_filters = n_filters * 32, kernel_size = 3, batchnorm = batchnorm )

    u8 = Conv2DTranspose( n_filters * 16, (3, 3), strides = (2, 2), padding = 'same' )( c8 )
    u8 = concatenate( [u8, c5] )
    u8 = Dropout( dropout )( u8, training = training )
    c9 = conv2d_block( u8, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm )

    u9 = Conv2DTranspose( n_filters * 8, (3, 3), strides = (2, 2), padding = 'same' )( c9 )
    u9 = concatenate( [u9, c4] )
    u9 = Dropout( dropout )( u9, training = training )
    c10 = conv2d_block( u9, n_filters = n_filters * 8, kernel_size = 3, batchnorm = batchnorm )

    u10 = Conv2DTranspose( n_filters * 4, (3, 3), strides = (2, 2), padding = 'same' )( c10 )
    u10 = concatenate( [u10, c3] )
    u10 = Dropout( dropout )( u10, training = training )
    c11 = conv2d_block( u10, n_filters = n_filters * 4, kernel_size = 3, batchnorm = batchnorm )

    u11 = Conv2DTranspose( n_filters * 2, (3, 3), strides = (2, 2), padding = 'same' )( c11 )
    u11 = concatenate( [u11, c2] )
    u11 = Dropout( dropout )( u11, training = training )
    c12 = conv2d_block( u11, n_filters = n_filters * 2, kernel_size = 3, batchnorm = batchnorm )

    u12 = Conv2DTranspose( n_filters * 1, (3, 3), strides = (2, 2), padding = 'same' )( c12 )
    u12 = concatenate( [u12, c1] )
    u12 = Dropout( dropout )( u12, training = training )
    c13 = conv2d_block( u12, n_filters = n_filters * 1, kernel_size = 3, batchnorm = batchnorm )

    prediction = Conv2D(1, (1, 1), activation = "sigmoid" )( c13 )

    outputs = prediction

    model = Model( inputs = input_img, outputs = [outputs] )
    return model
