import numpy as np
import math
import tensorflow as tf



def pad_image_to_multiple(image,multiple):
    """ Zero-pad image spatially to multiple of given number 
    
    # Input arguments
    image:        2D / 3D numpy array 
    multiple:     Integer
    
    # Example:
    image = np.ones((2,5))
    padded_image = pad_image_to_multiple(image,4)
    
    image is a 2x5 matrix of ones:
        [[1. 1. 1. 1. 1.]
         [1. 1. 1. 1. 1.]]
    padded_image is a 4x8 matrix padded with zeros, with image in upper left corner:
        [[1. 1. 1. 1. 1. 0. 0. 0.]
         [1. 1. 1. 1. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 0.]]
    """
    nrows_pad = math.ceil(image.shape[0]/multiple)*multiple
    ncols_pad = math.ceil(image.shape[1]/multiple)*multiple
    if image.ndim == 2:
        image_padded = np.zeros_like(image,shape=[nrows_pad, ncols_pad])
        image_padded[0:image.shape[0], 0:image.shape[1]] = image        
    else:
        image_padded = np.zeros_like(image,shape=[nrows_pad, ncols_pad, image.shape[-1]])
        image_padded[0:image.shape[0], 0:image.shape[1],:] = image
    return image_padded





def resampling_layer(resampling_type,
            filter_channels, 
            kernel_size, 
            resampling_factor = 2,
            name=None,
            initializer_mean = 0.0,
            initializer_std = 0.02,
            apply_batchnorm = True,
            apply_dropout = False,
            dropout_rate = 0.5):
    
    """ Spatial resampling 2D convolutional layer
    
    # Input parameters:
    resampling_type:    'downsample' (convolution) or 'upsample' (transpose convolution)
    filter_channels:    Number of filters / "depth" of output
                        For images, this corresponds to number of color / wavelength channels
    kernel_size:        Spatial size of convolutional kernel
                        For images, if kernel_size = 3, each filter processes a 3x3 pixel neighborhood
                        
    # Notes
    - Based on TF example pix2pix: https://www.tensorflow.org/tutorials/generative/pix2pix
    """
    # Validate resampling layer type
    if resampling_type not in ['downsample','upsample']:
        raise ValueError(f"{resampling_type} is not a valid resampling type.")
    
    # Create kernel initializer for normally distributed random numbers
    initializer = tf.random_normal_initializer(
        mean=initializer_mean, stddev=initializer_std)                    
    
    # Initialize as sequential (stack of layers)
    resamp_layer = tf.keras.Sequential(name=name)
    
    # Add 2D convolutional layer
    if resampling_type == 'downsample':
        resamp_layer.add(
            tf.keras.layers.Conv2D(
                filter_channels, 
                kernel_size, 
                strides=downsampling_factor, 
                padding='same',                          
                kernel_initializer=initializer, 
                use_bias=not(apply_batchnorm)))   
    else:
        resamp_layer.add(
            tf.keras.layers.Conv2DTranspose(
                filter_channels, 
                kernel_size, 
                strides=upsampling_factor, 
                padding='same',
                kernel_initializer=initializer,
                use_bias=not(apply_batchnorm)))

    # Add (optional) batch normalization layer
    if apply_batchnorm:
        resamp_layer.add(tf.keras.layers.BatchNormalization())                

    # Add (optional) dropout layer
    if apply_dropout:
        resamp_layer.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Add activation layer
    if resampling_type == 'downsample':
        resamp_layer.add(tf.keras.layers.LeakyReLU()) 
    else:
        resamp_layer.add(tf.keras.layers.ReLU()) 

    return resamp_layer