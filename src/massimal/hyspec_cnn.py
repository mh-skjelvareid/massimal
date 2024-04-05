import numpy as np
import math
import tensorflow as tf
from collections.abc import Iterable

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



# Function for extracting tiles
def labeled_image_to_tensor_tiles(image,labels,tile_shape,
                                  tile_strides=None,padding='SAME',
                                  min_labeled_fraction = 0.05):
    """ Split image and label mask into smaller tiles
    
    # Usage:
    (image_tiles,label_tiles) = ...
    
    # Input arguments:
    image:        3D numpy array with dimensions (rows, columns, channels)
    labels:       2D numpy array with dimensions (rows,columns)
    tile_shape:   Tuple of integers, (tile_rows, tile_cols)
    
    # Keyword arguments
    tile_strides: Tuple of integers, (row_stride, col_stride)
                  If None, set equal to tile_shape (no overlap between tiles)
    padding:      'VALID' or 'SAME' (see tensorflow.image.extract_patches)
                  Default: 'SAME'
    min_labeled_fraction:   Use this to filter out tiles with zero or low
                            number of labeled pixels. Set to zero to include all 
                            pixels.
    """
    
    if tile_strides is None: tile_strides = tile_shape
    
    image_tensor = tf.reshape(tf.convert_to_tensor(image),(1,)+image.shape)
    label_tensor = tf.reshape(tf.convert_to_tensor(labels),(1,)+labels.shape + (1,))
    
    sizes = [1,*tile_shape,1]
    strides = [1,*tile_strides,1]
    rates = [1,1,1,1]
    
    image_tiles = tf.image.extract_patches(image_tensor, sizes, strides, rates, padding=padding)
    image_tiles = tf.reshape(image_tiles,[-1,*tile_shape,image.shape[-1]])
    label_tiles = tf.image.extract_patches(label_tensor, sizes, strides, rates, padding=padding)
    label_tiles = tf.reshape(label_tiles,[-1,*tile_shape])
    
    # Filter out tiles with zero or few annotated pixels (optional)
    if min_labeled_fraction > 0:
        labeled_tiles_mask = np.array(
            [(np.count_nonzero(tile)/np.size(tile))>min_labeled_fraction for tile in label_tiles])
        image_tiles = tf.boolean_mask(image_tiles,labeled_tiles_mask)
        label_tiles = tf.boolean_mask(label_tiles,labeled_tiles_mask)
        
    return image_tiles, label_tiles



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
                strides=resampling_factor, 
                padding='same',                          
                kernel_initializer=initializer, 
                use_bias=not(apply_batchnorm)))   
    else:
        resamp_layer.add(
            tf.keras.layers.Conv2DTranspose(
                filter_channels, 
                kernel_size, 
                strides=resampling_factor, 
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

    
    
def unet(input_channels, output_channels, first_layer_channels, depth, 
         model_name=None, flip_aug=True, trans_aug=False, 
         apply_batchnorm = True, apply_dropout = False):
    """ Simple encoder-decoder U-Net architecture
    
    # Arguments:
    input_channels:         Number of channels in input image
    output_channels:        Number of classes (including background) to segment between
    first_layer_channels:   Number of channels in first downsampling layer
                            Each consecutive downsampling layer doubles the number of channels
                            In upsampling, each layer halves the number of channels
    depth:                  Number of resampling steps to perform.
                            Example: If depth = 3, the original image is downsampled to 
                            resolutions 1/2, 1/4 and 1/8 of the original resolution, and then
                            upsampled to the original resolution via the same steps.
                            The total number of down- and upsampling layers is thus
                            equal to 2*depth (6 for the example above).
    
    # Keyword arguments:
    model_name:               Name of model
    flip_aug:           If true, a RandomFlip augmentation layer is included
                        before the first downsampling layer
    trans_aug:          If true, a RandomTranslation augmentation layer with 
                        height and width factor of 20% is included
                        before the first downsampling layer
    apply_batchnorm:    If (boolean) scalar, indicate whether to use batch normalization
                        in all downsampling / upsampling layers
                        If tuple of booleans (length equal to total number of 
                        downsampling / upsampling layers), indicate use of batch noarmalization
                        for each layer
    apply_dropout:      If (boolean) scalar, indicate whether to use dropout (rate 0.5)
                        in all downsampling / upsampling layers.
                        If tuple of booleans (length equal to total number of 
                        downsampling / upsampling layers), indicate use of dropout
                        for each layer
                        
    # Outputs:
    model:              Keras U-Net model
    
    # Notes:
    - Based on TF tutorial: https://www.tensorflow.org/tutorials/images/segmentation

    """
    resamp_kernel_size = 4

    # Create vectors for batchnorm / dropout booleans if scalar
    if not isinstance(apply_batchnorm,Iterable):
        apply_batchnorm = [apply_batchnorm for _ in range(depth*2)]

    if not isinstance(apply_dropout,Iterable):
        apply_dropout = [apply_dropout for _ in range(depth*2)]


    # Define input
    inputs = tf.keras.layers.Input(shape=[None, None, input_channels],name='input_image')   # Using None to signal variable image width and height (Ny,Nx,3)
    x = inputs    # x used as temparary variable for data flowing between layers

    # Add augmentation layer(s)
    if flip_aug or trans_aug:
        aug_layer = tf.keras.Sequential(name='augmentation')
        if flip_aug:
            aug_layer.add(tf.keras.layers.RandomFlip())
        if trans_aug:
            aug_layer.add(tf.keras.layers.RandomTranslation(height_factor=0.2,width_factor=0.2))
        x = aug_layer(x)

    # Add initial convolution layer with same resolution as input image
    x = tf.keras.layers.Conv2D(first_layer_channels,kernel_size=3,padding='same', name = 'initial_convolution',activation='relu')(x)

    # Define downsampling layers
    down_stack = []
    nchannels_downsamp = [first_layer_channels*(2**(i+1)) for i in range(depth)]
    names_downsamp = [f'downsamp_res_1/{(2**(i+1))}' for i in range(depth)]  
    for channels, name, batchnorm, dropout in zip(nchannels_downsamp,names_downsamp,apply_batchnorm[0:depth],apply_dropout[0:depth]):
        down_stack.append(resampling_layer('downsample',
                                           channels,
                                           resamp_kernel_size,
                                           name = name,
                                           apply_batchnorm=batchnorm,
                                           apply_dropout=dropout))

    # Define upsampling layers
    up_stack = []
    nchannels_upsamp = [first_layer_channels*(2**(depth-1))] + [first_layer_channels*(2**i) for i in range(depth-1,0,-1)]
    names_upsamp = [f'upsamp_res_1/{2**i}' for i in range(depth-1,-1,-1)]   
    for channels, name, batchnorm, dropout in zip(nchannels_upsamp,names_upsamp,apply_batchnorm[depth:], apply_dropout[depth:]):
        up_stack.append(resampling_layer('upsample',
                                         channels,
                                         resamp_kernel_size,
                                         name = name,
                                         apply_batchnorm=batchnorm,
                                         apply_dropout=dropout))    

    # Downsampling through the model
    skips = [x]                   # Add output from first layer (before downsampling) to skips list
    for down in down_stack:
        x = down(x)               # Run input x through layer, then set x equal to output
        skips.append(x)           # Add layer output to skips list

    skips = reversed(skips[:-1])  # Reverse list, and don't include skip for last layer ("bottom of U") 

    # Upsampling and establishing the skip connections
    names_skip = [f'skipconnection_res_1/{2**i}' for i in range(depth-1,-1,-1)]   
    for up, skip, skipname in zip(up_stack, skips, names_skip):
        x = up(x)                                     # Run input x through layer, then set x to output
        x = tf.keras.layers.Concatenate(name=skipname)([x, skip])  # Stack layer output together with skip connection 

    # Final layer
    last = tf.keras.layers.Conv2D(output_channels, 
                                  kernel_size = 3,
                                  padding='same',
                                  activation='softmax',
                                  name='classification')    
    x = last(x)

    model =  tf.keras.Model(inputs=inputs, outputs=x,name=model_name)

    return model


def add_background_zero_weight(image, labels):    
    """ Add weight "image" with zero weight for background
    
    # Typical usage: 
    dataset_with_weights = dataset.map(add_background_zero_weight)
    
    """
    label_mask = tf.greater(labels,0)
    zeros = tf.zeros_like(labels,dtype=tf.float32)
    ones = tf.ones_like(labels,dtype=tf.float32)
    
    # "Multiplex" using label mask, ones for annotated pixels, zeros for background
    sample_weights = tf.where(label_mask, ones, zeros)  # 

    return image, labels, sample_weights


def unet_classify_single_image(unet,image):
    """ Classify single image using UNet
    
    # Arguments:
    unet:     Trained Unet model (Keras)
    image:    Single image (3D Numpy array)
    
    # Returns
    labels:     2D image with integer class labels, found by using
                np.argmax() on unet output (which has one channel per class)
    """
    
    # Get activations by running predict(), insert extra dimension for 1-element batch
    activations = np.squeeze(unet.predict(np.expand_dims(image,axis=0)))
    labels = np.argmax(activations,axis=2)
    return labels


def unet_classify_image_batch(unet,batch):
    """ Classify image batch using UNet
    
    # Arguments:
    unet:     Trained Unet model (Keras)
    batch:    Batch of images (4D NumPy array)
    
    # Returns
    labels:     3D array with integer class labels, found by using
                np.argmax() on unet output (which has one channel per class)
    """
    
    # Get activations by running predict(), use argmax to find class label
    activations = unet.predict(batch)
    labels = np.argmax(activations,axis=3)
    return labels