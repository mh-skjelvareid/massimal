import numpy as np
import sklearn.base
import skimage.morphology
import hyspec_io
import cv2 
import math
import tensorflow as tf
import numpy.random


class ImageVectorizer:
    """ Helper class for vectorizing and de-vectorizing image data for machine learning

    # Usage:
    image_vectorizer = ImageVectorizer(nRows,nCols)

    # Notes:
    Many machine learning algorithms require the input to be a 2D "X" matrix
    with each row corresponding to an observation, and each column containing
    a feature belonging to that observation.

    Input data in the form of an image usually has 3 dimensions - two
    spatial dimensions and a third dimension corresponding to 1 or more image
    "channels" or features. In order to convert the 3D image to a 2D "X" matrix,
    the image pixels are "stacked" so that the X matrix has dimensions
    (nPixels, nChannels). Such stacking is fairly straightforward, but
    rearranging data into an image requires knowing the original image
    dimensions.

    The ImageVectorizer "remembers" the original size of the image, making it
    easier to convert data back into the original image shape.

    """

    def __init__(self,nRows,nCols):
        """ ImageVectorizer instantiation

        # Required arguments
        nRows:     Number of image rows (image.shape[0])
        nCols:     Number of image columns (image.shape[1])

        """

        self.nRows = nRows          # Number of image rows (vertical size)
        self.nCols = nCols          # Number of image columns (horizontal size)
        self.nObs  = nRows*nCols    # Number of observations (image pixels)

    def image3d_to_matrix(self,image):
        """ Convert 3D image to 2D matrix

        # Required arguments
        image:    3D image, dimensions (nRows,nCols,nChannels)

        # Returns
        matrix:   2D matrix, dimensions (nRows*nCols, nChannels)

        """
        return np.reshape(image,(self.nObs,image.shape[2]))

    def matrix_to_image3d(self,matrix):
        """ Convert 2D matrix to 3D image

        # Required arguments
        matrix:   2D matrix, dimensions (nRows*nCols, nChannels)

        # Returns
        image:    3D image, dimensions (nRows,nCols,nChannels)

        """
        return np.reshape(matrix,(self.nRows, self.nCols, matrix.shape[1]))

    def image2d_to_vector(self, image):
        """ Convert 2D image to 1D vector

        # Required arguments
        image:    2D image, dimensions (nRows,nCols)

        # Returns
        vector:   1D vector, dimensions (nRows*nCols)

        """
        return np.reshape(image, self.nObs)

    def vector_to_image2d(self,vector):
        """ Convert 1D vector to 2D image

        # Required arguments
        vector:   1D vector, dimensions (nRows*nCols)

        # Returns
        image:    2D image, dimensions (nRows,nCols)

        """
        return np.reshape(vector, (self.nRows,self.nCols))


# Define class for using only a subset of wavelengths
class WavelengthSelector(sklearn.base.BaseEstimator):
    """ Simple class for selecting a range of wavelength in a matrix

    # Usage:
    ws = WavelengthSelector(wl,wl_min,wl_max)

    # Notes:
    Useful for selecting wavelength range as part of ML pipeline

    """

    def __init__(self, wl, wl_min, wl_max):
        self.wl = wl
        self.wl_min = wl_min
        self.wl_max = wl_max
        self.wl_ind = (wl >= wl_min) & (wl <= wl_max)

    def transform(self, X):
        """ Extract subset of X matrix based on selected wavelengths

        # Required arguments:
        X:      2D matrix, samples along 1. dim, wavelengths along 2. dim

        """
        cols = X[:,self.wl_ind] # Extract columns
        return cols

    def fit(self, X, y=None):
        return self


def save_classification_as_envi(class_im, map_info, filename):
    """ Save classification image (numeric labels) as georeferenced ENVI file

    # Required arguments:
    class_im:   2D classification image, typically integers
    map_info:   Map info from imported hyperspectral image metadata
                    (metadata['map info'])
    filename:   Path to header file for ENVI file output
                File should have the extension ".hdr".

    """

    # Create dict and insert map info
    metadata = {}
    metadata['map info'] = map_info

    # Save file
    hyspec_io.save_envi_image(header_filename = filename,
                          image = class_im,
                          metadata = metadata,
                          dtype = class_im.dtype)


def apply_classifier_to_image(classifier,image,fill_zeros=True):
    """ Apply classifier to hyperspectral image

    # Required arguments:
    classifier:     Trained classifier which accepts data as X matrix with
                    samples along 1. dimension and wavelengths along 2. dimension.
    image:          3D tensor with spatial axes along 2 first dimensions and
                    wavelength along 3. dimension. Internally, the matrix is
                    reshaped into a 2D matrix with wavelength along 2. dim.

    # Optional arguments:
    fill_zeros:     Fill in zeros in the classification result using inpainting
                    (zeros can potentially appear in georeferenced images)

    """

    # Make image vectorizer
    im_vz = ImageVectorizer(image.shape[0],image.shape[1])

    # Apply classifier to whole image
    y_pred = classifier.predict(im_vz.image3d_to_matrix(image))
    y_pred_im = im_vz.vector_to_image2d(y_pred)

    # Set prediction results for "zero pixels" to zero (background)
    zero_mask = np.all(image==0,axis=2)
    y_pred_im[zero_mask] = 0

    if fill_zeros:
        # Determine which pixels to inpaint
        nonzero_mask_holes_filled = skimage.morphology.binary_closing(~zero_mask,footprint=skimage.morphology.disk(radius=3))
        inpaint_mask = zero_mask & nonzero_mask_holes_filled

        # Inpaint zero pixels inside image
        y_pred_im = cv2.inpaint(np.ubyte(y_pred_im),np.ubyte(inpaint_mask),inpaintRadius=3,flags=cv2.INPAINT_NS)

    return y_pred_im


def save_pca_model(pca_model,X_notscaled,npz_filename,n_components = 'all'):
    """ Save PCA weights and X mean and std as NumPy npz file
    
    # Arguments:
    pca_model       sklearn.decomposition.PCA model which has beed fitted to (scaled) data X_scaled
    X_notscaled     X array, mean value for each feature in X matrix (before scaling)
    npz_filename    Path to *.npz file where data will be saved
    n_components    Number of PCA components to include
    
    # Notes:
    The function will save the following arrays to the npz. file:
        - W_pca:  PCA "weights", shape (N_features, N_components)  
        - X_mean: X mean values, shape (1,N_features,)
        - X_std:  X standard deviations, shape (1,N_features)
        - explained_variance_ratio, shape (N_components)
    """
    # If n_components specified, only use n first components
    if n_components != 'all':
        W_pca = np.transpose(pca_model.components_[0:n_components,:])
        explained_variance_ratio = pca_model.explained_variance_ratio_[0:n_components]
    else:
        W_pca = np.transpose(pca_model.components_)
        explained_variance_ratio = pca_model.explained_variance_ratio_
        
    # Save as npz file
    np.savez(npz_filename,
         W_pca = W_pca,
         X_mean = np.mean(X_notscaled,axis=0),
         X_std = np.std(X_notscaled,axis=0),
         explained_variance_ratio = explained_variance_ratio)
    

def read_pca_model(npz_filename,include_explained_variance=False):
    """ Load PCA weights and X mean and std from NumPy npz file
    
    # Arguments:
    npz_filename    Path to *.npz file where data is saved
    
    # Returns:
    W_pca:    PCA "weights", shape (N_features, N_components)  
    X_mean:   X mean values, shape (1,N_features,)
    X_std:    X standard deviations, shape (1,N_features)
    explained_variance_ratio (if include_explained_variance = True)
    """
    return_list = []
    with np.load(npz_filename) as npz_files:
        return_list.append(npz_files['W_pca'])
        return_list.append(npz_files['X_mean'])
        return_list.append(npz_files['X_std'])
        if include_explained_variance:
            return_list.append(npz_files['explained_variance_ratio'])
        
    return tuple(return_list)


def pca_transform_image(image,W_pca,X_mean,X_std=None):
    """ Apply PCA transform to 3D image cube 
    
    # Arguments:
    image       NumPy array with 3 dimensions (n_rows,n_cols,n_channels)
    W_pca       PCA weight matrix with 2 dimensions (n_channels,n_components)
    X_mean      Mean value vector, to be subtracted from data ("centering")
                Length (n_channels,)
    
    # Keyword arguments:
    X_std       Standard deviation vector, to be used for scaling (z score)
                If None, no scaling is performed
                Length (n_channels)
                
    # Returns:
    image_pca   Numpy array with dimension (n_rows, n_cols, n_channels)
    
    # Notes:
    - Input pixels which are zero across all channels are set to zero in the 
    output PCA image as well.
    
    """
    # Create mask for nonzero values
    nonzero_mask = ~np.all(image==0,axis=2,keepdims=True)
    
    # Vectorize image
    im_vec = np.reshape(image,(-1,image.shape[-1]))
    
    # Subtract mean (always) and scale (optional)
    im_vec_norm = im_vec-X_mean
    if X_std is not None:
        im_vec_norm = im_vec_norm/X_std

    # PCA transform through matrix multiplication (projection to rotated coordinate system)
    im_vec_pca = im_vec_norm @ W_pca
    
    # Reshape into image, and ensure that zero-value input pixels are also zero in output
    im_pca = np.reshape(im_vec_pca,image.shape[0:2]+(im_vec_pca.shape[-1],))*nonzero_mask

    return im_pca


def kfold_generator(dataset,k):
    """ Generator for K-fold splitting into training and validation datasets
    
    # Arguments:
    dataset    Tensorflow dataset
    k          Number of folds (see https://scikit-learn.org/stable/modules/cross_validation.html)
    
    # Returns
    training_dataset      Tensorflow dataset
    validation_dataset    Tensorflow dataset
    
    # Notes:
    The generator returns k sets of training and validation datasets when iterated over.
    
    # Example use:
    dataset = tf.data.Dataset.from_tensor_slices((np.arange(9),np.arange(9)%3))
    for data,label in dataset.as_numpy_iterator():
        print(f'Data: {data}, label: {label}')
    for training_dataset, validation_dataset in kfold_generator(dataset,3):
        print('----')
        for data,label in training_dataset.as_numpy_iterator():
            print(f'Training data: {data}, label: {label}')
        for data,label in validation_dataset.as_numpy_iterator():
            print(f'Validation data: {data}, label: {label}')
    """
    n_datapoints = dataset.cardinality()
    dataset = dataset.shuffle(n_datapoints,reshuffle_each_iteration=False)
    samples_per_fold = n_datapoints//k
    for i in range(k):
        validation_dataset = dataset.skip(i*samples_per_fold).take(samples_per_fold)
        # Merge parts before/after validation dataset to create training dataset
        training_dataset = dataset.take(i*samples_per_fold)
        training_dataset = training_dataset.concatenate(dataset.skip((i+1)*samples_per_fold))
        yield (training_dataset,validation_dataset)
        
        
        
def sample_weights_balanced(y):
    """ Create sample weigths which are inversely proportional to class frequencies 
    
    # Arguments:
    y        Numpy vector with (numerical) labels
    
    # Returns:
    sample_weights  Numpy vector with same shape as y
                    Classes with a low number of samples get higher weights
                    See 'balanced' option in
                    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_sample_weight.html
    
    # Notes
    - Useful in combination with score() function for various classifiers,
    to calculate a balanced score in case on unbalanced datasets
    """
    sample_weights = np.zeros(len(y),dtype=float)
    for label in np.unique(y_val):
        label_mask = (y_val == label)
        sample_weights[label_mask] = len(y)/np.count_nonzero(label_mask)
    return sample_weights


def random_sample_image(image,frac=0.05,ignore_zeros=True,replace=False):
    """ Draw random samples from image

    # Usage:
    samp = random_sample_image(image,...)

    # Required arguments:
    image:  3D numpy array with hyperspectral image, wavelengths along
            third axis (axis=2)

    # Optional arguments:
    frac:           Number of samples expressed as a fraction of the total
                    number of samples in the image. Range: [0 - 1]
    ignore_zeros:   Do not include samples that are equal to zeros across all
                    bands.
    replace:        Whether to select samples with or without replacement.

    # returns
    samp:   2D numpy array of size NxB, with N denoting number of samples and B
            denoting number of bands.
    """

    # Create mask
    if ignore_zeros:
        mask = ~np.all(image==0,axis=2)
    else:
        mask = np.ones(image.shape[:-1],axis=2)

    # Calculate number of samples
    n_samp = np.int64(frac*np.count_nonzero(mask))

    # Create random number generator
    rng = numpy.random.default_rng()
    samp = rng.choice(image[mask],size=n_samp,axis=0,replace=replace)

    return samp