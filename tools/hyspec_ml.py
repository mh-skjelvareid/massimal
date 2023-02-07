import numpy as np
from sklearn.base import BaseEstimator

from skimage.morphology import binary_closing
from skimage.morphology import disk

import hyspec_io
import cv2
import math

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
class WavelengthSelector(BaseEstimator):
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
        nonzero_mask_holes_filled = binary_closing(~zero_mask,footprint=disk(radius=3))
        inpaint_mask = zero_mask & nonzero_mask_holes_filled

        # Inpaint zero pixels inside image
        y_pred_im = cv2.inpaint(np.ubyte(y_pred_im),np.ubyte(inpaint_mask),inpaintRadius=3,flags=cv2.INPAINT_NS)

    return y_pred_im

