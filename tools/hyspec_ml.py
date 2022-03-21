import numpy as np

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
