# hyspec_stats

# Imports
import numpy as np
from numpy.random import default_rng

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
    n_samp = np.int64(frac*X.shape[0])

    # Create random number generator
    rng = default_rng()
    samp = rng.choice(image[mask],size=n_samp,axis=0,replace=replace)

    return samp
