#%% Imports
from skimage import exposure
import numpy as np

#%% image_percentile_stretch
def percentile_stretch(image,percentiles=(2,98),separate_bands=True):
    """ Image stretch based on image intensity percentiles

    # Usage:
    im_rescaled = image_percentile_stretch(image,percentiles,...)

    # Required arguments:
    image:          2D or 3D numpy array, image bands stacked along 3rd
                    dimension

    # Optional arguments:
    percentiles:    2-element tuple of percentiles, (p_low,p_high).
                    Default: (2,98)
                    Percentiles are assumed to be in the [0,100] range.
                    Image values at or below p_low are shown as black,
                    values at or above p_high are shown as white.
    separate_bands: Default: True.
                    Perform the stretch separately for each band.
                    If false, the percentiles are calculated "globally"
                    across the whole image, and the same "stretch" is
                    applied to each band.

    Returns:
        im_rescaled:    Image linearly "stretched" between percentile values.

    The function uses skimage.exposure.rescale_intensity() for rescaling.
    See https://scikit-image.org/docs/stable/api/skimage.exposure.html
    """

    im_rescaled = np.zeros(image.shape)

    if separate_bands:
        assert image.ndim == 3      # Assuming 3 dimensions
        for ii in range(image.shape[2]):
            p_low,p_high = np.percentile(image[:,:,ii], percentiles)
            im_rescaled[:,:,ii] = exposure.rescale_intensity(image[:,:,ii], in_range=(p_low,p_high))
    else:
        p_low,p_high = np.percentile(image, percentiles)
        im_rescaled = exposure.rescale_intensity( image, in_range=(p_low,p_high))

    return im_rescaled
