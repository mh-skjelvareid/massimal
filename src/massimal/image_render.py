#%% Imports
from skimage import exposure
import numpy as np

#%% image_percentile_stretch
def percentile_stretch(image,percentiles=(2,98),separate_bands=True, ignore_zeros=True):
    """ Image stretch based on image intensity percentiles

    # Usage:
    im_rescaled = percentile_stretch(image,percentiles,...)

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
    ignore_zeros:   Default: True
                    If true, the calculation of percentiles does not include
                    pixels that are equal to zero in every band.

    Returns:
    im_rescaled:    Image linearly "stretched" between percentile values.

    The function uses skimage.exposure.rescale_intensity() for rescaling.
    See https://scikit-image.org/docs/stable/api/skimage.exposure.html
    """

    # Preallocate output array
    im_rescaled = np.zeros_like(image)
    
    # Determine output range / dtype
    if np.issubdtype(image.dtype,np.integer):
        out_range = 'dtype'
    else:
        out_range = (0,1)  # Float

    # Create mask indicating non-zero and non-NaN pixels
    if ignore_zeros:
        mask = ~np.all(image==0,axis=2)
    else:
        mask = np.ones(image.shape[:-1],dtype=bool)
    #mask &= ~np.any(np.isnan(image),axis=2)

    # Case: Stretch bands separately
    if separate_bands:
        assert image.ndim == 3      # Assuming 3 dimensions
        # for ii in range(image.shape[2]):
        for ii,image_band in enumerate(np.moveaxis(image,2,0)):
            p_low,p_high = np.percentile(image_band[mask], percentiles)
            im_rescaled[:,:,ii] = exposure.rescale_intensity(image_band, in_range=(p_low,p_high),out_range=out_range)

    # Case: Stretch whole image based on "global" percentiles
    else:
        p_low,p_high = np.percentile(image[mask], percentiles)
        im_rescaled = exposure.rescale_intensity( image, in_range=(p_low,p_high),out_range=out_range)

    return im_rescaled


def absolute_stretch(image,limits):
    """ Image stretch based on absolute limits (not data-dependent)

    # Usage:
    im_rescaled = absolute_stretch(image,limits,...)

    # Required arguments:
    image:          2D or 3D numpy array, image bands stacked along 3rd
                    dimension
    limits:         Numpy array indicating upper and lower limits
                    To apply a "global" stretch (same limits for each band),
                    limits should be a 2-element array: [low,high]
                    To apply a individual stretch to each band,
                    limits should be a [N_bands,2] size array

    # Optional arguments:
    ignore_zeros:   Default: True
                    If true, the calculation of percentiles does not include
                    pixels that are equal to zero in every band.

    Returns:
        im_rescaled:    Image linearly "stretched" between percentile values.

    The function uses skimage.exposure.rescale_intensity() for rescaling.
    See https://scikit-image.org/docs/stable/api/skimage.exposure.html
    """

    # Preallocate output array
    im_rescaled = np.zeros(image.shape)
    
    # Determine output range / dtype
    if np.issubdtype(image.dtype,np.integer):
        out_range = 'dtype'
    else:
        out_range = (0,1)  # Float

    # Case: Stretch bands separately
    if limits.size > 2:
        assert image.ndim == 3      # Assuming 3 dimensions
        assert limits.shape[0] == image.shape[2]

        for ii,image_band in enumerate(np.moveaxis(image,2,0)):
            im_rescaled[:,:,ii] = exposure.rescale_intensity(image_band, in_range=(limits[ii,0],limits[ii,1]),out_range=out_range)

    # Case: Stretch whole image based on "global" limits
    else:
        im_rescaled = exposure.rescale_intensity(image, in_range=(limits[0],limits[1]),out_range=out_range)

    return im_rescaled
