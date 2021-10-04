# preprocess

# Imports
import cv2
import numpy as np

# detect_saturated
def detect_saturated(image,sat_val=2**12-1,src_ax=2):
    """ Detect saturated pixels in single- or multiband image

    # Usage:
    mask = detect_saturated(image,...)

    # Required arguments:
    image:  2D or 3D image array. Multiband images are assumed to have bands
            stacked along 3rd axis (axis = 2).

    # Optional arguments:
    sat_val: Value of saturated pixels. Default: 4095 (max. for 12 bit)
    src_ax:  Number for axis that is searched along. Default. 2 (3rd axis)

    # Returns:
    mask:   boolean array, True if any of the image bands had values at or
            above sat_val.
    """

    mask = np.any( np.atleast_3d(image) >= sat_val, axis = src_ax)

    return mask


# inpaint_masked
def inpaint_masked(in_im, mask, inpaint_radius=3, inpaint_alg = 'ns'):
    """ Inpaint masked pixels in all bands of multiband image

    # Usage:
    out_im = detect_saturated(in_im,mask,...)

    # Required arguments:
    in_im:  2D or 3D image array. Multiband images are assumed to have bands
            stacked along 3rd axis (axis = 2).
    mask:   2D array with first 2 dimensions matching those of image. Non-zero
            (True) elements mark pixels to be inpainted.

    # Optional arguments:
    inpaint_radius: "Radius of a circular neighborhood of each point inpainted
                    that is considered by the algorithm." (from OpenCV doc.)
                    Default: 3
    inpaint_alg:    Either 'ns' (default) or 'telea'. Correspond to two
                    algorithms implemented in OpenCV.

    # Returns:
    out_im: Output image with masked pixels inpainted.

    # Notes:
    This is a wrapper function for the inpainting algorithms in OpenCV. These
    only have support for 1- or 3-band images. The main contribution of the
    function is to over iterate an arbitrary number of bands and apply
    inpainting to each separately. Processing time is highly dependent on the
    number of pixels to be inpainted.

    The OpenCV library (CPU only) can be installed with
        pip install opencv-contrib-python

    Inpainting is computationally intensive and can take minutes (maybe even
    hours) for a hyperspectral image. Installing OpenCV with GPU support will
    (possibly) speed the function up considerably. See OpenCV docs for details.
    """

    # Convert image to float with single precision (32-bit), ensure 3D.
    # Note: .copy() needed to avoid modifying input image
    out_im = np.atleast_3d(np.single(in_im.copy()))

    # Convert mask to 8-bit unsigned array
    mask = np.ubyte(mask)

    # Assuming that image and mask have matching width and height
    assert mask.shape[0:2] == in_im.shape[0:2]

    # Convert to OpenCV inpaint algorithm flags
    if inpaint_alg.lower() == 'ns':
        alg_flag = cv2.INPAINT_NS
    elif inpaint_alg.lower() == 'telea':
        alg_flag = cv2.INPAINT_TELEA
    else:
        raise ValueError("inpaint_alg must be \'ns\' or \'telea\'")

    # Loop over each image band and apply inpainting
    for ii in range(out_im.shape[2]):
        out_im[:,:,ii] = cv2.inpaint( out_im[:,:,ii],mask,inpaint_radius,alg_flag)

    # Return
    return out_im
