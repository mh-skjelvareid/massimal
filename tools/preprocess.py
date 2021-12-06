# preprocess

# Imports
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
import hyspec_io
import misc
import os
import spectral


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
    function is to iterate over an arbitrary number of bands and apply
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
        print('Inpaiting band ' + str(ii) + ' of ' + str(out_im.shape[2]),end="\r")
        out_im[:,:,ii] = cv2.inpaint( out_im[:,:,ii],mask,inpaint_radius,alg_flag)

    # Return
    return out_im


def remove_glint_flatspec(image,wl,nir_band=(780,840)):
    """ Remove sun/sky glint assuming flat glint spectrum

    # Usage:
    image_noglint = remove_glint_flatspec(image,wl,...)

    # Required arguments:
    in_im:  3D numpy array with hyperspectral image, wavelengths along 3rd dim.
    wl:     1D array of wavelenghs (numeric)

    # Optional arguments:
    nir_band:   2-element tuple with upper and lower limit of NIR band
                The average value of the NIR band is subtracted from the
                original image.

    # Returns:
    image_noglint:  Hyperspectral image with estimated glint subtracted.

    """

    # Calculated wavelength indices of NIR band
    nir_ind = (wl > nir_band[0]) & (wl < nir_band[1])

    # Subtract average NIR value from all of image.
    mean_nir = np.mean(image[:,:,nir_ind], axis=2, keepdims=True)
    image_noglint = image - mean_nir

    return image_noglint



class HedleySunGlint:
    """ Hedley sun glint removal regression model

    # Usage:
    GlintMod = HedleySunGlint()

    # Notes:
    This implementation is based on the following paper: Hedley, J. D.,
    Harborne, A. R., & Mumby, P. J. (2005). Technical note: Simple and robust
    removal of sun glint for mapping shallow‐water benthos. International
    Journal of Remote Sensing, 26(10), 2107–2112.
    https://doi.org/10.1080/01431160500034086

    """

    def __init__(self, vis_band = (350,750), nir_band = (780,840)):
        """ Hedley sun glint removal: Model instantiation

        # Optional arguments:
        vis_band:   2-element tuple with upper and lower wavelength of VIS band (nm).
        nir_band:   2-element tuple with upper and lower limit of NIR band.
                    The NIR band is used to estimate the amount of sun glint present
                    in the VIS band.
        """
        self.vis_band = vis_band
        self.nir_band = nir_band


    def fit(self,spec,wl):
        """ Fit a linear regression model connecting NIR and VIS data

        # Required arguments:
        spec:   2D numpy array with hyperspectral image, wavelengths along 2nd dim.
                The cube should contain data from a homogenous bottom (e.g. deep sea)
                and a representative variation of sun/sky glint.
        wl:     1D array of wavelenghs (numeric). Must match the size of the 2nd
                dimension of spec.

        """

        # Calculate VIS and NIR indices
        self.wl = wl
        self.vis_ind = (wl >= self.vis_band[0]) & (wl <= self.vis_band[1])
        self.nir_ind = (wl >= self.nir_band[0]) & (wl <= self.nir_band[1])

        # Calculate mean NIR value in band
        nir = np.mean(spec[:,self.nir_ind], axis=1, keepdims=True)

        # Extract VIS data
        vis = spec[:,self.vis_ind]

        # Fit a linear regression model
        reg = LinearRegression().fit(nir, vis)
        self.b = np.reshape(reg.coef_,(1,-1))   # Row vector, for mat. mult.

        # Estimate minimum NIR value as 2nd percentile (more robust than strict minimum)
        self.min_nir = np.percentile(nir,2)


    def remove_glint(self,data):
        """ Remove sun glint by applying previously trained model

        # Required arguments:
        data:   2D or 3D spectral data, wavelengths along last dim.
                The wavelengths must match those of the training data unsed
                to fit the model.

        # Returns:
        vis:    2D or 3D spectral data, limited to visible range, with sun
                glint removed.
        """

        # Shape into 2D array, save original shape for later
        input_shape = data.shape
        data = np.reshape(data,(-1,data.shape[-1]))

        # Extract VIS and NIR bands
        vis = data[:,self.vis_ind]
        nir = np.mean(data[:,self.nir_ind], axis=1, keepdims=True)

        # Offset NIR, taking into account "ambient" (minimum) NIR
        # NOTE: This may be unnecessary for UAV data (minimal atmosphere effects)?
        nir = nir - self.min_nir
        nir[nir < 0] = 0;

        # Estimate sun glint in VIS range and subtract it
        vis = vis - nir @ self.b    # Matrix mult. with slope b for each VIS band

        # Reshape data to fit original dimensions
        output_shape = input_shape[:-1] + (vis.shape[-1],)
        vis = np.reshape(vis,output_shape)

        # Return
        return vis
