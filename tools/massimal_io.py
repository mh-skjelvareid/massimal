#%% Imports
import spectral
import numpy as np

#%%
def load_envi_image(header_filename,image_filename=None):
    """ Function for "full" tile augmentation (all combinations)

    # Usage:
        (im_cube,wl,rgb_ind,metadata) = load_envi_image(header_filename,...)

    # Required arguments:
        header_filename: Path to ENVI file header.

    # Optional arguments:
        image_filename: Path to ENVI data file, useful if data file is not found
                        automatically

    Returns:
        im_cube:    Full image
        wl:         Wavelength vector
        rgb_ind:    3-element tuple with indices to default RGB bands,
                    [640,550,460] nm
        metadata:   Image metadata (dictionary). Can be used as input to
                    spectral.io.envi.save_image()

    The function uses spectral.io.envi.open() to read the file, and has the same
    input arguments.
    """

    # Load image
    im_handle = spectral.io.envi.open(header_filename,image_filename)
    im_cube = np.array(im_handle.load())

    # Read wavelengths
    wl = np.array([float(i) for i in im_handle.metadata['wavelength']])

    # Define default wavelengths for RGB display
    rbg_default = (640, 550, 460)

    # Get indices for standard RGB render
    rgb_ind = tuple((np.abs(wl - value)).argmin() for value in rbg_default)

    # Returns
    return (im_cube,wl,rgb_ind,im_handle.metadata)
