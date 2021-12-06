#%% Imports
import spectral
import numpy as np

#%% load_envi_image
def load_envi_image(header_filename,image_filename=None):
    """ Load image in ENVI format, with wavelenghts and RGB indices

    # Usage:
    (im_cube,wl,rgb_ind,metadata) = load_envi_image(header_filename,...)

    # Required arguments:
    header_filename: Path to ENVI file header.

    # Optional arguments:
    image_filename: Path to ENVI data file, useful if data file is not found
                    automatically.

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


def save_envi_image(header_filename,image,metadata,dtype = 'uint16', **kwargs):
    """ Save ENVI file with parameters compatible with Spectronon

    # Usage:
    save_envi_image(header_filename,image,metadata)

    # Required arguments:
    header_filename:    Path to header file.
                        Data file will be saved in the same location and with
                        the same name, but without the '.hdr' extension
    image:              Numpy array with hyperspectral image
    metadata:           Dict containing (updated) image metadata.
                        See load_envi_image()

    Optional arguments:
    dtype:      Data type for ENVI file. Follows numpy naming convention.
                Typically 'uint16' or 'single' (32-bit float)
    **kwargs:   Additional keyword arguments passed on to
                spectral.envi.save_image()
    """

    # Save file
    spectral.envi.save_image(header_filename,image,
        dtype=dtype, metadata=metadata, force=True, ext=None, **kwargs)
