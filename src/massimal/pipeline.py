# Imports
import spectral
import zipfile
import numpy as np
from typing import Union
from pathlib import Path


def read_envi(header_path: Union[Path,str], 
              image_path: Union[Path,str,None]=None,
              write_byte_order_if_missing = True
              ) -> tuple[np.ndarray,np.ndarray,dict]:
    """ Load image in ENVI format, including wavelength vector and other metadata

    Usage:
    ------
    (image,wl,rgb_ind,metadata) = read_envi(header_path,...)

    Arguments:
    ----------
    header_path: Path | str
        Path to ENVI file header.

    Keyword arguments:
    ------------------
    image_path: Path | str
        Path to ENVI data file, useful if data file is not found
        automatically from header file name (see spectral.io.envi.open).
    write_byte_order_if_missing: bool
        Flag to indicate if the string "byte order = 0" should be written
        to the header file in case of MissingEnviHeaderParameter error
        (byte order is required by the "spectral" library, but is missing
        in some Resonon ENVI files)

    Returns:
    --------
    image: np.ndarray
        image, shape (n_lines, n_samples, n_channels) 
    wl: np.ndarray
        Wavelength vector, shape (n_channels,). None if no wavelengths listed. 
    metadata:  dict
        Image metadata (ENVI header content). 
    """

    # Open image handle
    try:
        im_handle = spectral.io.envi.open(header_path,image_path)
    except spectral.io.envi.MissingEnviHeaderParameter as e:
        print(f'Header file has missing parameter: {header_path}')
        print(e)
        byte_order_missing_str = 'Mandatory parameter "byte order" missing from header file.'
        if str(e)==byte_order_missing_str and write_byte_order_if_missing:
            print('Writing "byte order = 0" to header file and retrying')
            try:
                with open(header_path, 'a') as file:
                    file.write('byte order = 0\n')
            except OSError:
                print(f'Error writing to header file {header_path}')
            im_handle = spectral.io.envi.open(header_path,image_path)
            
    # Read wavelengths
    if 'wavelength' in im_handle.metadata:
        wl = np.array([float(i) for i in im_handle.metadata['wavelength']])
    else:
        wl = None

    # Read data from disk
    image = np.array(im_handle.load())  

    # Returns
    return (image,wl,im_handle.metadata)


def save_envi(header_path:Union[Path,str],
              image:np.ndarray,
              metadata:dict, 
              dtype:Union[np.dtype,str,None]=None) -> None:
    """ Save ENVI file with parameters compatible with Spectronon

    # Usage:
    save_envi(header_path,image,metadata)

    # Arguments:
    ------------
    header_path: Path | str
        Path to header file.
        Data file will be saved in the same location and with
        the same name, but without the '.hdr' extension.
    image: np.ndarray     
        Numpy array with hyperspectral image
    metadata:           
        Dict containing (updated) image metadata.
        See load_envi_image()

    Optional arguments:
    -------------------
    dtype:      
        Data type for ENVI file. Follows numpy naming convention.
        Typically 'uint16' or 'single' (32-bit float)
        If None, dtype = image.dtype
    """

    if dtype is None:
        dtype = image.dtype
    
    # Save file
    spectral.envi.save_image(header_path,image,
        dtype=dtype, metadata=metadata, force=True, ext=None)


def bin_image(image:np.ndarray,
              line_bin_size:int=1, 
              sample_bin_size:int=1,
              channel_bin_size:int=1,
              average:bool=True) -> np.ndarray:
    """ Bin image cube (combine neighboring pixels) 
    
    Arguments
    ---------
    image: np.ndarray
        Image formatted as 3D NumPy array, with shape
        (n_lines, n_samples, n_channels). If the original array
        is only 2D, extend it t0 3D by inserting a singleton axis.
        For example, for a "single-line image" with shape (900,600),
        use image = np.expand_dims(image,axis=0), resulting in shape
        (1,900,600).

    Keyword arguments:
    ------------------
    line_bin_size, sample_bin_size, channel_bin_size: int
        Bin size, i.e. number of neighboring pixels to merge, 
        for line, sample and channel dimensions, respectively.
    average: bool
        Whether to use averaging across neighboring pixels. 
        If false, neighboring pixels are simply summed. Note that 
        this shifts the statistical distribution of pixel values.

    References:
    -----------
    Inspired by https://stackoverflow.com/a/36102436 
    See also https://en.wikipedia.org/wiki/Pixel_binning 
    
    """
    assert image.ndim == 3
    n_lines, n_samples, n_channels = image.shape
    assert (n_lines % line_bin_size) == 0
    assert (n_samples % sample_bin_size) == 0
    assert (n_channels % channel_bin_size) == 0

    n_lines_binned = n_lines // line_bin_size
    n_samples_binned = n_samples // sample_bin_size
    n_channels_binned = n_channels // channel_bin_size

    image = image.reshape(n_lines_binned,line_bin_size,
                          n_samples_binned,sample_bin_size,
                          n_channels_binned,channel_bin_size)
    if average:
        image = np.mean(image,axis=(1,3,5))
    else:
        image = np.sum(image,axis=(1,3,5))

    return image


class RadianceCalibrationDataset:
    """ A radiance calibration dataset for Resonon hyperspectral cameras.
    
    Attributes:
    -----------
    calibration_file: Path
        Path to Imager Calibration Pack (*.icp) file
    calibration_dir: Path
        Folder into which all the data in the calibration_file
        is put (unzipped).
    gain_file_path: Path
        Path to "gain file", i.e. ENVI file with "gain spectra", 
        per-wavelength values for converting from raw (digital numbers) 
        data to radiance data (in physical units).
    dark_frame_paths: list[Path]
        List of paths to "dark frames", i.e. files with a single line / spectrum
        taken with no light incident on the sensor, taken with different gain
        and shutter values. 

    Methods:
    --------
    get_rad_conv_frame():
        Returns radiance conversion frame, shape (n_samples, n_channels)
    get_closest_dark_frame(gain,shutter)
        Returns dark frame which best matches given gain and shutter values
    
    """

    def __init__(self, 
                 calibration_file: Union[Path,str],
                 calibration_dir_name: str = 'radiance_calibration_frames'):
        """ Un-zip calibration file and create radiance calibration dataset 
        
        Arguments:
        ----------
        calibration_file: Path | str
            Path to *.icp Resonon "Imager Calibration Pack" file
        calibration_dir_name: str
            Name of subdirectory into which calibration frames are unzipped

        Raises:
        -------
        zipfile.BadZipfile:
            If given *.icp file is not a valid zip file
        
        """
        # Register image calibration "pack" (*.icp) and check that it exists
        self.calibration_file = Path(calibration_file)
        assert self.calibration_file.exists()

        # Unzip into same directory 
        self.calibration_dir = self.calibration_file.parent / calibration_dir_name
        self.calibration_dir.mkdir(exist_ok=True)
        self._unzip_calibration_file()

        # Register (single) gain curve file and multiple dark frame files
        self.gain_file_path = self.calibration_dir / 'gain.bip.hdr'
        assert self.gain_file_path.exists()
        self.dark_frame_paths = list(self.calibration_dir.glob('offset*gain*shutter.bip.hdr'))

        # Get dark frame gain and shutter info from filenames
        self._get_dark_frames_gain_shutter()

        # Sort gain/shutter values and corresponding filenames 
        self._sort_dark_frame_gains_shutters_paths()


    def _unzip_calibration_file(self,unzip_into_nonempty_dir:bool=False) -> None:
        """ Unzip *.icp file (which is a zip file) """
        if not unzip_into_nonempty_dir and any(list(self.calibration_dir.iterdir())): 
            print(f'INFO: Non-empty calibration directory {self.calibration_dir}')
            print('INFO: Skipping unzipping of calibration file, assuming unzipping already done.') 
            return
        try:
            with zipfile.ZipFile(self.calibration_file, mode='r') as zip_file:
                for filename in zip_file.namelist():
                    zip_file.extract(filename, self.calibration_dir)
        except zipfile.BadZipFile:
            print(f'File {self.calibration_file} is not a valid ZIP file.')
        except Exception as e:
            print('An unexpected error occured when extracting calibration file '
                  f'{self.calibration_file}')
            print(e)
            

    def _get_dark_frames_gain_shutter(self) -> None:
        """ Extract and save gain and shutter values for each dark frame """
        # Example dark frame pattern: 
        # offset_600bands_4095ceiling_5gain_900samples_75shutter.bip.hdr
        dark_frame_gains = []
        dark_frame_shutters = []
        for dark_frame_path in self.dark_frame_paths:
            # Strip file extensions, split on underscores, keep only gain and shutter info
            _, _, _, gain_str, _, shutter_str = dark_frame_path.name.split('.')[0].split('_')
            dark_frame_gains.append(int(gain_str[:-4]))
            dark_frame_shutters.append(int(shutter_str[:-7]))
        # Save as NumPy arrays
        self._dark_frame_gains = np.array(dark_frame_gains,dtype=float)
        self._dark_frame_shutters = np.array(dark_frame_shutters,dtype=float)


    def _sort_dark_frame_gains_shutters_paths(self) -> None:
        """ Sort gain/shutter values and corresponding file names """
        gain_shutter_path_sorted = sorted(zip(
            self._dark_frame_gains, self._dark_frame_shutters, self.dark_frame_paths))
        self._dark_frame_gains = np.array([gain for gain,_,_ in gain_shutter_path_sorted])
        self._dark_frame_shutters = np.array([shutter for _,shutter,_ in gain_shutter_path_sorted])
        self.dark_frame_paths = [path for _,_,path in gain_shutter_path_sorted]

            
    def _get_closest_dark_frame_path(self, 
                               gain: Union[int,float], 
                               shutter: Union[int,float]
                               ) -> Path:
        """ Search for dark frame with best matching gain and shutter values """
        # First search for files with closest matching gain
        candidate_gains = np.unique(self._dark_frame_gains)
        closest_gain = candidate_gains[np.argmin(abs(candidate_gains - gain))]
        
        # Then search (in subset) for single file with closest matching shutter 
        candidate_shutters = np.unique(self._dark_frame_shutters[self._dark_frame_gains == closest_gain])
        closest_shutter = self._dark_frame_shutters[np.argmin(abs(candidate_shutters - shutter))]
        
        # Return best match
        best_match_mask = ((self._dark_frame_gains==closest_gain) & 
                           (self._dark_frame_shutters==closest_shutter))
        best_match_ind = np.nonzero(best_match_mask)[0]
        assert len(best_match_ind) == 1 # There should only be a single best match
        return self.dark_frame_paths[best_match_ind[0]], closest_gain, closest_shutter
        
    
    def get_closest_dark_frame(self, 
                               gain: Union[int,float], 
                               shutter: Union[int,float]
                               ) -> tuple[np.ndarray, np.ndarray, dict, float, float]:
        """ Get dark frame which most closely matches given gain and shutter values 
        
        Arguments:
        ----------
        gain:
            Gain value used for search, typically gain value of image that should
            be converted from raw to radiance. Values follow Resonon convention 
            used in header files++ (logarithmic values, 20 log10).
        shutter:
            Shutter value used for search, typically shutter value of image that should
            be converted from raw to radiance. Values follow Resonon convention 
            used in header files++ (unit: milliseconds).

        Returns:
        --------
        frame: np.ndarray
            Single dark frame, shape (1,n_samples,n_channels)
        wl: np.ndarray 
            Vector of wavelengths for each spectral channel (nanometers)
        metadata: dict
            Metadata from ENVI header, formatted as dictionary
        """
        closest_file, closest_gain, closest_shutter = self._get_closest_dark_frame_path(
            gain=gain,shutter=shutter)
        frame, wl, metadata = read_envi(closest_file)
        return frame, wl, metadata, closest_gain, closest_shutter


    def get_rad_conv_frame(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """ Read and return radiance conversion curve ("gain" file) 
        
        Returns:
        --------
        frame: np.ndarray
            Radiance conversion frame, shape (1,n_samples,n_channels)
        wl: np.ndarray 
            Vector of wavelengths for each spectral channel (nanometers)
        metadata: dict
            Metadata from ENVI header, formatted as dictionary
        
        """
        return read_envi(self.gain_file_path)  

        
class RadianceConverter:
    """ A class for converting raw hyperspectral images from Pika L cameras to radiance.
    
    Attributes:
    -----------
    radiance_calibration_file: Path | str
        Path to Imager Calibration Pack (*.icp) file.
    rc_dataset: RadianceCalibrationDataset
        A radiance calibration dataset object representing the calibration
        data supplied in the *.icp file. 
    rad_conv_frame: np.ndarray
        Frame representing conversion factors from raw data to radiance for
        every pixel and wavelength.
    rad_conv_metadata: dict
        ENVI metadata for rad_conv_frame

    Methods:
    --------
    convert_raw_image_to_radiance(raw_image, raw_image_metadata):
        Convert single image (3D array) from raw to radiance
    convert_raw_file_to_radiance(raw_image, raw_image_metadata):
        Read raw file, convert, and save as radiance image
    
    Notes:
    ------
    Most image sensors register some amount of "dark current", i.e. a signal
    which is present even though no photons enter the sensor. The dark current 
    should be subtracted for the measurement to be as accurate as possible. 
    The amount of dark current also depends on camera gain (signal amplification)
    and camera shutter (time available for the sensor to collect photons). 
    The *.icp calibration file (ICP = "Imager Calibration Pack") is a ZIP archine
    which includes dark current measurements taken at different gain and shutter
    settings (raw images). To remove dark current from a new image, this set of
    dark current frames is searched, and the one which best matches the gain and
    shutter values of the new image is used as a reference. The dark frame
    needs to be scaled to account for differences in binning, gain and shutter
    between it and the new image. The dark frame is then subtracted from the image. 

    To convert raw images from digital numbers to radiance, a physical quantity,
    the raw images are multiplied with a "radiance conversion frame". This frame
    represents "microflicks per digital number" for every pixel and every 
    spectral channel, where the spectral radiance unit "flick" is defined as  
    "watts per steradian per square centimeter of surface per micrometer of span 
    in wavelength" (see https://en.wikipedia.org/wiki/Flick_(physics) ). 
    The radiance conversion frame also needs to be scaled to account for differences 
    in binning, gain and shutter.

    When both the dark current frame (DCF) and the radiance conversion frame (RCF)
    have been scaled to match the raw input image (IM), the radiance output image
    (OI) is given by OI = (IM-DCF)*RCF 

    Note that the conversion assumes that the camera response is completely linear
    after dark current is removed. This may not be completely accurate, but is 
    assumed to be within an acceptable margin of error. 
    
    """
    def __init__(self, radiance_calibration_file: Union[Path,str]):
        """ Create radiance converter object
        
        Arguments:
        ----------
        radiance_calibration_file: Path | str
            Path to Imager Calibration Pack (*.icp) file.
            The *.icp file is a zip archive, and the file will be unzipped into
            a subfolder in the same folder containing the *.icp file. 
        
        """
        
        # Set radiance calibration file
        # if radiance_calibration_file is None:
        #     icp_files = [self.raw_data_dir.glob('*.icp')]
        #     if len(icp_files) == 0:
        #         raise FileNotFoundError(f'No calibration file (*.icp) found in {self.raw_data_dir}')
        #     elif len(icp_files) > 1:
        #         raise ValueError(f'More than one calibration file (*.icp) found in {self.raw_data_dir}')
        #     self.radiance_calibration_file = icp_files[0]
        # else:
        #     self.radiance_calibration_file = radiance_calibration_file

        # Set up radiance calibration data
        self.radiance_calibration_file = Path(radiance_calibration_file)
        self.rc_dataset = RadianceCalibrationDataset(calibration_file=radiance_calibration_file)
        self.rad_conv_frame = None
        self.rad_conv_metadata = None
        self._get_rad_conv_frame()


    def _get_rad_conv_frame(self) -> None:
        """ Read radiance conversion frame from file and save as attribute """
        rad_conv_frame,_,rad_conv_metadata = self.rc_dataset.get_rad_conv_frame()
        assert rad_conv_metadata['sample binning'] == '1'
        assert rad_conv_metadata['spectral binning'] == '1'
        assert rad_conv_metadata['samples'] == '900'
        assert rad_conv_metadata['bands'] == '600'
        self.rad_conv_frame = rad_conv_frame
        self.rad_conv_metadata = rad_conv_metadata


    def _get_best_matching_dark_frame(self, raw_image_metadata: dict) -> tuple[np.ndarray,dict]:
        """ Get dark fram from calibration data that best matches input data """
        dark_frame,_,dark_frame_metadata,_,_ = self.rc_dataset.get_closest_dark_frame(
            gain=float(raw_image_metadata['gain']), shutter=float(raw_image_metadata['shutter']))
        return (dark_frame, dark_frame_metadata)


    def _scale_dark_frame(self, dark_frame: np.ndarray, dark_frame_metadata:dict,
                               raw_image_metadata:dict) -> np.ndarray:
        """ Scale dark frame to match binning, gain and shutter for input image """
        assert dark_frame_metadata['sample binning'] == '1'
        assert dark_frame_metadata['spectral binning'] == '1'
        binning_factor = (float(raw_image_metadata['sample binning']) * 
                          float(raw_image_metadata['spectral binning']))
        dark_frame = bin_image(dark_frame,
                               sample_bin_size=int(raw_image_metadata['sample binning']),
                               channel_bin_size=int(raw_image_metadata['spectral binning']))
        dark_frame = dark_frame * binning_factor

        return dark_frame
    

    def _scale_rad_conv_frame(self,raw_image_metadata:dict) -> np.ndarray:
        """ Scale radiance conversion frame to match binning, gain and shutter for input image """
        # Scaling due to binning
        binning_factor = 1.0 / (float(raw_image_metadata['sample binning']) * 
                                float(raw_image_metadata['spectral binning']))

        # Scaling due to gain differences
        rad_conv_gain = 10**(float(self.rad_conv_metadata['gain']) / 20.0)
        input_gain = 10**(float(raw_image_metadata['gain']) / 20.0)
        gain_factor = rad_conv_gain / input_gain
        
        # Scaling due to shutter differences
        rad_conv_shutter = float(self.rad_conv_metadata['shutter'])
        input_shutter = float(raw_image_metadata['shutter'])
        shutter_factor = rad_conv_shutter / input_shutter

        # Bin (average) radiance conversion frame to have same dimensions as input
        rad_conv_frame = bin_image(self.rad_conv_frame,
                                   sample_bin_size=int(raw_image_metadata['sample binning']),
                                   channel_bin_size=int(raw_image_metadata['spectral binning']),
                                   average=True) 

        # Combine factors and scale frame
        scaling_factor = binning_factor * gain_factor * shutter_factor
        rad_conv_frame = rad_conv_frame * scaling_factor

        return rad_conv_frame
        

    def convert_raw_image_to_radiance(self, 
                                      raw_image: np.ndarray,
                                      raw_image_metadata: dict
                                      ) -> np.ndarray:
        """ Convert raw image (3d array) to radiance image 
        
        Arguments:
        ----------
        raw_image: np.ndarray
            Raw hyperspectral image, shape (n_lines, n_samples, n_channels)
            The image is assumed to have been created by a Resonon Pika L
            camera, which has 900 spatial pixels x 600 spectral channels
            before any binning has been applied. Typically, spectral binning
            with a bin size of 2 is applied during image aqusition, resulting 
            in images with shape (n_lines, 900, 300). It is assumed that no
            spectral or spatial (sample) cropping has been applied. Where binning
            has been applied, it is assumed that 
                - n_samples*sample_bin_size = 900
                - n_channels*channel_bin_size = 600
        raw_image_metadata: dict
            ENVI metadata formatted as dict. 
            See spectral.io.envi.open()

        Returns:
        --------
        radiance_image: np.ndarray (int16)
            Radiance image with same shape as raw image, with spectral radiance
            in units of microflicks.

        Raises:
        -------
        ValueError:
            In case the raw image does not have the expected dimensions.
        """
        # Check input dimensions
        if int(raw_image_metadata['samples'])*int(raw_image_metadata['sample binning']) != 900:
            raise ValueError('Sample count and binning does not correspond to '
                             '900 samples in the original image.')
        if int(raw_image_metadata['bands'])*int(raw_image_metadata['spectral binning']) != 600:
            raise ValueError('Spectral band count and binning does not correspond to '
                             '600 spectral bands in the original image.')

        # Get dark frame and radiance conversion frames scaled to input image
        dark_frame, dark_frame_metadata = self._get_best_matching_dark_frame(raw_image_metadata)
        dark_frame = self._scale_dark_frame(dark_frame, dark_frame_metadata,raw_image_metadata)
        rad_conv_frame = self._scale_rad_conv_frame(raw_image_metadata)

        # Flip frames if necessary
        if (('flip radiometric calibration' in raw_image_metadata ) and
            (raw_image_metadata['flip radiometric calibration'] == 'True')):
            dark_frame = np.flip(dark_frame,axis=1) 
            rad_conv_frame = np.flip(rad_conv_frame,axis=1)

        # Subtract dark current and convert to radiance (microflicks)
        # Note: dark_frame and rad_conv_frame are implicitly "expanded" by broadcasting
        radiance_image = (raw_image - dark_frame) * rad_conv_frame

        # Set negative (non-physical) values to zero
        radiance_image[radiance_image>0] = 0

        # Convert to 16-bit integer format (more efficient for storage)
        return radiance_image.astype(np.int16)
    


    def convert_raw_file_to_radiance(self,
                                raw_header_path: Union[Path,str],
                                radiance_header_path: Union[Path,str]
                                ) -> None:
        """ Read raw image file, convert to radiance, and save to file
        
        Arguments:
        ----------
        raw_header_path: Path | str
            Path to raw hyperspectral image acquired with Resonon Pika L camera.
        radiance_header_path: Path | str
            Path to save converted radiance image to.

        # Notes:
        --------
        The radiance image is saved with the same metadata as the raw image.
        """
        raw_image,_,raw_image_metadata = read_envi(raw_header_path)
        radiance_image = self.convert_raw_image_to_radiance(raw_image,raw_image_metadata)
        save_envi(radiance_header_path,radiance_image,raw_image_metadata)
