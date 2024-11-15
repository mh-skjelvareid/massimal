import spectral
import zipfile
import numpy as np
from typing import Union
from pathlib import Path


def read_envi(header_filename: Union[Path,str], 
              image_filename: Union[Path,str,None]=None) -> tuple[np.ndarray,np.ndarray,dict]:
    """ Load image in ENVI format, including wavelength vector and other metadata

    Usage:
    ------
    (image,wl,rgb_ind,metadata) = read_envi(header_filename,...)

    Arguments:
    ----------
    header_filename: Path | str
        Path to ENVI file header.

    Keyword arguments:
    ------------------
    image_filename: Path | str
        Path to ENVI data file, useful if data file is not found
        automatically from header file name (see spectral.io.envi.open).

    Returns:
    --------
    image: np.ndarray
        image, shape (n_lines, n_samples, n_channels) 
    wl: np.ndarray
        Wavelength vector, shape (n_channels,) 
    metadata:  dict
        Image metadata (ENVI header content). 
    """

    # Open image handle
    im_handle = spectral.io.envi.open(header_filename,image_filename)

    # Read wavelengths
    wl = np.array([float(i) for i in im_handle.metadata['wavelength']])

    # Read data from disk
    image = np.array(im_handle.load())  

    # Returns
    return (image,wl,im_handle.metadata)


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
    get_radiance_conversion_frame():
        Returns radiance conversion frame, shape (n_samples, n_channels)
    get_closest_dark_frame(gain,shutter)
        Returns dark frame which best matches given gain and shutter values
    
    """

    def __init__(self, calibration_file: Union[Path,str]):
        """ Un-zip calibration file and create radiance calibration dataset 
        
        Arguments:
        ----------
        calibration_file: Path | str
            Path to *.icp Resonon "Imager Calibration Pack" file

        Raises:
        -------
        ValueError:
            If given *.icp file is not a valid zip file
        
        """
        # Register image calibration "pack" (*.icp) and check that it exists
        self.calibration_file = Path(calibration_file)
        assert self.calibration_file.exists()

        # Unzip into same directory 
        self.calibration_dir = self.calibration_file.parent / 'calibration_frames'
        self.calibration_dir.mkdir(exist_ok=True)
        self.unzip_calibration_file()

        # Register (single) gain curve file and multiple dark frame files
        self.gain_file_path = self.calibration_dir / 'gain.bip.hdr'
        assert self.gain_file_path.exists()
        self.dark_frame_paths = sorted(self.calibration_dir.glob('offset*gain*shutter.bip.hdr'))

        # Get dark frame gain and shutter info from filenames
        self._get_dark_frames_gain_shutter()

    def _unzip_calibration_file(self):
        """ Unzip *.icp file (which is a zip file) into same directory """
        if not zipfile.is_zipfile(self.calibration_file):
            raise ValueError(f'{self.calibration_file} is not a valid zip file.')
        with zipfile.ZipFile(self._filename, mode='r') as cal_file:
            # TODO: Put in try-catch block
            for filename in cal_file.namelist():
                self.calibration_file.extract(filename, self.calibration_dir)


    def _get_dark_frames_gain_shutter(self):
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
        self._dark_frame_gains = np.array(dark_frame_gains,dtype=np.float)
        self._dark_frame_shutters = np.array(dark_frame_shutters,dtype=np.float)

            
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
        best_match_mask = (self._dark_frame_gains==closest_gain) & (self._dark_frame_shutters==closest_shutter)
        best_match_ind = np.nonzero(best_match_mask)
        assert len(best_match_ind) == 1 # There should only be a single best match
        return self.dark_frame_paths[best_match_ind], closest_gain, closest_shutter
        
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
            TODO: Check shape
        wl: np.ndarray 
            Vector of wavelengths for each spectral channel (nanometers)
        metadata: dict
            Metadata from ENVI header, formatted as dictionary
        """
        closest_file, closest_gain, closest_shutter = self._get_closest_dark_frame_path()
        frame, wl, metadata = read_envi(closest_file)
        return frame, wl, metadata, closest_gain, closest_shutter


    def get_radiance_conversion_frame(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """ Read and return radiance conversion curve ("gain" file) 
        
        Returns:
        --------
        frame: np.ndarray
            TODO: Check shape
        wl: np.ndarray 
            Vector of wavelengths for each spectral channel (nanometers)
        metadata: dict
            Metadata from ENVI header, formatted as dictionary
        
        """
        return read_envi(self.gain_file_path)  




class RadianceProcessor:

    def __init__(self, 
                 raw_data_dir: Union[Path,str],
                 radiance_calibration_file: Union[Path,str]=None):
        
        # Set raw data dir 
        self.raw_data_dir = Path(raw_data_dir)
        
        # Set radiance calibration file
        if radiance_calibration_file is None:
            icp_files = [self.raw_data_dir.glob('*.icp')]
            if len(icp_files) == 0:
                raise ValueError(f'No calibration file (*.icp) found in {self.raw_data_dir}')
            elif len(icp_files) > 1:
                raise ValueError(f'More than one calibration file (*.icp) found in {self.raw_data_dir}')
            self.radiance_calibration_file = icp_files[0]
        else:
            self.radiance_calibration_file = icp_files[0]

        

    