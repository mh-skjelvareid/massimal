# Imports
import json
import logging
import subprocess
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pyproj
import rasterio
import spectral
from numpy.polynomial import Polynomial
from rasterio.crs import CRS
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)


def read_envi(
    header_path: Union[Path, str],
    image_path: Union[Path, str, None] = None,
    write_byte_order_if_missing=True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load image in ENVI format, including wavelength vector and other metadata

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
        im_handle = spectral.io.envi.open(header_path, image_path)
    except spectral.io.envi.MissingEnviHeaderParameter as e:
        logging.debug(f"Header file has missing parameter: {header_path}")
        byte_order_missing_str = (
            'Mandatory parameter "byte order" missing from header file.'
        )
        if str(e) == byte_order_missing_str and write_byte_order_if_missing:
            logging.debug('Writing "byte order = 0" to header file and retrying')
            try:
                with open(header_path, "a") as file:
                    file.write("byte order = 0\n")
            except OSError:
                logger.error(
                    f"Error writing to header file {header_path}", exc_info=True
                )

            try:
                im_handle = spectral.io.envi.open(header_path, image_path)
            except Exception:
                logger.error(
                    f"Unsucessful when reading modified header file {header_path}",
                    exc_info=True,
                )
                return

    # Read wavelengths
    if "wavelength" in im_handle.metadata:
        wl = np.array([float(i) for i in im_handle.metadata["wavelength"]])
    else:
        wl = None

    # Read data from disk
    image = np.array(im_handle.load())

    # Returns
    return (image, wl, im_handle.metadata)


def save_envi(
    header_path: Union[Path, str], image: np.ndarray, metadata: dict, **kwargs
) -> None:
    """Save ENVI file with parameters compatible with Spectronon

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

    # Save file
    spectral.envi.save_image(
        header_path, image, metadata=metadata, force=True, ext=None, **kwargs
    )


def wavelength_array_to_header_string(wavelengths: np.ndarray):
    """Convert numeric wavelength array to single ENVI-formatted string"""
    wl_str = [f"{wl:.3f}" for wl in wavelengths]  # Convert each number to string
    wl_str = "{" + ", ".join(wl_str) + "}"  # Join into single string
    return wl_str


def update_header_wavelengths(wavelengths: np.ndarray, header_path: Union[Path, str]):
    """Update ENVI header wavelengths"""
    header_path = Path(header_path)
    header_dict = spectral.io.envi.read_envi_header(header_path)
    wl_str = wavelength_array_to_header_string(wavelengths)
    header_dict["wavelength"] = wl_str
    spectral.io.envi.write_envi_header(header_path, header_dict)


def bin_image(
    image: np.ndarray,
    line_bin_size: int = 1,
    sample_bin_size: int = 1,
    channel_bin_size: int = 1,
    average: bool = True,
) -> np.ndarray:
    """Bin image cube (combine neighboring pixels)

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

    image = image.reshape(
        n_lines_binned,
        line_bin_size,
        n_samples_binned,
        sample_bin_size,
        n_channels_binned,
        channel_bin_size,
    )
    if average:
        image = np.mean(image, axis=(1, 3, 5))
    else:
        image = np.sum(image, axis=(1, 3, 5))

    return image


def savitzky_golay_filter(
    image, window_length: int = 13, polyorder: int = 3, axis: int = 2
) -> np.ndarray:
    """Filter hyperspectral image using Savitzky-Golay filter with default arguments"""
    return savgol_filter(
        image, window_length=window_length, polyorder=polyorder, axis=axis
    )


def closest_wl_index(wl_array: np.ndarray, target_wl: Union[float, int]) -> int:
    """Get index in sampled wavelength array closest to target wavelength"""
    return np.argmin(abs(wl_array - target_wl))


def rgb_subset_from_hsi(
    hyspec_im: np.ndarray, hyspec_wl, rgb_target_wl=(650, 550, 450)
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3 bands from hyperspectral image representing red, green, blue

    Arguments:
    ----------
    hyspec_im: np.ndarray
        Hyperspectral image, shape (n_lines, n_samples, n_bands)
    hyspec_wl: np.ndarray
        Wavelengths for each band of hyperspectral image, in nm.
        Shape (n_bands,)

    Returns:
    --------
    rgb_im: np.ndarray
        3-band image representing red, green and blue color (in that order)
    rgb_wl: np.ndarray
        3-element vector with wavelengths (in nm) corresponding to
        each band of rgb_im.

    """
    wl_ind = [closest_wl_index(hyspec_wl, wl) for wl in rgb_target_wl]
    rgb_im = hyspec_im[:, :, wl_ind]
    rgb_wl = hyspec_wl[wl_ind]
    return rgb_im, rgb_wl


def convert_long_lat_to_utm(
    long: Union[float, np.ndarray], lat: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], int]:
    """Convert longitude and latitude coordinates (WGS84) to UTM

    # Input parameters:
    long:
        Longitude coordinate(s), scalar or array
    lat:
        Latitude coordinate(s), scalar or array

    Returns:
    UTMx:
        UTM x coordinate ("Easting"), scalar or array
    UTMy:
        UTM y coordinate ("Northing"), scalar or array
    UTM_epsg :
        EPSG code (integer) for UTM zone
    """
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=min(long),
            south_lat_degree=min(lat),
            east_lon_degree=max(long),
            north_lat_degree=max(lat),
        ),
    )
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    proj = pyproj.Proj(utm_crs)
    UTMx, UTMy = proj(long, lat)

    return UTMx, UTMy, utm_crs.to_epsg()


class RadianceCalibrationDataset:
    """A radiance calibration dataset for Resonon hyperspectral cameras.

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

    def __init__(
        self,
        calibration_file: Union[Path, str],
        calibration_dir_name: str = "radiance_calibration_frames",
    ):
        """Un-zip calibration file and create radiance calibration dataset

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
        self.gain_file_path = self.calibration_dir / "gain.bip.hdr"
        assert self.gain_file_path.exists()
        self.dark_frame_paths = list(
            self.calibration_dir.glob("offset*gain*shutter.bip.hdr")
        )

        # Get dark frame gain and shutter info from filenames
        self._get_dark_frames_gain_shutter()

        # Sort gain/shutter values and corresponding filenames
        self._sort_dark_frame_gains_shutters_paths()

    def _unzip_calibration_file(self, unzip_into_nonempty_dir: bool = False) -> None:
        """Unzip *.icp file (which is a zip file)"""
        if not unzip_into_nonempty_dir and any(list(self.calibration_dir.iterdir())):
            logger.info(f"Non-empty calibration directory {self.calibration_dir}")
            logger.info("Assuming calibration file already unzipped.")
            return
        try:
            with zipfile.ZipFile(self.calibration_file, mode="r") as zip_file:
                for filename in zip_file.namelist():
                    zip_file.extract(filename, self.calibration_dir)
        except zipfile.BadZipFile:
            logger.error(
                f"File {self.calibration_file} is not a valid ZIP file.", exc_info=True
            )
        except Exception:
            logger.error(
                f"Unexpected error when extracting calibration file {self.calibration_file}",
                exc_info=True,
            )

    def _get_dark_frames_gain_shutter(self) -> None:
        """Extract and save gain and shutter values for each dark frame"""
        # Example dark frame pattern:
        # offset_600bands_4095ceiling_5gain_900samples_75shutter.bip.hdr
        dark_frame_gains = []
        dark_frame_shutters = []
        for dark_frame_path in self.dark_frame_paths:
            # Strip file extensions, split on underscores, keep only gain and shutter info
            _, _, _, gain_str, _, shutter_str = dark_frame_path.name.split(".")[
                0
            ].split("_")
            dark_frame_gains.append(int(gain_str[:-4]))
            dark_frame_shutters.append(int(shutter_str[:-7]))
        # Save as NumPy arrays
        self._dark_frame_gains = np.array(dark_frame_gains, dtype=float)
        self._dark_frame_shutters = np.array(dark_frame_shutters, dtype=float)

    def _sort_dark_frame_gains_shutters_paths(self) -> None:
        """Sort gain/shutter values and corresponding file names"""
        gain_shutter_path_sorted = sorted(
            zip(
                self._dark_frame_gains, self._dark_frame_shutters, self.dark_frame_paths
            )
        )
        self._dark_frame_gains = np.array(
            [gain for gain, _, _ in gain_shutter_path_sorted]
        )
        self._dark_frame_shutters = np.array(
            [shutter for _, shutter, _ in gain_shutter_path_sorted]
        )
        self.dark_frame_paths = [path for _, _, path in gain_shutter_path_sorted]

    def _get_closest_dark_frame_path(
        self, gain: Union[int, float], shutter: Union[int, float]
    ) -> Path:
        """Search for dark frame with best matching gain and shutter values"""
        # First search for files with closest matching gain
        candidate_gains = np.unique(self._dark_frame_gains)
        closest_gain = candidate_gains[np.argmin(abs(candidate_gains - gain))]

        # Then search (in subset) for single file with closest matching shutter
        candidate_shutters = np.unique(
            self._dark_frame_shutters[self._dark_frame_gains == closest_gain]
        )
        closest_shutter = self._dark_frame_shutters[
            np.argmin(abs(candidate_shutters - shutter))
        ]

        # Return best match
        best_match_mask = (self._dark_frame_gains == closest_gain) & (
            self._dark_frame_shutters == closest_shutter
        )
        best_match_ind = np.nonzero(best_match_mask)[0]
        assert len(best_match_ind) == 1  # There should only be a single best match
        return self.dark_frame_paths[best_match_ind[0]], closest_gain, closest_shutter

    def get_closest_dark_frame(
        self, gain: Union[int, float], shutter: Union[int, float]
    ) -> tuple[np.ndarray, np.ndarray, dict, float, float]:
        """Get dark frame which most closely matches given gain and shutter values

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
            gain=gain, shutter=shutter
        )
        frame, wl, metadata = read_envi(closest_file)
        return frame, wl, metadata, closest_gain, closest_shutter

    def get_rad_conv_frame(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Read and return radiance conversion curve ("gain" file)

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
    """A class for converting raw hyperspectral images from Pika L cameras to radiance.

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

    def __init__(self, radiance_calibration_file: Union[Path, str]):
        """Create radiance converter object

        Arguments:
        ----------
        radiance_calibration_file: Path | str
            Path to Imager Calibration Pack (*.icp) file.
            The *.icp file is a zip archive, and the file will be unzipped into
            a subfolder in the same folder containing the *.icp file.

        """

        self.radiance_calibration_file = Path(radiance_calibration_file)
        self.rc_dataset = RadianceCalibrationDataset(
            calibration_file=radiance_calibration_file
        )
        self.rad_conv_frame = None
        self.rad_conv_metadata = None
        self._get_rad_conv_frame()

    def _get_rad_conv_frame(self) -> None:
        """Read radiance conversion frame from file and save as attribute"""
        rad_conv_frame, _, rad_conv_metadata = self.rc_dataset.get_rad_conv_frame()
        assert rad_conv_metadata["sample binning"] == "1"
        assert rad_conv_metadata["spectral binning"] == "1"
        assert rad_conv_metadata["samples"] == "900"
        assert rad_conv_metadata["bands"] == "600"
        self.rad_conv_frame = rad_conv_frame
        self.rad_conv_metadata = rad_conv_metadata

    def _get_best_matching_dark_frame(
        self, raw_image_metadata: dict
    ) -> tuple[np.ndarray, dict]:
        """Get dark fram from calibration data that best matches input data"""
        dark_frame, _, dark_frame_metadata, _, _ = (
            self.rc_dataset.get_closest_dark_frame(
                gain=float(raw_image_metadata["gain"]),
                shutter=float(raw_image_metadata["shutter"]),
            )
        )
        return (dark_frame, dark_frame_metadata)

    def _scale_dark_frame(
        self,
        dark_frame: np.ndarray,
        dark_frame_metadata: dict,
        raw_image_metadata: dict,
    ) -> np.ndarray:
        """Scale dark frame to match binning for input image"""
        assert dark_frame_metadata["sample binning"] == "1"
        assert dark_frame_metadata["spectral binning"] == "1"
        binning_factor = float(raw_image_metadata["sample binning"]) * float(
            raw_image_metadata["spectral binning"]
        )
        dark_frame = bin_image(
            dark_frame,
            sample_bin_size=int(raw_image_metadata["sample binning"]),
            channel_bin_size=int(raw_image_metadata["spectral binning"]),
        )
        dark_frame = dark_frame * binning_factor
        # NOTE: Dark frame not scaled based on differences in gain and shutter because
        # the best matching dark frame has (approx.) the same values already.

        return dark_frame

    def _scale_rad_conv_frame(self, raw_image_metadata: dict) -> np.ndarray:
        """Scale radiance conversion frame to match binning, gain and shutter for input image"""
        # Scaling due to binning
        binning_factor = 1.0 / (
            float(raw_image_metadata["sample binning"])
            * float(raw_image_metadata["spectral binning"])
        )

        # Scaling due to gain differences
        rad_conv_gain = 10 ** (float(self.rad_conv_metadata["gain"]) / 20.0)
        input_gain = 10 ** (float(raw_image_metadata["gain"]) / 20.0)
        gain_factor = rad_conv_gain / input_gain

        # Scaling due to shutter differences
        rad_conv_shutter = float(self.rad_conv_metadata["shutter"])
        input_shutter = float(raw_image_metadata["shutter"])
        shutter_factor = rad_conv_shutter / input_shutter

        # Bin (average) radiance conversion frame to have same dimensions as input
        rad_conv_frame = bin_image(
            self.rad_conv_frame,
            sample_bin_size=int(raw_image_metadata["sample binning"]),
            channel_bin_size=int(raw_image_metadata["spectral binning"]),
            average=True,
        )

        # Combine factors and scale frame
        scaling_factor = binning_factor * gain_factor * shutter_factor
        rad_conv_frame = rad_conv_frame * scaling_factor

        return rad_conv_frame

    def convert_raw_image_to_radiance(
        self,
        raw_image: np.ndarray,
        raw_image_metadata: dict,
        set_saturated_pixels_to_zero: bool = True,
        saturation_value: int = 2**12 - 1,
    ) -> np.ndarray:
        """Convert raw image (3d array) to radiance image

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
        radiance_image: np.ndarray (int16, microflicks)
            Radiance image with same shape as raw image, with spectral radiance
            in units of microflicks = 10e-5 W/(m2*nm). Microflicks are used
            to be consistent with Resonon formatting, and because microflick
            values typically are in a range suitable for (memory-efficient)
            encoding as 16-bit unsigned integer.

        Raises:
        -------
        ValueError:
            In case the raw image does not have the expected dimensions.

        References:
        -----------
        - ["flick" unit](https://en.wikipedia.org/wiki/Flick_(physics))
        """
        # Check input dimensions
        if (
            int(raw_image_metadata["samples"])
            * int(raw_image_metadata["sample binning"])
            != 900
        ):
            raise ValueError(
                "Sample count and binning does not correspond to "
                "900 samples in the original image."
            )
        if (
            int(raw_image_metadata["bands"])
            * int(raw_image_metadata["spectral binning"])
            != 600
        ):
            raise ValueError(
                "Spectral band count and binning does not correspond to "
                "600 spectral bands in the original image."
            )

        # Get dark frame and radiance conversion frames scaled to input image
        dark_frame, dark_frame_metadata = self._get_best_matching_dark_frame(
            raw_image_metadata
        )
        dark_frame = self._scale_dark_frame(
            dark_frame, dark_frame_metadata, raw_image_metadata
        )
        rad_conv_frame = self._scale_rad_conv_frame(raw_image_metadata)

        # Flip frames if necessary
        if ("flip radiometric calibration" in raw_image_metadata) and (
            raw_image_metadata["flip radiometric calibration"] == "True"
        ):
            dark_frame = np.flip(dark_frame, axis=1)
            rad_conv_frame = np.flip(rad_conv_frame, axis=1)

        # Subtract dark current and convert to radiance (microflicks)
        # Note: dark_frame and rad_conv_frame are implicitly "expanded" by broadcasting
        radiance_image = (raw_image - dark_frame) * rad_conv_frame

        # Set negative (non-physical) values to zero
        radiance_image[radiance_image < 0] = 0

        # Set saturated pixels to zero (optional)
        if set_saturated_pixels_to_zero:
            radiance_image[np.any(raw_image >= saturation_value, axis=2)] = 0

        # Convert to 16-bit integer format (more efficient for storage)
        return radiance_image.astype(np.uint16)

    def convert_raw_file_to_radiance(
        self,
        raw_header_path: Union[Path, str],
        radiance_header_path: Union[Path, str],
        interleave: str = "bip",
    ) -> None:
        """Read raw image file, convert to radiance, and save to file

        Arguments:
        ----------
        raw_header_path: Path | str
            Path to raw hyperspectral image acquired with Resonon Pika L camera.
        radiance_header_path: Path | str
            Path to save converted radiance image to.
            The name of the header file should match the 'interleave' argument
            (default: bip), e.g. 'radiance_image.bip.hdr'

        Keyword arguments:
        ------------------
        interleave: str, {'bip','bil','bsq'}
            String inticating how binary image file is organized.
            See spectral.io.envi.save_image()

        # Notes:
        --------
        The radiance image is saved with the same metadata as the raw image.
        """
        raw_image, _, raw_image_metadata = read_envi(raw_header_path)
        radiance_image = self.convert_raw_image_to_radiance(
            raw_image, raw_image_metadata
        )
        save_envi(
            radiance_header_path,
            radiance_image,
            raw_image_metadata,
            interleave=interleave,
        )


class IrradianceConverter:
    def __init__(
        self,
        irrad_cal_file: Union[Path, str],
        irrad_cal_dir_name: str = "downwelling_calibration_spectra",
        wl_min: Union[int, float, None] = 370,
        wl_max: Union[int, float, None] = 1000,
    ):
        # Save calibration file path
        self.irrad_cal_file = Path(irrad_cal_file)
        assert self.irrad_cal_file.exists()

        # Unzip calibration frames into subdirectory
        self.irrad_cal_dir = self.irrad_cal_file.parent / irrad_cal_dir_name
        self.irrad_cal_dir.mkdir(exist_ok=True)
        self._unzip_irrad_cal_file()

        # Load calibration data
        self._load_cal_dark_and_sensitivity_spectra()

        # Set valid wavelength range
        self.wl_min = self._irrad_wl[0] if wl_min is None else wl_min
        self.wl_max = self._irrad_wl[-1] if wl_max is None else wl_max
        self._valid_wl_ind = (self._irrad_wl >= wl_min) & (self._irrad_wl <= wl_max)

    def _unzip_irrad_cal_file(self, unzip_into_nonempty_dir: bool = False) -> None:
        """Unzip *.dcp file (which is a zip file)"""
        if not unzip_into_nonempty_dir and any(list(self.irrad_cal_dir.iterdir())):
            logger.info(
                f"Non-empty downwelling calibration directory {self.irrad_cal_dir}"
            )
            logger.info(
                "Skipping unzipping of downwelling calibration file, "
                "assuming unzipping already done."
            )
            return
        try:
            with zipfile.ZipFile(self.irrad_cal_file, mode="r") as zip_file:
                for filename in zip_file.namelist():
                    zip_file.extract(filename, self.irrad_cal_dir)
        except zipfile.BadZipFile:
            logger.error(
                f"File {self.irrad_cal_file} is not a valid ZIP file.", exc_info=True
            )
        except Exception:
            logger.error(
                f"Error while extracting downwelling calibration file {self.irrad_cal_file}",
                exc_info=True,
            )

    def _load_cal_dark_and_sensitivity_spectra(self):
        """Load dark current and irradiance sensitivity spectra from cal. files"""
        # Define paths
        cal_dark_path = self.irrad_cal_dir / "offset.spec.hdr"
        irrad_sens_path = self.irrad_cal_dir / "gain.spec.hdr"
        assert cal_dark_path.exists()
        assert irrad_sens_path.exists()

        # Read from files
        cal_dark_spec, cal_dark_wl, cal_dark_metadata = read_envi(cal_dark_path)
        irrad_sens_spec, irrad_sens_wl, irrad_sens_metadata = read_envi(irrad_sens_path)

        # Save attributes
        assert np.array_equal(cal_dark_wl, irrad_sens_wl)
        self._irrad_wl = cal_dark_wl
        self._cal_dark_spec = np.squeeze(cal_dark_spec)  # Remove singleton axes
        self._cal_dark_metadata = cal_dark_metadata
        self._irrad_sens_spec = np.squeeze(irrad_sens_spec)  # Remove singleton axes
        self._irrad_sens_metadata = irrad_sens_metadata
        self._cal_dark_shutter = float(cal_dark_metadata["shutter"])
        self._irrad_sens_shutter = float(irrad_sens_metadata["shutter"])

    def convert_raw_spectrum_to_irradiance(
        self,
        raw_spec: np.ndarray,
        raw_metadata: np.ndarray,
        set_irradiance_outside_wl_limits_to_zero: bool = True,
        keep_original_dimensions: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """

        Returns:
        --------
        irradiance_spectrum: np.ndarray, shape (n_wl,)
            Spectrom converted to spectral irradiance, unit W/(m2*nm)
        wl: np.ndarray, shape (n_wl,)
            Wavelengths for each element in irradiance spectrum

        Notes:
        -------
        The irradiance conversion spectrum is _inversely_ scaled with
        the input spectrum shutter. E.g., if the input spectrum has a higher shutter
        value than the calibration file (i.e. higher values per amount of photons),
        the conversion spectrum values are decreased to account for this.
        Dark current is assumed to be independent of shutter value.
        """

        original_input_dimensions = raw_spec.shape
        raw_spec = np.squeeze(raw_spec)

        if raw_spec.ndim > 1:
            raise ValueError("Raw irradiance spectrum must be a 1D array.")

        # Scale conversion spectrum according to difference in shutter values
        raw_shutter = float(raw_metadata["shutter"])
        scaled_conv_spec = (
            self._irrad_sens_shutter / raw_shutter
        ) * self._irrad_sens_spec

        # Subtract dark current, multiply with radiance conversion spectrum
        # NOTE: Resonon irradiance unit is uW/(pi*cm2*um) = 10e-5 W/(pi*m2*nm)
        cal_irrad_spec = (raw_spec - self._cal_dark_spec) * scaled_conv_spec

        # Convert to standard spectral irradiance unit W/(m2*nm)
        cal_irrad_spec = cal_irrad_spec * (np.pi / 100_000)

        # Set spectrum outside wavelength limits to zero
        if set_irradiance_outside_wl_limits_to_zero:
            cal_irrad_spec[~self._valid_wl_ind] = 0

        if keep_original_dimensions:
            cal_irrad_spec = np.reshape(cal_irrad_spec, original_input_dimensions)

        return cal_irrad_spec

    def convert_raw_file_to_irradiance(
        self, raw_spec_path: Union[Path, str], irrad_spec_path: Union[Path, str]
    ):
        """Read raw spectrum, convert to irradiance, and save"""
        raw_spec, _, spec_metadata = read_envi(raw_spec_path)
        irrad_spec = self.convert_raw_spectrum_to_irradiance(raw_spec, spec_metadata)
        spec_metadata["unit"] = "W/(m2*nm)"
        save_envi(irrad_spec_path, irrad_spec, spec_metadata)


class WavelengthCalibrator:
    def __init__(self):
        self._fh_line_indices = None
        self._fh_wavelengths = None
        self._wl_poly_coeff = None
        self.reference_spectrum_path = None
        self.wl_cal = None
        self.max_wl_diff = None

        self.fraunhofer_wls = {
            "L": 382.04,
            "G": 430.78,
            "F": 486.13,
            "b1": 518.36,
            "D": 589.30,
            "C": 656.28,
            "B": 686.72,
            "A": 760.30,  # Not well defined (O2 band), approximate
            "Z": 822.70,
        }

    @staticmethod
    def detect_absorption_lines(
        spec: np.ndarray,
        wl: np.ndarray,
        distance: int = 20,
        width: int = 5,
        rel_prominence: float = 0.1,
    ):
        """Detect absorption lines using local peak detection"""
        wl_550_ind = closest_wl_index(wl, 550)
        prominence = spec[wl_550_ind] * rel_prominence
        peak_indices, peak_properties = find_peaks(
            -spec, distance=distance, width=width, prominence=prominence
        )
        return peak_indices, peak_properties

    @staticmethod
    def fit_wavelength_polynomial(
        sample_indices: np.ndarray, wavelengths: np.ndarray, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit 2nd degree polynomial to set of (sample, wavelength) pairs

        Arguments:
        ----------
        sample_indices: np.ndarray
            Indices of samples in a sampled spectrum, typically for spectral
            peaks / absorption lines with known wavelengths.
        wavelengths: np.ndarray
            Wavelengths (in physical units) corresponding to sample_indices.
        n_samples: int
            Total number of samples in sampled spectrum

        Returns:
        --------
        wl_cal:
            Array of (calibrated) wavelengths, shape (n_samples,)
        wl_poly_coeff:
            Coefficients for 2nd degree polynomial, ordered from degree zero upward
            (index corresponds to polynomial degree)

        """
        polynomial_fitted = Polynomial.fit(
            sample_indices, wavelengths, deg=2, domain=[]
        )
        wl_cal = polynomial_fitted(np.arange(n_samples))
        wl_poly_coeff = polynomial_fitted.coef
        return wl_cal, wl_poly_coeff

    def _filter_fraunhofer_lines(self, line_indices, orig_wl, win_width_nm=20):
        """Calibrate wavelength values from known Fraunhofer absorption lines

        Arguments:
        ----------
        line_indices: np.ndarray
            Indices of samples in spectrum where a (potential) absorption
            line has been detected. Indices must be within the range
            (0, len(orig_wl))
        orig_wl: np.ndarray
            Original wavelength vector. Values are assumed to be "close enough"
            to be used to create search windows for each Fraunhofer line,
            typically within a few nm. Shape (n_samples,)
        win_width_nm:
            The width of the search windows in units of nanometers.

        Returns:
        filtered_line_indices:
            Indices for absorption lines found close to Fraunhofer line
        fraunhofer_wavelengths:
            Corresponding Fraunhofer line wavelengths for filtered absorption lines

        """

        filtered_line_indices = []
        fraunhofer_wavelengths = []
        for fh_line_wl in self.fraunhofer_wls.values():
            # Find index of closest sample to Fraunhofer wavelength
            fh_wl_ind = closest_wl_index(orig_wl, fh_line_wl)

            # Calculate half window width in samples at current wavelength
            wl_resolution = orig_wl[fh_wl_ind + 1] - orig_wl[fh_wl_ind]
            win_half_width = round((0.5 * win_width_nm) / wl_resolution)

            # Calculate edges of search window
            win_low = fh_wl_ind - win_half_width
            win_high = fh_wl_ind + win_half_width

            # Find lines within search window, accept if single peak found
            peaks_in_window = line_indices[
                (line_indices >= win_low) & (line_indices <= win_high)
            ]
            if len(peaks_in_window) == 1:
                filtered_line_indices.append(peaks_in_window[0])
                fraunhofer_wavelengths.append(fh_line_wl)

        return filtered_line_indices, fraunhofer_wavelengths

    def fit(self, spec: np.ndarray, wl_orig: np.ndarray) -> np.ndarray:
        """Detect absorption lines in spectrum and fit polynomial wavelength function

        Arguments:
        ----------
        spec: np.ndarray
            Sampled radiance/irradiance spectrum, shape (n_samples,)
        wl_orig: np.ndarray
            Wavelengths corresponding to spectral samples, shape (n_samples)
            Wavelengths values are assumed to be close (within a few nm)
            to their true values.
        """
        spec = np.squeeze(spec)
        if spec.ndim > 1:
            raise ValueError("Spectrum must be a 1D array")

        line_indices, _ = self.detect_absorption_lines(spec, wl_orig)
        fh_line_indices, fh_wavelengths = self._filter_fraunhofer_lines(
            line_indices, wl_orig
        )
        if len(fh_line_indices) < 3:
            raise ValueError(
                "Too low data quality: Less than 3 absorption lines found."
            )
        wl_cal, wl_poly_coeff = self.fit_wavelength_polynomial(
            fh_line_indices, fh_wavelengths, len(wl_orig)
        )

        self._fh_line_indices = fh_line_indices
        self._fh_wavelengths = fh_wavelengths
        self._wl_poly_coeff = wl_poly_coeff
        self.wl_cal = wl_cal
        self.max_wl_diff = np.max(abs(wl_cal - wl_orig))

    def fit_batch(self, spectrum_header_paths: Iterable[Union[Path, str]]):
        """Calibrate wavelength based on spectrum with highest SNR (among many)

        Arguments:
        ----------
        spectrum_header_paths: Iterable[Path | str]
            Paths to multiple spectra. The spectrum with the highest maximum
            value will be used for wavelength calibration.
            Spectra are assumed to be ENVI files.

        """
        spectrum_header_paths = list(spectrum_header_paths)
        spectra = []
        for spectrum_path in spectrum_header_paths:
            spectrum_path = Path(spectrum_path)
            try:
                spec, wl, _ = read_envi(spectrum_path)
            except OSError:
                logger.warning(f"Error opening spectrum {spectrum_path}", exc_info=True)
                logger.warning("Skipping spectrum.")
            spectra.append(np.squeeze(spec))

        spectra = np.array(spectra)
        best_spec_ind = np.argmax(np.max(spectra, axis=1))
        cal_spec = spectra[best_spec_ind]
        self.reference_spectrum_path = str(spectrum_header_paths[best_spec_ind])

        self.fit(cal_spec, wl)

    def update_header_wavelengths(self, header_path: Union[Path, str]):
        """Update header file with calibrated wavelengths

        Arguments:
        ----------
        header_paths: Path | str
            Iterable with paths multiple spectra.

        """
        if self.wl_cal is None:
            raise AttributeError("Attribute wl_cal is not set - fit (calibrate) first.")
        update_header_wavelengths(self.wl_cal, header_path)


class ImuDataParser:
    @staticmethod
    def read_lcf_file(lcf_file_path, time_rel_to_file_start=True):
        """Read location files (.lcf) generated by Resonon Airborne Hyperspectral imager

        Arguments:
        ----------
        lcf_file_path:
            Path to lcf file. Usually a "sidecar" file to an hyperspectral image
            with same "base" filename.

        Keyword arguments (optional):
        time_rel_to_file_start:
            Boolean indicating if first timestamp should be subtracted from each
            timestamp, making time relative to start of file.

        # Returns:
        ----------
        lcf_data:
            Dictionary with keys describing the type of data, and data
            formatted as numpy arrays. All arrays have equal length.

            The 7 types of data:
            - 'time': System time in seconds, relative to some (unknown)
            starting point. Similar to "Unix time" (seconds since January 1. 1970),
            but values indicate starting point around 1980. The values are
            usually offset to make the first timestamp equal to zero.
            See flag time_rel_to_file_start.
            - 'roll': Roll angle in radians, positive for "right wing up"
            - 'pitch': Pitch angle in radians, positive nose up
            - 'yaw': (heading) in radians, zero at due North, PI/2 at due East
            - 'longitude': Longitude in decimal degrees, negative for west longitude
            - 'latitude': Latitude in decimal degrees, negative for southern hemisphere
            - 'altitude': Altitude in meters relative to the WGS-84 ellipsiod.

        # Notes:
        - The LCF file format was shared by Casey Smith at Resonon on February 16. 2021.
        - The LCF specification was inherited from Space Computer Corp.
        """

        # Load LCF data
        lcf_raw = np.loadtxt(lcf_file_path)
        column_headers = [
            "time",
            "roll",
            "pitch",
            "yaw",
            "longitude",
            "latitude",
            "altitude",
        ]
        lcf_data = {header: lcf_raw[:, i] for i, header in enumerate(column_headers)}

        if time_rel_to_file_start:
            lcf_data["time"] -= lcf_data["time"][0]

        return lcf_data

    @staticmethod
    def read_times_file(times_file_path, time_rel_to_file_start=True):
        """Read image line timestamps (.times) file generated by Resonon camera

        Arguments:
        ----------
        times_file_path:
            Path to times file. Usually a "sidecar" file to an hyperspectral image
            with same "base" filename.
        time_rel_to_file_start:
            Boolean indicating if times should be offset so that first
            timestamp is zero. If not, the original timestamp value is returned.

        # Returns:
        ----------
        times:
            Numpy array containing timestamps for every line of the corresponding
            hyperspectral image. The timestamps are in units of seconds, and are
            relative to when the system started (values are usually within the
            0-10000 second range). If time_rel_to_file_start=True, the times
            are offset so that the first timestamp is zero.

            The first timestamp of the times file and the  first timestamp of the
            corresponding lcf file (GPS/IMU data) are assumed to the
            recorded at exactly the same time. If both sets of timestamps are
            offset so that time is measured relative to the start of the file,
            the times can be used to calculate interpolated GPS/IMU values
            for each image line.

        """
        image_times = np.loadtxt(times_file_path)
        if time_rel_to_file_start:
            image_times = image_times - image_times[0]
        return image_times

    @staticmethod
    def interpolate_lcf_to_times(lcf_data, image_times, convert_to_list=True):
        """Interpolate LCF data to image line times"""
        lcf_data_interp = {}
        lcf_times = lcf_data["time"]
        for lcf_key, lcf_value in lcf_data.items():
            lcf_data_interp[lcf_key] = np.interp(image_times, lcf_times, lcf_value)
            if convert_to_list:
                lcf_data_interp[lcf_key] = lcf_data_interp[lcf_key].tolist()
        return lcf_data_interp

    def read_and_save_imu_data(self, lcf_path, times_path, json_path):
        lcf_data = self.read_lcf_file(lcf_path)
        times_data = self.read_times_file(times_path)
        lcf_data_interp = self.interpolate_lcf_to_times(lcf_data, times_data)

        with open(json_path, "w", encoding="utf-8") as write_file:
            json.dump(lcf_data_interp, write_file, ensure_ascii=False, indent=4)

    @staticmethod
    def read_imu_json_file(imu_json_path):
        with open(imu_json_path, "r") as imu_file:
            imu_data = json.load(imu_file)
        return imu_data


class ReflectanceConverter:
    """A class for converting images from Resonon Pika L cameras to reflectance"""

    def __init__(
        self,
        wl_min: Union[int, float] = 400,
        wl_max: Union[int, float] = 930,
        irrad_spec_paths=None,
    ):
        """

        Keyword arguments:
        wl_min, wl_max: int | float
            Minimum/maximum wavelength to include in the reflectance image.
            The signal-to-noise ratio of both radiance images and irradiance
            spectra is generally lower at the low and high ends. When
            radiance is divided by noisy irradiance values close to zero, the
            noise can "blow up". Limiting the wavelength range can ensure
            that the reflectance images have more well-behaved values.
        irrad_spec_paths:
            List of paths to irradiance spectra which can be used as reference
            spectra when convering radiance to irradiance.

        """
        self.wl_min = float(wl_min)
        self.wl_max = float(wl_max)
        if irrad_spec_paths is not None:
            irrad_spec_mean, irrad_wl, irrad_spectra = self.get_mean_irrad_spec(
                irrad_spec_paths
            )
        else:
            irrad_spec_mean, irrad_wl, irrad_spectra = None, None, None
        self.ref_irrad_spec_mean = irrad_spec_mean
        self.ref_irrad_spec_wl = irrad_wl
        self.ref_irrad_spectra = irrad_spectra

    @staticmethod
    def get_mean_irrad_spec(irrad_spec_paths):
        """Read irradiance spectra from file and calculate mean"""
        irrad_spectra = []
        for irrad_spec_path in irrad_spec_paths:
            if irrad_spec_path.exists():
                irrad_spec, irrad_wl, _ = read_envi(irrad_spec_path)
                irrad_spectra.append(irrad_spec.squeeze())
        irrad_spectra = np.array(irrad_spectra)
        irrad_spec_mean = np.mean(irrad_spectra, axis=0)
        return irrad_spec_mean, irrad_wl, irrad_spectra

    @staticmethod
    def conv_spec_with_gaussian(
        spec: np.ndarray, wl: np.ndarray, gauss_fwhm: float
    ) -> np.ndarray:
        """Convolve spectrum with Gaussian kernel to smooth / blur spectral details

        Arguments:
        ----------
        spec: np.ndarray
            Input spectrum, shape (n_bands,)
        wl: np.ndarray, nanometers
            Wavelengths corresponding to each spectrum value, shape (n_bands,)
        gauss_fwhm
            "Full width half maximum" (FWHM) of Gaussian kernel used for
            smoothin the spectrum. FWHM is the width of the kernel in nanometers
            at the level where the kernel values are half of the maximum value.

        Returns:
        --------
        spec_filtered: np.ndarray
            Filtered / smoothed version of spec, with same dimensions

        Notes:
        ------
        - When the kernel extends outside the data while filtering, edges are handled
        by repeating the nearest sampled value (edge value).

        """
        sigma_wl = gauss_fwhm * 0.588705  # sigma = FWHM / 2*sqrt(2*ln(2))
        dwl = np.mean(np.diff(wl))  # Downwelling wavelength sampling dist.
        sigma_pix = sigma_wl / dwl  # Sigma in units of spectral samples
        spec_filtered = gaussian_filter1d(input=spec, sigma=sigma_pix, mode="nearest")
        return spec_filtered

    @staticmethod
    def interpolate_irrad_to_image_wl(
        irrad_spec: np.ndarray,
        irrad_wl: np.ndarray,
        image_wl: np.ndarray,
    ) -> np.ndarray:
        """Interpolate downwelling spectrum to image wavelengths"""
        return np.interp(x=image_wl, xp=irrad_wl, fp=irrad_spec)

    def convert_radiance_image_to_reflectance(
        self,
        rad_image: np.ndarray,
        rad_wl: np.ndarray,
        irrad_spec: np.ndarray,
        irrad_wl: np.ndarray,
        convolve_irradiance_with_gaussian: bool = True,
        gauss_fwhm: float = 3.5,  # TODO: Find "optimal" default value for Pika-L
        smooth_with_savitsky_golay=False,
    ):
        """Convert radiance image to reflectance using downwelling spectrum

        Arguments:
        ----------
        rad_image:
            Spectral radiance image in units of microflicks = 10e-5 W/(sr*m2*nm)
            Shape (n_lines, n_samples, n_bands)
        raw_wl:
            Wavelengths (in nanometers) corresponding to each band in rad_image
        irrad_spec:
            Spectral irradiance in units of W/(m2*nm)
        irrad_wl:
            Wavelengths (in nanometers) corresponding to each band in irrad_spec

        Keyword arguments:
        ------------------
        convolve_irradiance_with_gaussian: bool
            Indicate if irradiance spectrum should be smoothed with Gaussian kernel.
            This may be useful if irradiance is measured with a higher spectral
            resolution than radiance, and thus has sharper "spikes".
        gauss_fwhm: float
            Full-width-half-maximum for Gaussian kernel, in nanometers.
            Only used if convolve_irradiance_with_gaussian==True
        smooth_with_savitsky_golay: bool
            Whether to smooth the reflectance spectra using a Savitzky-Golay filter

        """

        # Check that spectrum is 1D, then expand to 3D for broadcasting
        irrad_spec = np.squeeze(irrad_spec)
        assert irrad_spec.ndim == 1

        # Limit output wavelength range
        valid_image_wl_ind = (rad_wl >= self.wl_min) & (rad_wl <= self.wl_max)
        rad_wl = rad_wl[valid_image_wl_ind]
        rad_image = rad_image[:, :, valid_image_wl_ind]

        # Make irradiance spectrum compatible with image
        irrad_spec = irrad_spec * 100_000  # Convert from W/(m2*nm) to uW/(cm2*um)
        if convolve_irradiance_with_gaussian:
            irrad_spec = self.conv_spec_with_gaussian(irrad_spec, irrad_wl, gauss_fwhm)
        irrad_spec = self.interpolate_irrad_to_image_wl(irrad_spec, irrad_wl, rad_wl)
        irrad_spec = np.expand_dims(irrad_spec, axis=(0, 1))

        # Convert to reflectance, assuming Lambertian (perfectly diffuse) surface
        refl_image = np.pi * (
            rad_image.astype(np.float32) / irrad_spec.astype(np.float32)
        )
        refl_wl = rad_wl

        # Spectral smoothing (optional)
        if smooth_with_savitsky_golay:
            refl_image = savitzky_golay_filter(refl_image)

        return refl_image, refl_wl, irrad_spec

    def convert_radiance_file_to_reflectance(
        self,
        radiance_image_header: Union[Path, str],
        irradiance_header: Union[Path, str],
        reflectance_image_header: Union[Path, str],
        use_mean_ref_irrad_spec: bool = False,
        **kwargs,
    ):
        """

        Arguments:
        radiance_image_header:
            Path to header file for radiance image.
        irradiance_header:
            Path to ENVI file containing irradiance measurement
            corresponding to radiance image file.
            Not used if use_mean_ref_irrad_spec is True - in this
            case, it can be set to None.
        reflectance_image_header:
            Path to header file for (output) reflectance image.
            Binary file will be saved with same name, except .hdr extension.

        Keyword arguments:
        ------------------
        use_mean_ref_irrad_spec:
            Whether to use mean of irradiance reference spectra (see __init__)
            rather than an irradiance spectrum recorded together with the
            radiance image. This may be useful in cases where the recorded
            irradiance spectra are missing, or have low quality, e.g. at low
            sun angles where movement of the UAV strongly affects the measurement.

        """
        rad_image, rad_wl, rad_meta = read_envi(radiance_image_header)
        if use_mean_ref_irrad_spec:
            if self.ref_irrad_spec_mean is None:
                raise ValueError("Missing reference irradiance spectra.")
            irrad_spec = self.ref_irrad_spec_mean
            irrad_wl = self.ref_irrad_spec_wl
        else:
            irrad_spec, irrad_wl, _ = read_envi(irradiance_header)
        refl_im, refl_wl, _ = self.convert_radiance_image_to_reflectance(
            rad_image, rad_wl, irrad_spec, irrad_wl
        )
        wl_str = wavelength_array_to_header_string(refl_wl)
        refl_meta = rad_meta
        refl_meta["wavelength"] = wl_str
        save_envi(reflectance_image_header, refl_im, refl_meta)


class GlintCorrector:
    def __init__(
        self, method: str = "flat_spec", smooth_with_savitsky_golay: bool = True
    ):
        """Initialize glint corrector

        Keyword arguments:
        method:
            Method for removing / correcting for sun/sky glint.
            Currently, only 'flat_spec' is implemented.
        smooth_with_savitsky_golay: bool
            Whether to smooth glint corrected images using a
            Savitsky-Golay filter.

        """
        self.method = method
        self.smooth_with_savitsky_golay = smooth_with_savitsky_golay

    @staticmethod
    def get_nir_ind(
        wl,
        nir_band: tuple[float] = (740.0, 805.0),
        nir_ignore_band: tuple[float] = (753.0, 773.0),
    ):
        """Get indices of NIR band

        Keyword arguments:
        ------------------
        nir_band: tuple[float, float]
            Lower and upper edge of near-infrared (NIR) band.
        nir_ignore_band: tuple [float, float]
            Lower and upper edge of band to ignore (not include in indices)
            with nir_band. Default value corresponds to O2 absorption band
            around 760 nm.

        Notes:
        ------
        - Default values are at relatively short wavelengths (just above visible)
        in order to generate a NIR image with high signal-no-noise level.
        The default nir_ignore_band

        """
        nir_ind = (wl >= nir_band[0]) & (wl <= nir_band[1])
        ignore_ind = (wl >= nir_ignore_band[0]) & (wl <= nir_ignore_band[1])
        nir_ind = nir_ind & ~ignore_ind
        return nir_ind

    def remove_glint_flat_spec(
        self, refl_image: np.ndarray, refl_wl: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Remove sun and sky glint from image assuming flat glint spectrum

        Arguments:
        ----------
        refl_image:
            Reflectance image, shape (n_lines, n_samples, n_bands)
        refl_wl:
            Wavelengths (in nm) for each band in refl_image

        Returns:
        --------
        refl_image_glint_corr:
            Glint corrected reflectance image, same shape as refl_image.
            The mean NIR value is subtracted from each spectrum in the input
            image. Thus, only the spectral baseline / offset is changed -
            the original spectral shape is maintained.

        Notes:
        - The glint correction is based on the assumption that there is
        (approximately) no water-leaving radiance in the NIR spectral region.
        This is often the case, since NIR light is very effectively
        absorbed by water.
        - The glint correction also assumes that the sun and sky glint
        reflected in the water surface has a flat spectrum, so that the
        glint in the visible region can be estimated from the glint in the
        NIR region. This is usually _not_ exactly true, but the assumption
        can be "close enough" to still produce useful results.
        """
        nir_ind = self.get_nir_ind(refl_wl, **kwargs)
        nir_im = np.mean(refl_image[:, :, nir_ind], axis=2, keepdims=True)
        refl_image_glint_corr = refl_image - nir_im

        if self.smooth_with_savitsky_golay:
            refl_image_glint_corr = savitzky_golay_filter(
                refl_image_glint_corr, **kwargs
            )

        return refl_image_glint_corr

    def glint_correct_image_file(self, image_path, glint_corr_image_path, **kwargs):
        """Read reflectance file, apply glint correction, and save result"""
        if self.method == "flat_spec":
            image, wl, metadata = read_envi(image_path)
            glint_corr_image = self.remove_glint_flat_spec(image, wl, **kwargs)
            save_envi(glint_corr_image_path, glint_corr_image, metadata)
        else:
            raise ValueError(f"Glint correction method {self.method=} invalid.")


class ImageFlightMetadata:
    """

    Attributes:
    -----------
    u_alongtrack:
        Unit vector (easting, northing) pointing along flight direction
    u_crosstrack:
        Unit vector (easting, northing) pointing left relative to
        flight direction. The direction is chosen to match that of
        the image coordinate system: Origin in upper left corner,
        down (increasing line number) corresponds to positive along-track
        direction, right (increasing sample number) corresponds to
        positive cross-track direction.


    """

    def __init__(
        self,
        imu_data: dict,
        image_shape: tuple[int],
        camera_opening_angle: float = 36.5,
        pitch_offset: float = 0.0,
        roll_offset: float = 0.0,
        assume_square_pixels: bool = True,
        altitude_offset: float = 0.0,
        **kwargs,
    ):
        """

        Arguments:
        ----------
        imu_data: dict
            Dictionary with imu_data, as formatted by ImuDataParser
        image_shape: tuple[int]
            Shape of image, typically (n_lines,n_samples,n_bands)


        Keyword arguments:
        ------------------
        camera_opening_angle: float (degrees)
            Full opening angle of camera, in degrees.
            Corresponds to angle between rays hitting leftmost and
            rightmost pixels of image.
        pitch_offset: float (degrees)
            How much forward the camera is pointing relative to nadir
        roll_offset: float (degrees)
            How much to the right ("right wing up") the camera is pointing
            relative to nadir.
        assume_square_pixels: bool
            Whether to assume that the original image was acquired with
            flight parameters (flight speed, frame rate, altitude)
            that would produce square pixels. If true, the altitude of the
            camera is estimated from the shape of the image and the (along-track)
            swath length. This can be useful in cases where absolute altitude
            measurement of the camera IMU is not very accurate.
        altitude_offset:
            Offset added to the estimated altitude. If the UAV was higher
            in reality than that estimated by the ImageFlightMetadata
            object, add a positive altitude_offset.

        """

        # Set input attributes
        self.imu_data = imu_data
        self.image_shape = image_shape[0:2]
        self.camera_opening_angle = camera_opening_angle * (np.pi / 180)
        self.pitch_offset = pitch_offset * (np.pi / 180)
        self.roll_offset = roll_offset * (np.pi / 180)
        self.altitude_offset = altitude_offset

        # Get UTM coordinates and CRS code
        utm_x, utm_y, utm_epsg = convert_long_lat_to_utm(
            imu_data["longitude"], imu_data["latitude"]
        )
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.camera_origin = np.array([utm_x[0], utm_y[0]])
        self.utm_epsg = utm_epsg

        # Time-related attributes
        t_total, dt = self._calc_time_attributes()
        self.t_total = t_total
        self.dt = dt

        # Along-track properties
        v_at, u_at, gsd_at, sl = self._calc_alongtrack_properties()
        self.v_alongtrack = v_at
        self.u_alongtrack = u_at
        self.gsd_alongtrack = gsd_at
        self.swath_length = sl

        # Altitude
        self.mean_altitude = self._calc_mean_altitude(assume_square_pixels)

        # Cross-track properties
        u_ct, sw, gsd_ct = self._calc_crosstrack_properties()
        self.u_crosstrack = u_ct
        self.swath_width = sw
        self.gsd_crosstrack = gsd_ct

        # Image origin (image transform offset)
        self.image_origin = self._calc_image_origin()

    def _calc_time_attributes(self):
        """Calculate time duration and sampling interval of IMU data"""
        t = np.array(self.imu_data["time"])
        dt = np.mean(np.diff(t))
        t_total = len(t) * dt
        return t_total, dt

    def _calc_alongtrack_properties(self):
        """Calculate along-track velocity, gsd, and swath length"""
        vx_alongtrack = (self.utm_x[-1] - self.utm_x[0]) / self.t_total
        vy_alongtrack = (self.utm_y[-1] - self.utm_y[0]) / self.t_total
        v_alongtrack = np.array((vx_alongtrack, vy_alongtrack))
        v_alongtrack_abs = np.linalg.norm(v_alongtrack)
        u_alongtrack = v_alongtrack / v_alongtrack_abs

        swath_length = self.t_total * v_alongtrack_abs
        gsd_alongtrack = self.dt * v_alongtrack_abs

        return v_alongtrack, u_alongtrack, gsd_alongtrack, swath_length

    def _calc_mean_altitude(self, assume_square_pixels):
        """Calculate mean altitude of uav during imaging

        Arguments:
        assume_square_pixels: bool
            If true, the across-track sampling distance is assumed to
            be equal to the alongtrack sampling distance. The altitude
            is calculated based on this and the number of cross-track samples.
            If false, the mean of the altitude values from the imu data
            is used. In both cases, the altitude offset is added.
        """
        if assume_square_pixels:
            swath_width = self.gsd_alongtrack * self.image_shape[1]
            altitude = swath_width / (2 * np.tan(self.camera_opening_angle / 2))
        else:
            altitude = np.mean(self.imu_data["altitude"])
        return altitude + self.altitude_offset

    def _calc_crosstrack_properties(self):
        """Calculate cross-track unit vector, swath width and sampling distance"""
        u_crosstrack = np.array(
            [-self.u_alongtrack[1], self.u_alongtrack[0]]
        )  # Rotate 90 CCW
        swath_width = 2 * self.mean_altitude * np.tan(self.camera_opening_angle / 2)
        gsd_crosstrack = swath_width / self.image_shape[1]
        return u_crosstrack, swath_width, gsd_crosstrack

    def _calc_image_origin(self):
        """Calculate location of image pixel (0,0) in georeferenced coordinates"""
        alongtrack_offset = (
            self.mean_altitude * np.tan(self.pitch_offset) * self.u_alongtrack
        )
        crosstrack_offset = (
            self.mean_altitude * np.tan(self.roll_offset) * self.u_crosstrack
        )
        # NOTE: Signs of cross-track elements in equation below are "flipped"
        # because UTM coordinate system is right-handed and image coordinate
        # system is left-handed. If the camera_origin is in the middle of the
        # top line of the image, u_crosstrack points away from the image
        # origin (line 0, sample 0).
        image_origin = (
            self.camera_origin
            - 0.5 * self.swath_width * self.u_crosstrack  # Edge of swath
            + crosstrack_offset
            - alongtrack_offset
        )
        return image_origin

    def get_image_transform(self, ordering="alphabetical"):
        """Get 6-element affine transform for image

        Keyword arguments:
        ------------------
        ordering: ['alphabetical','worldfile']
            If 'alphabetical', return A,B,C,D,E,F
            If 'worldfile', return A,D,B,E,C,F
            See https://en.wikipedia.org/wiki/World_file
        """
        A, D = self.gsd_crosstrack * self.u_crosstrack
        B, E = self.gsd_alongtrack * self.u_alongtrack
        C, F = self.image_origin

        if ordering == "alphabetical":
            return A, B, C, D, E, F
        elif ordering == "worldfile":
            return A, D, B, E, C, F
        else:
            error_msg = f"Invalid ordering argument {ordering}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class SimpleGeoreferencer:
    def georeference_hyspec_save_geotiff(
        self,
        image_path: Union[Path, str],
        imudata_path: Union[Path, str],
        geotiff_path: Union[Path, str],
        rgb_only: bool = True,
        nodata_value: int = -9999,
        **kwargs,
    ):
        """Georeference hyperspectral image and save as GeoTIFF

        Arguments:
        ----------
        image_path:
            Path to hyperspectral image header.
        imudata_path:
            Path to JSON file containing IMU data.
        geotiff_path:
            Path to (output) GeoTIFF file.
        rgb_only: bool
            Whether to only output an RGB version of the hyperspectral image.
            If false, the entire hyperspectral image is used. Note that
            this typically creates very large files that some programs
            (e.g. QGIS) can struggle to read.
        nodata_value:
            Value to insert in place of invalid pixels.
            Pixels which contain "all zeros" are considered invalid.
        """
        image, wl, _ = read_envi(image_path)
        if rgb_only:
            image, wl = rgb_subset_from_hsi(image, wl)
        self.insert_image_nodata_value(image, nodata_value)
        geotiff_profile = self.create_geotiff_profile(
            image, imudata_path, nodata_value=nodata_value, **kwargs
        )

        self.write_geotiff(geotiff_path, image, wl, geotiff_profile)

    @staticmethod
    def move_bands_axis_first(image):
        """Move spectral bands axis from position 2 to 0"""
        return np.moveaxis(image, 2, 0)

    @staticmethod
    def insert_image_nodata_value(image, nodata_value):
        """Insert nodata values in image (in-place)

        Arguments:
        ----------
        image:
            3D image array ordered as (lines, samples, bands)
            Pixels where every band value is equal to zero
            are interpreted as invalid (no data).
        nodata_value:
            Value to insert in place of invalid data.
        """
        nodata_mask = np.all(image == 0, axis=2)
        image[nodata_mask] = nodata_value

    @staticmethod
    def create_geotiff_profile(
        image: np.ndarray,
        imudata_path: Union[Path, str],
        nodata_value: int = -9999,
        **kwargs,
    ):
        """Create profile for writing image as geotiff using rasterio

        Arguments:
        ----------
        image:
            3D image array ordered, shape (n_lines,n_samples,n_bands).
        imudata_path:
            Path to JSON file containing IMU data for image

        Keyword arguments:
        ------------------
        nodata_value:

        """
        imu_data = ImuDataParser.read_imu_json_file(imudata_path)
        image_flight_meta = ImageFlightMetadata(imu_data, image.shape, **kwargs)
        transform = Affine(*image_flight_meta.get_image_transform())
        crs_epsg = image_flight_meta.utm_epsg

        profile = DefaultGTiffProfile()
        profile.update(
            height=image.shape[0],
            width=image.shape[1],
            count=image.shape[2],
            dtype=str(image.dtype),
            crs=CRS.from_epsg(crs_epsg),
            transform=transform,
            nodata=nodata_value,
        )

        return profile

    def write_geotiff(
        self,
        geotiff_path: Union[Path, str],
        image: np.ndarray,
        wavelengths: np.ndarray,
        geotiff_profile: dict,
    ):
        """Write image as GeoTIFF

        Arguments:
        ----------
        geotiff_path: Path | str
            Path to (output) GeoTIFF file
        image: np.ndarray
            Image to write, shape (n_lines, n_samples, n_bands)
        wavelengths: np.ndarray
            Wavelengths (in nm) corresponding to each image band.
            The wavelengths are used to set the descption of each band
            in the GeoTIFF file.

        # Notes:
        --------
        - Rasterio / GDAL required the image to be ordered "bands first",
        e.g. shape (bands, lines, samples). However, the default used by e.g.
        the spectral library is (lines, samples, bands), and this convention
        should be used consistenly to avoid bugs. This function moves the band
        axis directly before writing.

        """
        image = self.move_bands_axis_first(image)  # Band ordering requred by GeoTIFF
        band_names = [f"{wl:.3f}" for wl in wavelengths]
        with rasterio.Env():
            with rasterio.open(geotiff_path, "w", **geotiff_profile) as dataset:
                if band_names is not None:
                    for i in range(dataset.count):
                        dataset.set_band_description(i + 1, band_names[i])
                dataset.write(image)

    @staticmethod
    def update_image_file_transform(
        geotiff_path: Union[Path, str], imu_data_path: Union[Path, str], **kwargs
    ):
        """Update the affine transform of an image

        Arguments:
        ----------
        geotiff_path:
            Path to existing GeoTIFF file.
        imu_data_path:
            Path to JSON file with IMU data.

        Keyword arguments:
        ------------------
        **kwargs:
            Keyword arguments are passed along to create an ImageFlightMetadata object.
            Options include e.g. 'altitude_offset'. This can be useful in case
            the shape of the existing GeoTIFF indicates that some adjustments
            should be made to the image transform (which can be re-generated using
            an ImageFlightMetadata object).

        References:
        -----------
        - https://rasterio.readthedocs.io/en/latest/api/rasterio.rio.edit_info.html
        """
        imu_data = ImuDataParser.read_imu_json_file(imu_data_path)
        with rasterio.open(geotiff_path, "r") as dataset:
            im_width = dataset.width
            im_height = dataset.height
        image_flight_meta = ImageFlightMetadata(
            imu_data, image_shape=(im_height, im_width), **kwargs
        )
        new_transform = image_flight_meta.get_image_transform()
        rio_cmd = [
            "rio",
            "edit-info",
            "--transform",
            str(list(new_transform)),
            str(geotiff_path),
        ]
        subprocess.run(rio_cmd)


class PipelineProcessor:
    def __init__(self, dataset_dir: Union[Path, str]):
        """Create a pipeline for processing all data in a dataset

        Arguments:
        ----------
        dataset_dir:
            Path to folder containing dataset. The name of the folder
            will be used as the "base name" for all processed files.
            The folder should contain at least two subfolders:
            - 0_raw: Contains all raw images as saved by Resonon airborne system.
            - calibration: Contains Resonon calibration files for
            camera (*.icp) and downwelling irradiance sensor (*.dcp)

        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_base_name = dataset_dir.name
        self.raw_dir = dataset_dir / "0_raw"
        self.radiance_dir = dataset_dir / "1_radiance"
        self.reflectance_dir = dataset_dir / "2a_reflectance"
        self.reflectance_gc_dir = dataset_dir / "2b_reflectance_gc"
        self.reflectance_gc_rgb_dir = dataset_dir / "2b_reflectance_gc" / "rgb_geotiff"
        self.mosaic_dir = dataset_dir / "mosaics"
        self.calibration_dir = dataset_dir / "calibration"
        self.logs_dir = dataset_dir / "logs"

        if not self.raw_dir.exists():
            raise FileNotFoundError(f'Folder "0_raw" not found in {dataset_dir}')
        if not self.calibration_dir.exists():
            raise FileNotFoundError(f'Folder "calibration" not found in {dataset_dir}')

        # Get calibration file paths
        self.radiance_calibration_file = self._get_radiance_calibration_path()
        self.irradiance_calibration_file = self._get_irradiance_calibration_path()

        # Search for raw files, sort and validate
        self.raw_image_paths = list(self.raw_dir.rglob("*.bil.hdr"))
        self.raw_image_paths = sorted(self.raw_image_paths, key=self.get_image_number)
        times_paths, lcf_paths = self._validate_raw_files()
        self.times_paths = times_paths
        self.lcf_paths = lcf_paths

        # Search for raw irradiance spectrum files (not always present)
        self.raw_spec_paths = self._get_raw_spectrum_paths()

        # Create "base" file names numbered from 0
        self.base_file_names = self._create_base_file_names()

        # Create lists of processed file paths
        proc_file_paths = self._create_processed_file_paths()
        self.rad_im_paths = proc_file_paths["radiance"]
        self.irrad_spec_paths = proc_file_paths["irradiance"]
        self.imu_data_paths = proc_file_paths["imudata"]
        self.refl_im_paths = proc_file_paths["reflectance"]
        self.refl_gc_im_paths = proc_file_paths["reflectance_gc"]
        self.refl_gc_rgb_paths = proc_file_paths["reflectance_gc_rgb"]

        # Configure logging
        self._configure_file_logging()

    def _configure_file_logging(self):
        """Configure logging for pipeline"""

        # Create log file path
        self.logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_name = f"{timestamp}_{self.dataset_base_name}.log"
        log_path = self.logs_dir / log_file_name

        # Add file handler to module-level logger
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info("File logging initialized.")

    def _validate_raw_files(self):
        """Check that all expected raw files exist

        Returns:
        --------
        times_paths, lcf_paths: list[Path]
            Lists of paths to *.times and *.lcf files for every valid raw file
        """
        times_paths = []
        lcf_paths = []
        for raw_image_path in list(self.raw_image_paths):  # Use list() to copy
            file_base_name = raw_image_path.name.split(".")[0]
            binary_im_path = raw_image_path.parent / (file_base_name + ".bil")
            times_path = raw_image_path.parent / (file_base_name + ".bil.times")
            lcf_path = raw_image_path.parent / (file_base_name + ".lcf")
            if (
                not (binary_im_path.exists())
                or not (times_path.exists())
                or not (lcf_path.exists())
            ):
                warnings.warn(
                    f"Set of raw files for image {raw_image_path} is incomplete."
                )
                self.raw_image_paths.remove(raw_image_path)
            else:
                times_paths.append(times_path)
                lcf_paths.append(lcf_path)
        return times_paths, lcf_paths

    @staticmethod
    def get_image_number(raw_image_path):
        """Get image number from raw image

        Notes:
        ------
        Raw files are numbered sequentially, but the numbers are not
        zero-padded. This can lead to incorrect sorting of the images, e.g.
        ['im_1','im_2','im_10'] (simplified names for example) are sorted
        ['im_1','im_10','im_2']. By extracting the numbers from filenames
        of raw files and sorting explicitly on these (as integers),
        correct ordering can be achieved.
        """
        image_file_stem = raw_image_path.name.split(".")[0]
        image_number = image_file_stem.split("_")[-1]
        return int(image_number)

    def _create_base_file_names(self):
        """Create numbered base names for processed files"""
        base_file_names = [
            f"{self.dataset_base_name}_{i:03d}"
            for i in range(len(self.raw_image_paths))
        ]
        return base_file_names

    def _create_processed_file_paths(self):
        """Define default subfolders for processed files"""
        file_paths = {
            "radiance": [],
            "irradiance": [],
            "imudata": [],
            "reflectance": [],
            "reflectance_gc": [],
            "reflectance_gc_rgb": [],
        }

        for base_file_name in self.base_file_names:
            rad_path = self.radiance_dir / (base_file_name + "_radiance.bip.hdr")
            file_paths["radiance"].append(rad_path)
            irs_path = self.radiance_dir / (base_file_name + "_irradiance.spec.hdr")
            file_paths["irradiance"].append(irs_path)
            imu_path = self.radiance_dir / (base_file_name + "_imudata.json")
            file_paths["imudata"].append(imu_path)
            refl_path = self.reflectance_dir / (base_file_name + "_reflectance.bip.hdr")
            file_paths["reflectance"].append(refl_path)
            rgc_path = self.reflectance_gc_dir / (
                base_file_name + "_reflectance_gc.bip.hdr"
            )
            file_paths["reflectance_gc"].append(rgc_path)
            rgc_rgb_path = self.reflectance_gc_rgb_dir / (
                base_file_name + "_reflectance_gc_rgb.tiff"
            )
            file_paths["reflectance_gc_rgb"].append(rgc_rgb_path)
        return file_paths

    def _get_raw_spectrum_paths(self):
        """Search for raw files matching Resonon default naming"""
        spec_paths = []
        for raw_image_path in self.raw_image_paths:
            spec_base_name = raw_image_path.name.split("_Pika_L")[0]
            image_number = self.get_image_number(raw_image_path)
            spec_binary = (
                raw_image_path.parent
                / f"{spec_base_name}_downwelling_{image_number}_pre.spec"
            )
            spec_header = raw_image_path.parent / (spec_binary.name + ".hdr")
            if spec_binary.exists() and spec_header.exists():
                spec_paths.append(spec_header)
            else:
                spec_paths.append(None)
        return spec_paths

    def _get_radiance_calibration_path(self):
        """Search for radiance calibration file (*.icp)"""
        icp_files = list(self.calibration_dir.glob("*.icp"))
        if len(icp_files) == 1:
            return icp_files[0]
        elif len(icp_files) == 0:
            raise FileNotFoundError(
                f"No radiance calibration file (*.icp) found in {self.calibration_dir}"
            )
        else:
            raise ValueError(
                f"More than one radiance calibration file (*.icp) found in {self.calibration_dir}"
            )

    def _get_irradiance_calibration_path(self):
        """Search for irradiance calibration file (*.dcp)"""
        dcp_files = list(self.calibration_dir.glob("*.dcp"))
        if len(dcp_files) == 1:
            return dcp_files[0]
        elif len(dcp_files) == 0:
            raise FileNotFoundError(
                f"No irradiance calibration file (*.dcp) found in {self.calibration_dir}"
            )
        else:
            raise ValueError(
                f"More than one irradiance calibration file (*.dcp) found in {self.calibration_dir}"
            )

    def convert_raw_images_to_radiance(self, **kwargs):
        """Convert raw hyperspectral images (DN) to radiance (microflicks)"""
        logger.info("---- RADIANCE CONVERSION ----")
        self.radiance_dir.mkdir(exist_ok=True)
        radiance_converter = RadianceConverter(self.radiance_calibration_file)
        for raw_image_path, radiance_image_path in zip(
            self.raw_image_paths, self.rad_im_paths
        ):
            logger.info(f"Converting {raw_image_path.name} to radiance")
            try:
                radiance_converter.convert_raw_file_to_radiance(
                    raw_image_path, radiance_image_path
                )
            except Exception:
                logger.warning(
                    f"Error occured while processing {raw_image_path}", exc_info=True
                )
                logger.warning("Skipping file")

    def convert_raw_spectra_to_irradiance(self, **kwargs):
        """Convert raw spectra (DN) to irradiance (W/(m2*nm))"""
        logger.info("---- IRRADIANCE CONVERSION ----")
        self.radiance_dir.mkdir(exist_ok=True)
        irradiance_converter = IrradianceConverter(self.irradiance_calibration_file)
        for raw_spec_path, irrad_spec_path in zip(
            self.raw_spec_paths, self.irrad_spec_paths
        ):
            if raw_spec_path is not None:
                logger.info(
                    f"Converting {raw_spec_path.name} to downwelling irradiance"
                )
                try:
                    irradiance_converter.convert_raw_file_to_irradiance(
                        raw_spec_path, irrad_spec_path
                    )
                except Exception:
                    logger.error(
                        f"Error occured while processing {raw_spec_path}", exc_info=True
                    )
                    logger.error("Skipping file")

    def calibrate_irradiance_wavelengths(self, **kwargs):
        """Calibrate irradiance wavelengths using Fraunhofer absorption lines"""
        logger.info("---- IRRADIANCE WAVELENGTH CALIBRATION ----")
        if not (self.radiance_dir.exists()):
            raise FileNotFoundError(
                "Radiance folder with irradiance spectra does not exist"
            )
        wavelength_calibrator = WavelengthCalibrator()
        irradiance_spec_paths = list(self.radiance_dir.glob("*.spec.hdr"))
        if irradiance_spec_paths:
            wavelength_calibrator.fit_batch(irradiance_spec_paths)
            for irradiance_spec_path in irradiance_spec_paths:
                logger.info(f"Calibrating wavelengths for {irradiance_spec_path.name}")
                try:
                    wavelength_calibrator.update_header_wavelengths(
                        irradiance_spec_path
                    )
                except Exception:
                    logger.error(
                        f"Error occured while processing {irradiance_spec_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def parse_and_save_imu_data(self, **kwargs):
        """Parse *.lcf and *.times files with IMU data and save as JSON"""
        logger.info("---- IMU DATA PROCESSING ----")
        self.radiance_dir.mkdir(exist_ok=True)
        imu_data_parser = ImuDataParser()
        for lcf_path, times_path, imu_data_path in zip(
            self.lcf_paths, self.times_paths, self.imu_data_paths
        ):
            logger.info(f"Processing IMU data from {lcf_path.name}")
            try:
                imu_data_parser.read_and_save_imu_data(
                    lcf_path, times_path, imu_data_path
                )
            except Exception:
                logger.error(
                    f"Error occured while processing {lcf_path}", exc_info=True
                )
                logger.error("Skipping file")

    def convert_radiance_images_to_reflectance(self, **kwargs):
        """Convert radiance images (microflicks) to reflectance (unitless)"""
        logger.info("---- REFLECTANCE CONVERSION ----")
        self.reflectance_dir.mkdir(exist_ok=True)
        reflectance_converter = ReflectanceConverter(
            irrad_spec_paths=self.irrad_spec_paths
        )

        if all([not rp.exists() for rp in self.rad_im_paths]):
            warnings.warn(f"No radiance images found in {self.radiance_dir}")
        if all([not irp.exists() for irp in self.irrad_spec_paths]):
            warnings.warn(f"No irradiance spectra found in {self.radiance_dir}")

        for rad_path, irrad_path, refl_path in zip(
            self.rad_im_paths, self.irrad_spec_paths, self.refl_im_paths
        ):
            if rad_path.exists() and irrad_path.exists():
                logger.info(f"Converting {rad_path.name} to reflectance.")
                try:
                    reflectance_converter.convert_radiance_file_to_reflectance(
                        rad_path, irrad_path, refl_path, **kwargs
                    )
                except Exception:
                    logger.error(
                        f"Error occured while processing {rad_path}", exc_info=True
                    )
                    logger.error("Skipping file")

    def glint_correct_reflectance_images(self, **kwargs):
        """Correct for sun and sky glint in reflectance images"""
        logger.info("---- GLINT CORRECTION ----")
        self.reflectance_gc_dir.mkdir(exist_ok=True)
        glint_corrector = GlintCorrector()

        if all([not rp.exists() for rp in self.refl_im_paths]):
            warnings.warn(f"No reflectance images found in {self.reflectance_dir}")

        for refl_path, refl_gc_path in zip(self.refl_im_paths, self.refl_gc_im_paths):
            if refl_path.exists():
                logger.info(f"Applying glint correction to {refl_path.name}.")
                try:
                    glint_corrector.glint_correct_image_file(refl_path, refl_gc_path)
                except Exception:
                    logger.error(
                        f"Error occured while glint correcting {refl_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def georeference_glint_corrected_reflectance(self, **kwargs):
        """Create georeferenced GeoTIFF versions of glint corrected reflectance"""
        logger.info("---- GEOREFERENCING GLINT CORRECTED REFLECTANCE ----")
        self.reflectance_gc_rgb_dir.mkdir(exist_ok=True)
        georeferencer = SimpleGeoreferencer()

        if all([not rp.exists() for rp in self.refl_gc_im_paths]):
            warnings.warn(f"No reflectance images found in {self.reflectance_gc_dir}")

        for refl_gc_path, imu_data_path, geotiff_path in zip(
            self.refl_gc_im_paths, self.imu_data_paths, self.refl_gc_rgb_paths
        ):
            if refl_gc_path.exists() and imu_data_path.exists():
                logger.info(
                    f"Georeferencing and exporting RGB version of {refl_gc_path.name}."
                )
                try:
                    georeferencer.georeference_hyspec_save_geotiff(
                        refl_gc_path,
                        imu_data_path,
                        geotiff_path,
                        rgb_only=True,
                        **kwargs,
                    )
                except Exception:
                    logger.error(
                        f"Error occured while georeferencing RGB version of {refl_gc_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def mosaic_geotiffs(self, mosaic_path=None):
        """Convert set of rotated geotiffs into single mosaic with overviews

        Arguments:
        mosaic_path
            Path to output mosaic. If None, default dataset folder and name
            is used.

        # Explanation of gdalwarp arguments used:
        -overwrite:
            Overwrite existing files without error / warning
        -q:
            Suppress GDAL output (quiet)
        -r near:
            Resampling method: Nearest neighbor
        -of GTiff:
            Output format: GeoTiff

        # Explanation of gdaladdo agruments:
        -r average
            Use averaging when resampling to lower spatial resolution
        -q
            Suppress output (be quiet)
        """
        logger.info(f"Mosaicing GeoTIFFs in {self.reflectance_gc_rgb_dir}")
        self.mosaic_dir.mkdir(exist_ok=True)
        if mosaic_path is None:
            mosaic_path = self.mosaic_dir / (self.dataset_base_name + "_rgb.tiff")

        # Run as subprocess without invoking shell. Note input file unpacking.
        gdalwarp_args = [
            "gdalwarp",
            "-overwrite",
            "-q",
            "-r",
            "near",
            "-of",
            "GTiff",
            *[str(p) for p in self.refl_gc_rgb_paths if p.exists()],
            str(mosaic_path),
        ]
        subprocess.run(gdalwarp_args)
        # NOTE: Example code below runs GDAL from within shell - avoid if possible
        # geotiff_search_string = str(self.reflectance_gc_rgb_dir / '*.tiff')
        # gdalwarp_cmd = f"gdalwarp -overwrite -q -r near -of GTiff {geotiff_search_string} {mosaic_path}"
        # subprocess.run(gdalwarp_cmd,shell=True)

        # Add image pyramids to file
        logger.info(f"Adding image pyramids to mosaic {mosaic_path}")
        gdaladdo_args = ["gdaladdo", "-q", "-r", "average", str(mosaic_path)]
        subprocess.run(gdaladdo_args)

    def update_geotiff_transforms(self, **kwargs):
        """Batch update GeoTIFF transforms

        Image affine transforms are re-calculated based on IMU data and
        (optional) keyword arguments.

        Keyword arguments:
        ------------------
        **kwargs:
            keyword arguments accepted by ImageFlightSegment, e.g.
            "altitude_offset".

        """
        logger.info("---- UPDATING GEOTIFF AFFINE TRANSFORMS ----")
        georeferencer = SimpleGeoreferencer()

        if all([not gtp.exists() for gtp in self.refl_gc_rgb_paths]):
            warnings.warn(f"No GeoTIFF images found in {self.reflectance_gc_rgb_dir}")

        for imu_data_path, geotiff_path in zip(
            self.imu_data_paths, self.refl_gc_rgb_paths
        ):
            if imu_data_path.exists() and geotiff_path.exists():
                logger.info(f"Updating transform for {geotiff_path.name}.")
                try:
                    georeferencer.update_image_file_transform(
                        geotiff_path,
                        imu_data_path,
                        **kwargs,
                    )
                except Exception:
                    logger.error(
                        f"Error occured while updating transform for {geotiff_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def run(
        self,
        convert_raw_images_to_radiance=True,
        convert_raw_spectra_to_irradiance=True,
        calibrate_irradiance_wavelengths=True,
        parse_imu_data=True,
        convert_radiance_to_reflectance=True,
        glint_correct_reflectance=True,
        geotiff_from_glint_corrected_reflectance=True,
        mosaic_geotiffs=True,
        **kwargs,
    ):
        """Run pipeline process(es)

        Keyword arguments:
        ------------------
        convert_raw_images_to_radiance,
        convert_raw_spectra_to_irradiance,
        calibrate_irradiance_wavelengths,
        parse_imu_data,
        convert_radiance_to_reflectance,
        glint_correct_reflectance,
        geotiff_from_glint_corrected_reflectance,
        mosaic_geotiffs: bool
            All keyword arguments are boolean "switches" used to indicate
            whether a process should be run. By default, all are True.
            If e.g. raw images have already been processed, specify
            convert_raw_images_to_radiance=False to avoid reprocessing.
        """
        if convert_raw_images_to_radiance:
            try:
                self.convert_raw_images_to_radiance(**kwargs)
            except Exception:
                logger.error(
                    "Error while converting raw images to radiance", exc_info=True
                )
        if convert_raw_spectra_to_irradiance:
            try:
                self.convert_raw_spectra_to_irradiance(**kwargs)
            except Exception:
                logger.error(
                    "Error while converting raw spectra to irradiance", exc_info=True
                )
        if calibrate_irradiance_wavelengths:
            try:
                self.calibrate_irradiance_wavelengths(**kwargs)
            except Exception:
                logger.error(
                    "Error while calibrating irradiance wavelengths", exc_info=True
                )

        if parse_imu_data:
            try:
                self.parse_and_save_imu_data(**kwargs)
            except Exception:
                logger.error("Error while parsing and saving IMU data", exc_info=True)
        if convert_radiance_to_reflectance:
            try:
                self.convert_radiance_images_to_reflectance(**kwargs)
            except Exception:
                logger.error(
                    "Error while converting from radiance to reflectance", exc_info=True
                )

        if glint_correct_reflectance:
            try:
                self.glint_correct_reflectance_images(**kwargs)
            except Exception:
                logger.error(
                    "Error while glint correcting reflectance images", exc_info=True
                )

        if geotiff_from_glint_corrected_reflectance:
            try:
                self.georeference_glint_corrected_reflectance(**kwargs)
            except Exception:
                logger.error(
                    "Error while georeferencing glint corrected images ", exc_info=True
                )

        if mosaic_geotiffs:
            try:
                self.mosaic_geotiffs()
            except Exception:
                logger.error("Error while mosaicing geotiffs ", exc_info=True)


if __name__ == "__main__":
    """Code to execute when running file as script.
    Mostly used for debugging.
    """
    dataset_dir = Path(
        "/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201842-se_hsi"
    )
    pl = PipelineProcessor(dataset_dir)
    pl.run(
        convert_raw_images_to_radiance=False,
        convert_raw_spectra_to_irradiance=False,
        calibrate_irradiance_wavelengths=False,
        parse_imu_data=False,
        convert_radiance_to_reflectance=False,
        glint_correct_reflectance=False,
        pitch_offset=2,
        altitude_offset=18,
        use_mean_ref_irrad_spec=True,
    )

    # dataset_dir = Path('/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201815-se_hsi')
    # pl = PipelineProcessor(dataset_dir)
    # pl.run(convert_raw_images_to_radiance=False,
    #        convert_raw_spectra_to_irradiance=False,
    #        calibrate_irradiance_wavelengths=False,
    #        parse_imu_data=False,
    #        convert_radiance_to_reflectance=False,
    #        glint_correct_reflectance=False,
    #        pitch_offset=2,
    #        altitude_offset=18,
    #        use_mean_ref_irrad_spec=True)

    # dataset_dir = Path("/media/mha114/Massimal2/seabee-minio/smola/skalmen/aerial/hsi/20230620/massimal_smola_skalmen_202306201640-nw_hsi")
    # pl = PipelineProcessor(dataset_dir)
    # pl.run(altitude_offset=-2.2, pitch_offset=3.4,)
    # pl.parse_and_save_imu_data()
    # pl.georeference_glint_corrected_reflectance()

    # pl.update_geotiff_transforms(pitch_offset=2,altitude_offset=18) # pitch offset 3 ganske ok?

    # dataset_dir = Path(
    #     "/media/mha114/Massimal2/seabee-minio/larvik/olbergholmen/aerial/hsi/20230830/massimal_larvik_olbergholmen_202308301001-south-test_hsi"
    # )
    # pl = PipelineProcessor(dataset_dir)
    # # pl.glint_correct_reflectance_images()
    # pl.georeference_glint_corrected_reflectance(
    #     altitude_offset=-2.2, pitch_offset=3.4, roll_offset=-0.0
    # )
