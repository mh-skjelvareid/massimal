# Notes on code for reading Resonon calibration files

## Email communication with Resonon
Sent email to Casey Smith at Resonon asking about how to read calibration files without running Spectronon. His answer:

"Hi Martin, 
You have access to the Python source code for the plugins.  The ones your interested in would be found here, provided you installed it in the default location.
C:\Program Files\Spectronon3\ext\plugins\cube\correct"

I lloked in the folder and found files for all the functionality under "Cube->Correct", including radiance and reflectace calculation, georectification, ++. However, the code relies on other modules which are not openly included (as *.py files). I asked Casey specifically about the "CubeVault" module used to read calibration files:

"Hi Martin, 

This is a little outside our normal support.  I can get you the CubeVault file, but that of course depends on other Resonon proprietary packages.  The .icp is just a .zip file containing a gain file and a lot of dark files.  The code finds the dark file that best matches the parameters of the file to be converted, and grabs the gain file as well. "

He also included the cubevault.py file. The file had double line spacing, I've removed most empty lines by search-and-replace with pattern ^$\n (using regular expressions). I've also created a copy of the file, with comments added as I was trying to understand the code. 


## Contents of *.icp file
I tried opening the zip file. It contains:
- 1 "gain" file (data type 5, double)
- 1 "offset" file (data type 5, double)
- A bunch of additional offset files, named e.g. "offset_600bands_4095ceiling_0gain_900samples_5shutter". Data type 12, unsigned int (native for raw files). The only parameters that are changed are gain and shutter.

Opening the different files, I find that:
- The gain "spectrum" is a smooth curve with values around 20 at the lower end (390 nm), minimum value around 5 for approx 510 nm, and values up to approx 200 for the NIR end.
- The offset "spectrum" is very noisy, with values ranging between approx 0.015 and 0.080. I find this slightly confusing - how is this combined with other offset spectra?
- The offset spectra for specific shutters and gains are integers. For low gains, all values are zero. For high gains, values are typically 2-4, with some spikes up to approx 10.
- All the data has 600 bands and 900 samples. This indicates that the camera chip has 600x900 pixels. When used, the camera gives the option of 300, 200 or 150 spectral channels. This corresponds to spectral binning of 2, 3 or 4.


## Description of radiance conversion in code
I now see that the description of radiance conversion explains the contents of the icp ("Calibration pack") file:

"Converts raw data to units of radiance (microflicks) based on the instrument's radiometric 
calibration file (.icp). This process is outlined below:

1. The .icp file contains multiple calibration files, one *gain* cube that contains 
    the pixel-by-pixel transfer from digital number to calibrated radiance and typically many 
    *offset* cubes, which are dark noise cubes taken at different integration times and gains. 
    This plugin automatically finds the best suited offset file to use (Auto Remove Dark) unless 
    the user decides to provide one. 

2. The gain and offset files are scaled to match the binning factor used in the datacube. Note 
    that because the gain file represents the inverse instrument response, it gets scaled inversely.
    
3. The gain and offset data are averaged both spatially and spectrally to match the frame 
    size of the datacube.

4. The gain and offset data are flipped spatially depending on the *flip radiometric 
    calibration* header item in the datacube. (This header item is used to track left-right edges 
    in airborne data.)

5.  The gain data is scaled by the integration and gain ratios between it and the datacube. 
    Note that gain in Resonon headers is 20 log, not 10 log.

6. The offset data is subtracted from the datacube. The result is multiplied by the gain file. 

The resulting data is in units of microflicks."

## Review of randiance conversion code
- Many variations of data are handled by the code
    - Dark current taken from calibration file vs. supplied as a separate cube
    - Different amounts of binning between calibration files and data
    - Spatial or spectral cropping
- Binning (adding) increases the value in each pixel, this needs to be taken into account when converting between digital numbers and microflicks. The gain and offset cubes are scaled according to this. The resulting variables are called gc_binned_gain and gc_binned_offset (gc for "gain corrected"?)
- Gain and offset cubes generally don't have the same shape as "normal", for example due to binning (calibration cubes have 600 bands, iamge cubes have max 300 bands). This is handled by averaging, separate from the handling gain. Averaged outputs keep the same names  (gc_binned_gain and gc_binned_offset)
- Differences in gain between calibration cubes and images are corrected with a gain factor:
    - gain_factor = (gc_shutter * gc_gain) / (data_shutter * data_gain)
    - adjusted_gain = gain_factor * gc_binned_gain