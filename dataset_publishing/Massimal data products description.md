# Massimal data products description

The organization of data products from the Massimal project is inspired by the organization of products from e.g. Sentinel-2, AVIRIS, ENMAP and PRISMA. 

## Products from other sensors
Image products and metadata are organized in similar ways, often with "level 0" corresponding to raw data, "level 1" corresponding to some type of intermediate data / radiance data, and "level 2" corresponding to surface reflectance data (level most related to physical quentities on the ground). Orthorectification is always applied at L2, but not always at L1.  

### Sentinel-2
See [Sentiwiki](https://sentiwiki.copernicus.eu/web/s2-products):
- L0: Raw, compressed image data
- L1A: Decompressed image data with "geometric model to locate any pixel in the image"
- L1B: Top-of-atmosphere radiance in sensor gemetry
- L1C: Top-of-atmosphere reflectance, georeferenced. Includes products like cloud masks 
- L2A: Atmospherically corrected surface reflectance, georeferenced to UTM/WGS84

### AVIRIS
See [AVIRIS web page](https://avirisng.jpl.nasa.gov/ang_data_pipeline.html). AVIRIS has several internal "products", but only releases two:
- L1B: "Resampled calibrated data in units of spectral radiance as well as observational geometry and illumination parameters."
- L2: "Orthocorrected and atmospherically corrected reflectance data (32-bit floating point quantities from 0 to 1) as well as retrieved column water vapor and optical absorption paths for liquid H2O and ice."

## ENMAP
See [ENMAP data access](https://www.enmap.org/data_access/)
- L0: Raw data
- L1B: At-sensor radiance
- L1C: Orthorectified radiance
- L2A: Orthorectified reflectance, corrected for atmosphere
    - L2A land
    - L2A water

### PRISMA
The Prisma satellite has the following products (see Guarini et al., "PRISMA Hyperspectral Mission Products"):
- L0: 
    - Raw sensor data, calibration data, various metadata
    - 1000x1000 pixel hyperspectral images 
- QL (quicklook):
    - RGB preview generated from L0 data
- L1:
    - Radiance, top-of-atmosphere
- L2:
    - L2a - not described(?).
    - L2b: Surface radiance
    - L2c: Surface reflectance
    - L2d: Orthorectificed surface reflectance


## Massimal main image products:

### L0: Raw data (not published)
L0 corresponds to all the data collected during UAV flights. This includes:
    - Hyperspectral image (ENVI file)
        - Binary image file, band-interleaved (*.bil)
        - Header file with metadata (*.bil.hdr)
    - Downwelling spectrum (ENVI file). Note: Not included in every dataset due to technical problems.
        - Binary spectrum file, band-interleaved (*.spec)
        - Spectrum header file (*.spec.hdr)
    - Text file with timestamps for every image line (*.times)
    - Text file with sensor orientation and position from IMU (*.lcf)

Calibration files for the PikaL hyperspectral camera and the Flame downwelling irradiance spectrometer are also included. These are used for calculation of radiance and reflectance.

### L1a: Radiance
L1 images have been converted to radiance by calibration in the Resonon software, using the camera calibration file. This includes subtraction of "dark current" (check). Pixels that were saturated during image aquisition are set to zero for all bands. 

The data has the same file structure as L0, but also includes a "world file" (*.wld) with an affine transformation to georeference the image. The target coordinate system is UTM/WGS84. The UTM zone corresponds to the "local" UTM zone for the data (32N or 33N). 

The data also includes a "quick look" RGB representation of each image (*.png). Each image is contrast stretched using a 2%/98% percentile stretch, applied independently for each band. (wavelengths to be decided?)


### L2a: Reflectance (ENVI, not published?)
L2a images have been converted from radiance L to reflectance R by using the downwelling irradiance E:
    $$ R(\lambda) = \pi*\frac{L(\lambda)}{E(\lambda)}$$
which is based on an assumption that all reflections are perfectly Lambertian.  

Reflectance values are represented as "single" float values (32 bits) between 0 and 1. 

Reflectance images are limited to wavelengths below 930 nm, because the low signal-to-noise levels above 930 nm were causing spectral "spikes". 

### L2b: Reflectance (GeoTIFF, georeferenced, published)
Reflectance images (L2a) are georeferenced using the affine transform (saved as "world files"), and are saved as GeoTIFF images using single float values (32-bit). Both complete image cubes (hyperspectral image) and RGB images (3 bands, wavelengths to be decided?) are made.

### L2c: Water leaving reflectance (published)
Reflectance after applying sun glint correction (to be decided...). Same file format and georeferencing as L2b.

Limited to range [400,730] nm. There is almost no radiance above 730 nm due to water absorption. Signal-to-noise is also low both below 400 nm and above 730 nm. 


### Summary
The following data products are published
- L1: Radiance
    - ENVI file
    - Downwelling irradiance
    - Sensor position (*.times, *.lcf)
- L2b: Reflectance
    - GeoTIFF, georeferenced with affine transform (whole cube / rgb)
- L2c: Water-leaving reflectance (maybe)

## Annotation products

## Metadata products
### Affine transformation / "world files"
The information contained in the *.lcf file enables more accurate georeferencing than a simple affine transform, at least in theory. Affine transformation is limited to translation, rotation and scaling of the original rectangular image, and does not take all the details of the camera movement and the ground topography into account. However, in the Massimal project, most images were aquired with the camera mounted on a gimbal, which de-coupled the camera from the pitch and roll movements of the UAV. A simple affine transform was found to yield very similar results as "full" georeferencing (done via a plugin in Spectronon software, assuming a "flat earth"). Using an affine transform also has a number of advantages:
- No loss of resolution in the georeferencing process, and no "holes" in the resulting image
- The affine transformation can very easily be updated, while full georeferencing is very time-consuming.

The input data for georeferencing was also not perfectly accurate:
- Positions were measured with a standard GNSS system, which has an accuracy of approximately 3-5 meters. Combining the GNSS data with accelerometer and gyroscope data in a Kalman filter increases the accuracy, but positions are not as accurate as with RTK/PPK systems.
- Altitude was also measured based on GNSS data. Absolute altitude at take-off seemed to vary on the scale of a few meters. 
- The heading was measured using the IMU magnetometer, but data quality was consistently quite poor. Calculating heading based on GNSS data turned out to be more accurate.  

Using an affine transform seemed as a "good enough" georeferencing solution, given that perfect georeferencing was not possible due to imperfect IMU data. Note, however, that for many of the datasets it is possible to improve the georeferencing by manually identifying key points in the hyperspectral images and in the RGB base map (comment further?). 


