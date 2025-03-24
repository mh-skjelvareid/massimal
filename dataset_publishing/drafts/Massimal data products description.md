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

### L0b: Quicklook
Percentile stretched RGB versions of raw images. 

### L1a: Radiance
L1 images have been converted to radiance by calibration in the Resonon software, using the camera calibration file. This includes subtraction of "dark current" (check). Pixels that were saturated during image aquisition are set to zero for all bands. 

The data has the same file structure as L0, but also includes a "world file" (*.wld) with an affine transformation to georeference the image. The target coordinate system is UTM/WGS84. The UTM zone corresponds to the "local" UTM zone for the data (32N or 33N). 


### L2a: Reflectance (ENVI, not published?)
L2a images have been converted from radiance L to reflectance R by using the downwelling irradiance E:
    $$ R(\lambda) = \pi*\frac{L(\lambda)}{E(\lambda)}$$
which is based on an assumption that all reflections are perfectly Lambertian.  

Reflectance values are represented as "single" float values (32 bits) between 0 and 1. 

Reflectance images are limited to wavelengths below 930 nm, because the low signal-to-noise levels above 930 nm were causing spectral "spikes". 

Reflectance images are also georeferenced using the affine transform (saved as "world files"), and are saved as GeoTIFF images using single float values (32-bit). Both complete image cubes (hyperspectral image) and RGB images (3 bands, wavelengths to be decided?) are made.

### L2b: Water leaving reflectance (published)
Reflectance after applying sun glint correction (to be decided...). Same file format and georeferencing as L2b.

Limited to range [400,730] nm. There is almost no radiance above 730 nm due to water absorption. Signal-to-noise is also low both below 400 nm and above 730 nm. 

Pixels with mostly negative spectra are set to zero (masked out). 

Spectra are also smoothed using S-G filtering, and also binned/downsampled (?). The final result is saved as a GeoTIFF(?)


### Mosaics
RGB versions of images are mosaiced together into a single GeoTIFF. 


### Summary
The following data products are published
- L1: Radiance
    - ENVI file
    - Downwelling irradiance
    - Sensor position (*.times, *.lcf)
- L2a: Reflectance
    - ENVI file 
    - GeoTIFF, georeferenced with affine transform (whole cube / rgb)
- L2b: Water-leaving reflectance (maybe)
    - GeoTIFF

## Annotation products

## Metadata products
### Affine transformation / "world files"
The information contained in the *.lcf file enables more accurate georeferencing than a simple affine transform, at least in theory. Affine transformation is limited to translation, rotation and scaling of the original rectangular image, and does not take all the details of the camera movement and the ground topography into account. However, in the Massimal project, most images were aquired with the camera mounted on a gimbal, which de-coupled the camera from the pitch and roll movements of the UAV. A simple affine transform was found to yield very similar results as "full" georeferencing (done via a plugin in Spectronon software, assuming a "flat earth"). Using an affine transform also has a number of advantages:
- No loss of resolution in the georeferencing process, and no "holes" in the resulting image
- The affine transformation can very easily be updated, while full georeferencing is quite time-consuming.

The input data for georeferencing was also not perfectly accurate:
- Positions were measured with a standard GNSS system, which has an accuracy of approximately 3-5 meters. Combining the GNSS data with accelerometer and gyroscope data in a Kalman filter increases the accuracy, but positions are not as accurate as with RTK/PPK systems.
- Altitude was also measured based on GNSS data. Absolute altitude at take-off seemed to vary on the scale of a few meters. 
- The heading was measured using the IMU magnetometer, but data quality was consistently quite poor. Calculating heading based on GNSS data turned out to be more accurate.  

Using an affine transform seemed as a "good enough" georeferencing solution, given that perfect georeferencing was not possible due to imperfect IMU data. Note, however, that for many of the datasets it is possible to improve the georeferencing by manually identifying key points in the hyperspectral images and in the RGB base map (comment further?). 

## Hyperspectral naming pattern
hyspec\_area\_location\_date\_flight\_imagenumber_\productname.fileext

Example: hyspec_larvik_kongsbakkebukta_20230830_flight_03_image_17_radiance.bil

Raw data currently are named e.g.

Kongsbakkebukta_Pika_L_17.bil

Thw following changes could be made, in addition to adding location and date to the filename
- "Pika_L" does not need to be part of the filename, as it is the same for all files in the dataset
- Numbering should be done with a fixed number of digits, padded with leading zeroes. With the current naming, file "Kongsbakkebukta_Pika_L_17" comes before "Kongsbakkebukta_Pika_L_3". This propagates into incorrect ordering in subsequent processing, e.g. when importing into GIS programs, when manually searching for a file, etc.

Files have so far not been organized according to flight. This could be done manually, or files could be automatically grouped according to time "proximity". Use K-means clustering on timestamps? Should be easy to set number of clusters.

Need to rename both radiance and reflectance files using same naming convention. Water leaving reflectance can be created based on reflectance files.

## Massimal data folder structure (needs update)

Note: Python module (treelib)[https://treelib.readthedocs.io/en/latest/] or (anytree)[https://anytree.readthedocs.io/en/latest/]could be useful for representing the structure.

- root
    - area_location (e.g. bodo_juvika)
        - aerial
            - hyperspectral
                - date
                    - images
                        - L1_radiance
                            - \*.bil
                            - \*.bil.hdr
                            - \*.spec
                            - \*.spec.hdr
                            - \*.lcf
                            - \*.times
                            - \*.wld
                        - L2a_reflectance
                            - \*.bil
                            - \*.bil.hdr
                            - \*.lcf
                            - \*.times
                            - \*.wld
                            - \*.tif (rotated transform)
                        - L2b_water_leaving_reflectance
                            - \*.bil
                            - \*.bil.hdr
                            - \*.lcf
                            - \*.times
                            - \*.wld
                            - \*.tif (rotated transform)
                    - mosaic
                        - L2a_reflectance
                            - images (non-rotated transform)
                                \*.tif
                            - \*.vrt
                            - \*.ovr
                        - L2a_reflectance
                            - images (non-rotated transform)
                                - \*.tif
                            - \*.vrt
                            - \*.ovr                        
                    - annotations
                        - images 
                            - color_images
                                - \*.png
                            - grayscale_images
                                - \*.png
                            - \*.json
                        - hasty_json
                            - \*.json

                    - metadata
                        - hyperspectral_camera_calibration
                            - \*.icf
                        - irradiance_spectromenter_calibration
                            - \*.dcp
            - multispectral
                - date
                    - images (?)
                    - mosaic
                        - separate_bands
                            - blue
                            - green
                            - red
                            - red_edge
                            - nir
                            - lwir
                        - merged
                            - \*.tif
                    - report
            - rgb
                - date
                    - images
                    - mosaic
                    - report
        - ground
            - usv
                - date
                    - navigation_log
                        - *.csv
                    - images
                        - *.jpg / *.png (?) (geotagged)
                    - depth
                        - *.csv
                    - water_quality (?)
                        - *.csv
            - snorkeling_transects
                - date
                    - transect_name
                        - images
                            - \*.jpg / \*.png (?) (geotagged?) 
                        - start_end_points
                            - \*.csv
            - rov
                - date
                    - images
                        - \*.jpg / \*.png (?) (geotagged?) 
                    - start_end_points
                        - \*.csv
            - walking_transects
                - date
                    - images
                        - \*.jpg / \*.png (?) (geotagged?) 
                    - position_log
                        - \*.csv
            - boat_transects
                - date
                    - images
                        - \*.jpg / \*.png (?) (geotagged?) 
                    - position_log
                        - \*.csv


To do:
- Create folder structure
- Copy existing files into structure
- (Re)process ground images for geotagging
- Merge multispectral single band images into multiband images(?)

## Hyperspectral folder structure
    ├───0_raw
    ├───0b_quicklook
    ├───1a_radiance
    │   └───rgb
    ├───1b_radiance_sgc
    │   └───rgb
    ├───2a_reflectance
    │   └───rgb
    ├───2b_reflectance_sgc
    │   └───rgb
    ├───calibration
    ├───imudata
    ├───mosaics
    └───notes

## Logic for determining which data products are created
- radiance: If option set
- radiance_sgc: If radiance exists and option set and sgc "training cube" given
- reflectance: If radiance exists and option set
- reflectance_sgc: If reflectance / radiance_sgc exists (see below) and option set
- mosaic (all versions): If option set


Possible to calculate SGC version of reflectance in multiple ways:
1. Using SGC version of radiance (lin. reg.), if present
2. Using "flat spec" SGC method on reflectance 
3. Using lin. reg. SGC method on reflectance (equvivalent to 1?). Requires training cube
-> Need for practical experience with these to determine if all should be possible or if one is superior


## Notes on multispectral images
Ølbergholmen: Images taken with Micasense Altum camera. Not sure of the exact serial number for the camera. The bands given below are for serial number AL04 or lower.
Blue (475nm ±20nm), Green (560nm ±20nm), Red (668nm ±10nm), Red Edge (717nm ±10nm), NIR (840nm ±40nm), Thermal (11 μm ± 6 μm)


## GCPs
- Check out [GCPEditorPro](https://github.com/uav4geo/GCPEditorPro)

