# MASSIMAL hyperspectral image dataset

This readme file is part of a dataset containing hyperspectral images of shallow-water
marine habitats collected using an unmanned aerial vehicle (UAV, or "drone"). This file
provides details on the research project that collected the images, the imaging
equipment used, post-processing of the data, and the type and formatting of the data
provided.  


## Table of contents
1. [The MASSIMAL research project](#the-massimal-research-project)
2. [Hyperspectral imaging system](#hyperspectral-imaging-system)
3. [UAV platform for hyperspectral imaging](#uav-platform-for-hyperspectral-imaging)
4. [Field operations](#field-operations)
5. [MassiPipe data processing pipeline](#massipipe-data-processing-pipeline)
6. [Dataset contents](#dataset-contents)


## The MASSIMAL research project 
This dataset was collected as part of the "MASSIMAL" project (Mapping of Algae and
Seagrass using Spectral Imaging and Machine Learning). The project was conducted in the
period 2020-2024, and data collection and field work was performed at various locations
along the Norwegian coast in the period 2021-2023. 

The project was financed by the Norwegian Research Council (8 MNOK) and by UiT the
Arctic University of Norway (600 kNOK), and was a collaboration between 

- UiT the Arctic University of Norway ("UiT")
- Norwegian Institute for Water Research ("NIVA")
- Nord University ("Nord")

The following researchers were the main contributors to the project:

- Martin Hansen Skjelvareid (principal investigator), UiT
- Katalin Blix, UiT
- Eli Rinde, NIVA
- Kasper Hancke, NIVA
- Maia Røst Kile, NIVA
- Galice Guillaume Hoarau, Nord

All UAV flights were piloted by Sigfinn Andersen at FlyLavt AS, Bodø.

Additional information about the project can be found on the following websites:
- [UiT project page](https://en.uit.no/project/massimal)
- [Cristin research database project
  page](https://app.cristin.no/projects/show.jsf?id=2054355)
- [Norwegian Research Council project
  page](https://prosjektbanken.forskningsradet.no/project/FORISS/301317)
- [SeaBee data portal with Massimal
  data](https://geonode.seabee.sigma2.no/catalogue/#/search?q=massimal&f=dataset)

  
## Hyperspectral imaging system
Hyperspectral imaging was performed using a "[Airborne Remote Sensing
System](https://resonon.com/hyperspectral-airborne-remote-sensing-system)" manufactured
by Resonon. The system was configured with the following components:


### Hyperspectral camera
A [Pika-L](https://resonon.com/Pika-L) hyperspectral camera fitted with a lens with 8 mm
focal length and 36.5 degree field of view (see [lens
options](https://resonon.com/objective-lenses)) was used for all image acquisitions. The
Pika-L is a pushbroom sensor; it measures light coming from a narrow line in the imaged
scene. The light from the line is reflected by a diffraction grating, which splits the
light into its spectral components, onto a 900x600 pixel sensor. A single image frame
corresponds to 900 spatial pixels, with the light intensity sampled at 600 wavelengths
in the 390-1030 nm range. By default, pairs of spectral pixels are binned, resulting in
300 spectral channels. Resonon lists that the Pika-L has a spectral range of 400-1000 nm
and 281 spectral channels, indicating that the channels at the high and low ends of the
spectra are usually discarded. 

To create a 2D image of a scene, the camera has to be moved across the area of interest
with the line-of-view perpendicular to the direction of motion. The number of
across-track pixels is always 900, corresponding to the number of spatial pixels of the
sensor, while the number of along-track pixels depends on how many image lines are
acquired. 

The along-track sampling distance on the ground is given by the speed of the camera
platform divided by camera frame rate. The across-track sampling distance is defined by
the camera field of view and the relative distance between the camera and the ground. In
the Massimal project, the frame rate was always set to 100 frames per second, and the
UAV altitude and speed were adjusted to keep the two ground sampling distances as
similar as possible (typically around 3.5 cm).

The spectral resolution (FWHM) of the camera's 300 spectral channels averages around 2.7
nm. The table below shows the FWHM for selected channels based on Resonon's general
design documents. Individual cameras may vary slightly, but Resonon states the data
aligns well with experiments. Note the local minimum of 2.24 nm at ~470 nm and the local
maximum of 2.94 nm at ~745 nm.

| Wavelength (nm) | FWHM (nm) |
| :-------------: | :-------: |
|             400 |      2.97 |
|             415 |      2.55 |
|             430 |      2.36 |
|             445 |      2.27 |
|             460 |      2.24 |
|             475 |      2.24 |
|             490 |      2.26 |
|             505 |      2.29 |
|             520 |      2.33 |
|             535 |      2.39 |
|             550 |      2.44 |
|             565 |      2.50 |
|             580 |      2.57 |
|             595 |      2.63 |
|             610 |      2.68 |
|             625 |      2.74 |
|             640 |      2.79 |
|             655 |      2.83 |
|             670 |      2.86 |
|             685 |      2.89 |
|             700 |      2.92 |
|             715 |      2.93 |
|             730 |      2.94 |
|             745 |      2.94 |
|             760 |      2.94 |
|             775 |      2.93 |
|             790 |      2.91 |
|             805 |      2.89 |
|             820 |      2.87 |
|             835 |      2.84 |
|             850 |      2.80 |
|             865 |      2.77 |
|             880 |      2.73 |
|             895 |      2.69 |
|             910 |      2.65 |
|             925 |      2.61 |
|             940 |      2.57 |
|             955 |      2.53 |
|             970 |      2.50 |
|             985 |      2.47 |
|            1000 |      2.49 |


### On-board computer
The hyperspectral camera, IMU and radio modem for communicating with the ground station
were controlled by a small on-board computer. The computer was an [Intel
NUC](https://en.wikipedia.org/wiki/Next_Unit_of_Computing) (7th generation, model
NUC7i7DNK) running a Linux-based imaging firmware made by Resonon. 


### Inertial measurement unit (IMU)
An SBG Ellipse 2N inertial measurement unit was connected to the onboard computer. The
IMU consists of 3 accelerometers and 3 gyroscopes to measure translational and angular
accelerations of the camera, a GNSS receiver for measuring position and velocity, a
barometric altimeter for aiding altitude measurement, and a magnetometer aiding in
heading measurement. The sensor data are combined in an extended Kalman filter to
produce estimates of camera position (latitude, longitude, altitude) and orientation
(pitch, roll, yaw). The specified accuracy for the GNSS receiver was 2.0 m CEP.
Real-time or post-processing kinematic positioning (RTK/PPK) was not available for the
IMU.


### Downwelling irradiance measurement
The (spectral) downwelling irradiance is a measurement of the total intensity of light
coming from the sky, measured as power per area per wavelength (typically W/(m²·nm)).
The downwelling irradiance can be combined with a hyperspectral image to calculate the
reflectance for each image pixel.
 
A Flame-S-VIS-NIR spectrometer with a CC-3-DA cosine collector manufactured by Ocean
Insight was used to measure downwelling spectral irradiance. The spectrometer has 2048
spectral channels covering the range of 350-1000 nm. The optical resolution is 1.33 nm
and the spectral sampling distance is 0.3-0.4 nm. The spectrometer was mounted directly
to one of the arms of the UAV. 

**Note that for some datasets, downwelling irradiance was not measured due to a technical
failure (poor cable connection between spectrometer and on-board computer).**


## UAV platform for hyperspectral imaging

### Multirotor UAV
The UAV was a [Matrice 600 Pro](https://www.dji.com/no/support/product/matrice600-pro)
manufactured by DJI, a hexacopter design with 6 propellers, each 21 inches long. This
model was chosen because 

- Resonon has used the same model to carry their airborne hyperspectral systems.
- The model is one of the most affordable commercially available UAVs with high enough
  payload capacity to carry the hyperspectral system.
- DJI offered a compatible gimbal that was well suited for the hyperspectral camera.

 The UAV weighed 9.5 kg without payload, and approximately 14 kg with the full payload.
A typical flight lasted approximately 10 minutes, with a safety margin of at least 30 %
remaining battery capacity. Six TB47S batteries were used to power the UAV, and three
sets of batteries were used to enable multiple flights and battery changing in the
field. 

The arms and propellers of the UAV can be folded for transport. When fully extended, the
UAV has a "wingspan" and height of approximately 1.65x0.75 meters. During
transportation, the UAV was folded and transported in a case measuring 0.8x0.7x0.7
meters.


### Gimbal
The hyperspectral camera and the onboard computer were mounted to a DJI
[Ronin-MX](https://www.dji.com/no/ronin-mx) 3-axis gimbal. The purpose of the gimbal was
to keep the camera level and to "decouple" it mechanically from the UAV, which uses
tilting of the aircraft for manouvering. 

The camera was kept pointing directly downwards ("nadir") with the line of sight
perpendicular to the forward movement of the drone. For yawing movements, the gimbal was
set to follow the heading of the UAV, but with low angular accelerations. 


### Radio modem
For most missions, a lightweight radio modem (Digi XBee, 2.4 GHz) was used for
communication between the ground station and the on-board computer. The communication
link was mainly used for configuring imaging missions and monitoring that data recording
was proceeding as planned.


## Field operations

### Ground station
All datasets were acquired with a UAV ground station close to the area of interest
(usually 500 meters or less), with clear visual line of sight to the UAV. In some cases
a ground station could be set up close to the road, and in some cases all equipment had
to be transported via a small boat.  


### Mission setup
Most imaging missions were performed by defining a target area of interest, creating a
[KML file](https://en.wikipedia.org/wiki/Keyhole_Markup_Language) with a polygon
describing the area, and uploading the KML file to the UAV on-board computer. The
Airborne Remote Sensing System senses when the UAV is inside the target area, and
automatically starts and stops the recording accordingly. The KML file was also used in
flight planning software to create waypoints for flight lines. During field campaigns in
2021 and 2022, DJI Pilot was used for flight planning, while in 2023,
[UgCS](https://www.sphengineering.com/flight-planning/ugcs) was used.


### Splitting flight lines into multiple images
The Airborne Remote Sensing System is set up so that if an image reaches the limit of
2000 lines (2000 pixels vertically), the image is saved, and image recording continues
in a new image file. The practical effect of this is that images along a single
continuous flight line are split into multiple images - typically 4-10 images for each
flight line.


### Autoexposure
The Airborne Remote Sensing System includes an autoexposure feature. With this, the
camera gain and shutter is automatically adjusted, based on test images acquired by the
camera, to bring the distribution of image values into a suitable part of the camera
dynamic range. Autoexposure was used on a per-image basis, meaning that camera gain and
shutter were adjusted between each image.  

Note that using autoexposure occasionally resulted in suboptimal gain and shutter
values. For example, if the UAV was above (dark) water at the time of autoexposure and
then flew over (bright) land before the autoexposure could be recalculated, parts of the
image became saturated, resulting in invalid pixels. In processed images, saturated
pixels are set to zero across all channels, to indicate that the spectrum is invalid. 


### Gimbal operation
For most of the datasets collected in the project, the data was collected with the
camera mounted on a 3-axis stabilizing gimbal. Before takeoff, the gimbal was adjusted
to point the camera in the nadir direction, with the camera line of sight being
perpendicular to the direction of flight. During sharp turns, typically between straight
flight lines, the yawing movement would cause the camera heading to "lag behind" that of
the UAV for some time. However, the gimbal eventually returned the camera to the
original orientation relative to the UAV. This behavior was not ideal, and it probably
contributes to a "skewing" artifact seen in some images, but no solution to avoid it was
found during fieldwork.  


## MassiPipe data processing pipeline
The dataset has been processed by "MassiPipe", a data processing pipeline developed as
part of the MASSIMAL project. The pipeline has been used to 

- Convert raw hyperspectral images to radiance
- Convert raw spectrometer data to downwelling irradiance
- Process IMU data and create a "geotransform" for each image 
- Adding irradiance and geotransform data to radiance image header
- Create RGB versions of the images, including mosaics
 
MassiPipe can also be used to automatically generate additional image products, e.g.
reflectance images and glint corrected images, based on the radiance images distributed
in the dataset. See details in the description of dataset contents below. 

- MassiPipe documentation: https://mh-skjelvareid.github.io/massipipe/
- MassiPipe GitHub repository: https://github.com/mh-skjelvareid/massipipe
- DOI for MassiPipe: [10.5281/zenodo.14748766](https://doi.org/10.5281/zenodo.14748766)

The dataset was processed by version 0.3.0 of MassiPipe. Batch processing was run using a
Jupyter Notebook running on a "deep-learning-tools" package (v.9.1.1) available through
the [NIRD Toolkit](https://documentation.sigma2.no/nird_toolkit/overview.html). This is
a service  offered by the Norwegian Research Infrastructure Services (NRIS) and run by
[Sigma2](https://www.sigma2.no/). 


## Dataset contents
The following is a general description which is valid for most of the datasets produced
by the MASSIMAL project. Some deviations from this description may exist for special
cases. Review your specific dataset to check what type of data is included. 

Note that the raw data is not distributed as part of the dataset, as calibrated radiance
images was considered more generally useful than raw data. Due to storage and bandwidth
concerns, only one version of each hyperspectral image was included in the dataset. 


### Dataset structure and file naming
The dataset har the following structure:

    ├── 1a_radiance
    │   ├── rgb    
    │   |   ├── <DatasetName>_<ImageNumber>_radiance_rgb.tiff
    │   |   └── ...
    │   ├── <DatasetName>_<ImageNumber>_irradiance.spec
    │   ├── <DatasetName>_<ImageNumber>_irradiance.spec.hdr
    │   ├── <DatasetName>_<ImageNumber>_radiance.bip
    │   ├── <DatasetName>_<ImageNumber>_radiance.bip.hdr
    │   └── ...
    ├── imudata
    │   ├── <DatasetName>_<ImageNumber>_imudata.json
    │   └── ...
    ├── orthomosaic
    │   └── <DatasetName>_<ImageType>_rgb.tiff
    ├── quicklook
    │   ├── <DatasetName>_<ImageNumber>_quicklook.png
    │   └── ...
    ├── config.seabee.yaml
    ├── license.md
    └── readme.md

All datasets are named using the same pattern. Here is an example:

    massimal_larvik_kongsbakkebukta_202308301328_hsi

Here, "massimal" is the project name, "larvik" is the name of the general area,
"kongsbakkebukta" is the name of the specific location, "202308301328" is the date and
time when the first image was taken (using the pattern yyyymmddHHMM), and "hsi" is an
abbreviation of "hyperspectral image". For some datasets the timestamp is followed by a
tag with additional information, like the "-south1" tag in the example below:

    massimal_larvik_olbergholmen_202308300959-south1_hsi

Individual files are numbered with three digits, starting from 000. An example:

    massimal_larvik_kongsbakkebukta_202308301328_hsi_000_quicklook.png


### Configuration file - config.seabee.yaml 
The configuration file contains metadata about the dataset, and the parameters for
processing the dataset using MassiPipe. The file name includes "seabee" because the
dataset was prepared for publication and interactive exploration through the SeaBee data
portal, run by [SeaBee](https://seabee.no/) (Norwegian Infrastructure for Drone-based
Research, Mapping and Monitoring in the Coastal Zone).

For explanation of the MassiPipe processing parameters, see the [MassiPipe
documentation](https://mh-skjelvareid.github.io/massipipe/). The configuration file can
be edited to re-run the processing with modified parameters, and/or produce additional
image products. 


### Quicklook
The "quicklook" images are color images saved in PNG format, to be used for getting a
quick overview of the dataset. The data displayed as red, green and blue (RGB)
corresponds to slices of the hyperspectral radiance image extracted at the "RGB
wavelengths" set in the configuration (typically 640, 550 and 460 nm). Each color
channel has also been individually contrast stretched, using the 2nd and 98th
percentiles of the original data to set the lower and upper ends of the range of values
displayed. 

Note that since the image statistics change from image to image, the percentile
stretching may cause identical objects or nature types to appear different in different
images. Quicklook images are also not georeferenced. It is not recommended to use the
quicklook images for any type of analysis - use the radiance data (or further processed
image products) instead. 


### IMU data
During image acquisition, the stream of IMU data is logged to a text file (\*.lcf), and
timestamps for each image line is logged to another file (\*.times). The timing of IMU
data sampling and image line acquisition is not synchronized in the raw data. In
post-processing, IMU data is interpolated to match the timestamps for each image line.
E.g., if the image contains 2000 lines, each IMU data field has 2000 corresponding
values.

IMU data is stored as JSON files with 7 fields:
- **time**: Time represented as a single floating-point value, describing the number of
  seconds passed since January 1st 1970 (["UNIX
  time"](https://en.wikipedia.org/wiki/Unix_time)). 
- **roll**: Camera roll measured in radians. Positive values correspond to "right wing
  up", or pointing the camera to the right side of the flight line. 
- **pitch**: Camera pitch measured in radians. Positive values correspond "nose up", or
  pointing the camera forward, relative to nadir.
- **yaw**: Camera heading, measured in radians. Zero at due north, pi/2 at due east.
- **longitude**: Longitude in decimal degrees, positive for east longitude.
- **latitude**: Latitude in decimal degrees, positive for northern hemisphere.
- **altitude**: Altitude in meters relative to the WGS-84 ellipsiod.

*Note that while roll, pitch, longitude and latitude are relatively accurate, yaw and
altitude are less so.* The magnetometer which the IMU used to measure yaw (heading)
turned out to have poor accuracy, despite repeated calibrations. Absolute altitude
values also appear to be accurate only to approximately +/- 7 m, probably due to
limited accuracy when estimating altitude from GPS / GNSS. However, relative altitude
values for the same flight (same dataset) are fairly consistent, probably because
altitude estimation was aided by a barometric pressure sensor.  


### Downwelling irradiance spectra
For many of the datasets recorded in the project, a downwelling irradiance spectrum has
been recorded for each hyperspectral image. The irradiance spectra are placed together
with the hyperspectral images in the folder named 1a_radiance, and are saved in the same
ENVI format as the hyperspectral images. The file name extensions are \*.spec and
\*.spec.hdr for the binary and header files, respectively. 

The raw spectrum has been converted to units of W/(m²·nm) by subtracting a dark current
spectrum and multiplying with a gain spectrum. The wavelengths of the irradiance
spectrum have also been calibrated by detecting [Fraunhofer
lines](https://en.wikipedia.org/wiki/Fraunhofer_lines) in the recorded spectra. A
polynomial was fitted to the detected Fraunhofer lines, for which the wavelengths are
well-known, and the polynomial was used to calculate calibrated wavelengths for every
channel of the irradiance spectrum. 

For additional details regarding the calibration, see the code in
[massipipe.irradiance](https://github.com/mh-skjelvareid/massipipe/blob/main/massipipe/irradiance.py).

Note that the spectral resolution is much higher for the irradiance spectrometer than
for the hyperspectral camera. When using the irradiance spectrum for calculating
reflectance images, the spectrum should be smoothed and resampled to match the spectral
characteristics of the camera. Where possible, such a smoothed and resampled spectrum
has been written to the radiance image header. See the section on [irradiance in
radiance header files](#solar-irradiance) for details. 


### Radiance hyperspectral images

#### File format
The radiance hyperspectral images are placed in a the folder called `1a_radiance`, and
constitute the largest and most important part of the dataset. The radiance images are
saved in the [ENVI
format](https://www.nv5geospatialsoftware.com/docs/ENVIImageFiles.html), which splits
the data into two parts: 
- A binary file with file extension \*.bip (e.g. `image.bip`), and
- A text file with metadata, with the same file name as the binary file,
  but with an additional \*.hdr extension (e.g. `image.bip.hdr`).

  
#### Calibration and physical units
The radiance image has been converted from a raw image to a calibrated image with units
of [microflicks](https://en.wikipedia.org/wiki/Flick_(physics)) (µW/(cm²·µm·sr)). The
microflick unit may seem arbitrary or outdated, but is used here because typical values
are well suited for encoding using 16-bit unsigned integers.  

The calibration consists of two steps; subtracting the dark current noise from every
image frame, and then multiplying every frame with a "gain" frame which converts digital
numbers to microflicks. For details on the calibration process, see the code at
[massipipe.radiance](https://github.com/mh-skjelvareid/massipipe/blob/main/massipipe/radiance.py).


#### Georeferencing ("map info")
The radiance image headers include the field "[map
info](https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html)", which provides
information on georeferencing of the image. 

Georeferencing or georectification of an image is the process of placing each image
pixel at a defined position on the surface of the Earth. If accurate measurements of the
camera's position and orientation is available, together with a surface elevation
model, it is possible to perform georectification very accurately. However, since the
heading and altitude measurements from the IMU used in this project were not very
accurate, georectified versions of the hyperspectral images (using e.g. Resonons
[georectification plugin for
Spectronon](https://docs.resonon.com/airborne/Airborne%20Post%20Processing%20Summary.pdf))
tended to be somewhat distorted and inaccurate. The georectification process is also
lossy, in the sense that it is often not possible to recover the original image from the
georectified image. 

For the reasons listed above, the images in the dataset have not been fully
georectified. However, a simplified georectification based on an [affine
transform](https://en.wikipedia.org/wiki/Affine_transformation) has been applied. A
common application of the affine transform is so-called "[world
files](https://en.wikipedia.org/wiki/World_file)" for georeferencing raster images.  The
affine transform scales, rotates and translates an image from its original pixel
coordinates into a geospatial coordinate reference system. This has been found to
produce "good enough" georectification, given the quality of the IMU data, while also
being lossless.  


The affine transform has been calculated based on the IMU data and on the camera field
of view. See
[massipipe.georeferencing](https://github.com/mh-skjelvareid/massipipe/blob/main/massipipe/georeferencing.py)
for details. 

The "map info" field in the image header does not list the parameters of the affine
transform directly, but it is possible to translate between the two. For an in-depth
explanation, see the tutorial "[Understanding AVIRIS-NG data in ENVI format with rotated
grid](https://github.com/ornldaac/AVIRIS-NG_ENVI-rotatedgrid)". 

Here is an example of map info from a hyperspectral image: 

`map info = {UTM, 1, 1, 565020.16, 6541378.72, 0.0355271, 0.0369927, 32, North, WGS-84,
rotation=19.060428}.`

The map info parameters are (in order):
1. Projection name (always UTM for this project)
2. Reference pixel in x-direction, starting from 1 (always 1 for this project)
3. Reference pixel in y-direction, starting from 1 (always 1 for this project)
4. Reference pixel easting (565020.16 in the example above)
5. Reference pixel northing (6541378.72 in the example above)
6. X pixel size (across-track spatial sampling, 0.0355271 meters in the example)
7. Y pixel size (along-track spatial sampling, 0.0369927 meters in the example)
8. Projection zone (used only for UTM, 32 in the example)
9. Hemisphere (used only for UTM, North or South)
10. Geodetic datum / reference frame (always WGS-84 for this project)
11. Units / key-value pairs. The use of this parameter is poorly documented, but "units"
    (e.g. units=meters) and "rotation" (e.g. rotation=19.060428, as in the example) are
    common.

Note that as of February 2025, the "rotation" key-value pairs are accepted by software
like GDAL and QGIS.  

Note that it is possible for users of the data to do more advanced and/or accurate
georectification than than provided by the affine transform:
- Using the details of the provided IMU data to georeference the image line-by-line. See
  e.g. the Python package [gref4hsi](https://github.com/havardlovas/gref4hsi) for an
  open-source alternative to do this.
- Using a high-resolution base raster, for example the RGB mosaics also acquired in the
  Massimal project, finding corresponding image features between the base raster and the
  hyperspectral image, and warping the hyperspectral image to match the features. The
  QGIS "georeferencer" tool is one option to do this manually. 


#### Solar irradiance
Where possible, the downwelling irradiance spectrum has been included in the header file
of the radiance image, under the field "[solar
irradiance](https://www.nv5geospatialsoftware.com/docs/EnterOptionalHeaderInformation.html#SolarIrradiance)".
This field is one of the standard (but optional) fields of the ENVI data format, and is
usually used to specify the "top of atmosphere" irradiance for satellite imaging. The
units of solar irradiance are W/(m²·µm), and the number of spectral channels and the
wavelengths of the irradiance values correspond to those of the hyperspectral image. To
achieve this, the original irradiance measurements have been smoothed using a Gaussian
kernel with FWHM of 3.5 nm, and then resampled to match the wavelengths of the
hyperspectral camera. Note that spectral smoothing is needed for the shape of irradiance
spectrum to be closer to that of the spectra in the hyperspectral images. Because of its
higher spectral resolution, the original irradiance spectrum has some very sharp and
deep valleys caused by Fraunhofer lines. 

In the Massimal project, the purpose of measuring downwelling irradiance is to calculate
a reflectance image. While the amount of reflected light from an object can vary
drastically, depending on weather, time of day etc., the reflectance of the object can
remain more or less constant. This makes reflectance useful for habitat mapping. 

Many definitions of reflectance exist. One common defininition is $\rho =
(\pi*L)/E$, where $L$ denotes radiance (i.e. hyperspectral image) and $E$ denotes
downwelling irradiance. Under the simplifying assumption that all objects in the image
act as a [Lambertian reflector](https://en.wikipedia.org/wiki/Lambertian_reflectance),
i.e. that an incoming ray of light is reflected equally in all directions, $\rho$
corresponds to the [spectral irradiance
reflectance](https://www.oceanopticsbook.info/view/inherent-and-apparent-optical-properties/reflectances),
the ratio between upwelling and downwelling irradiances.  An alternative definition,
[remote sensing
reflectance](https://www.oceanopticsbook.info/view/inherent-and-apparent-optical-properties/reflectances)
$R$, omits the scaling factor $\pi$, such that $R = L/E$. The unit of $R$ is inverse
steradians. 

Note that the units of radiance in the hyperspectral image (microflicks, µW/(cm²·µm·sr))
and the units of downwelling irradiance in the ENVI header (W/(m²·µm)) are not directly
compatible. To convert the irradiance values from those in the header to units that
match the radiance images, multiply the irradiance values by 100. 

The MassiPipe pipeline can be configured to calculate reflectance images ($\rho$) directly from
radiance images and the irradiance spectrum in the header. 


### Radiance RGB GeoTIFFs
The folder `1a_radiance` has a subfolder named `rgb`, which contains RGB versions of the
radiance images, saved as GeoTIFF files. The red, green and blue images are "slices" of
the hyperspectral image at three wavelengths, as for the [quicklook](#quicklook) image.
However, the GeoTIFF contains the original values from the radiance image, and has not
been contrast stretched. 

The GeoTIFF images are georeferenced, and are useful as "lightweight" representations of
the hyperspectral image. Use compatible software (e.g. QGIS) to view the GeoTIFF
image in a geospatial context. 


### Mosaic
The mosaic is a single RGB GeoTIFF file for visualization of all the images in the
dataset. The mosaic is generated based on radiance RGB GeoTIFFs, but after combining all
the images, each color channel of the mosaic is percentile stretched (in the same way as
for quicklook images), and finally the mosaic is converted to 8-bit integer format. Note
that the conversion creates a suitable image contrast and compresses the image, but that
the values of the image lose their physical units. 

For most datasets, the mosaic is created directly from radiance data. The file name of
the mosaic then ends in \*_rad_rgb.tiff. However, for some datasets with significant
amounts of sun and sky glint in the water surface, the mosaic is created based on "glint
corrected" radiance. In these cases, the file name ends in *_rad_gc_rgb.tiff. 

Glint corrected radiance images are not distributed with the dataset, but can be
generated from the radiance images, for example by using MassiPipe. See the code at
[massipipe.glint](https://github.com/mh-skjelvareid/massipipe/blob/main/massipipe/glint.py)
for details on glint correction. 

