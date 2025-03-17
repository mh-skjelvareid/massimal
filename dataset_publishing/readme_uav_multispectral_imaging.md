# MASSIMAL multispectral UAV imaging dataset
This file is a description of a dataset with multispectral images of shallow coastal
areas in Norway, collected using a UAV (drone). The dataset was collected as part of a
research project, and two different multispectral cameras were used at two different
locations. This file describes both of these. 

The remainder of the file describes the MASSIMAL project, the UAVs and cameras used,
photogrammetry post-processing, and the structure of the dataset.


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

Additional information about the project can be found on the following websites:
- [UiT project page](https://en.uit.no/project/massimal)
- [Cristin research database project
  page](https://app.cristin.no/projects/show.jsf?id=2054355)
- [Norwegian Research Council project
  page](https://prosjektbanken.forskningsradet.no/project/FORISS/301317)
- [SeaBee data portal with Massimal
  data](https://geonode.seabee.sigma2.no/catalogue/#/search?q=massimal&f=dataset)


## Camera and UAV
Two different cameras and UAVs were used during two research campaigns in the Larvik and
Vega areas. However, both cameras were manufactured by MicaSense, and the datasets are
very similar. 

Full details regarding UAV and camera operation in the field were unfortunately not
available during the writing of this readme file, but most relevant information is
contained in the metadata of the images. 

### Datasets from Larvik (Ølbergholmen)
Multispectral imaging was performed by Medyan Ghareeb from NIVA, as part of the
[SeaBee](https://seabee.no/) research infrastructure project. A [MicaSense
Altum](https://support.micasense.com/hc/en-us/articles/360010025413-Altum-Integration-Guide)
camera and a [DJI Matrice
210](https://www.dji.com/no/support/product/matrice-200-series) UAV was used for
imaging. 

The MicaSense Altum camera (serial \# AL05-1934182-SC) uses a separate sensor for every
wavelength band. These bands are:

| Band name | Band number | Center wavelength (nm)| FWHM (nm) |
| :-------: | :---------: |:--------------------: | :-------: |
|      Blue |           1 |                   475 |         32|
|     Green |           2 |                   560 |         27|
|       Red |           3 |                   668 |         14|
|  Red edge |       **5** |                   717 |         12|
|       NIR |       **4** |                   842 |         57|
|   Thermal |           6 |                11 000 |      6 000|


### Datasets from Vega (Søla and Mellomskjær)
Multispectral imaging was performed by the company
[SpectroFly](https://www.spectrofly.dk/), using an "eBee" UAV manufactured by
[AgEagle](https://ageagle.com/). The UAV was equipped with a [MicaSense
RedEdge-M](https://support.micasense.com/hc/en-us/sections/4420305003415-RedEdge-M-Legacy)
camera. Data was processed post-flight using AgEagle's
[eMotion](https://ageagle.com/solutions/software/) software.

The RedEdge-M camera (serial \# RX01-1838284-SC) uses a separate sensor for every
wavelength band. These bands are:

| Band name | Band number | Center wavelength (nm)| FWHM (nm) |
| :-------: | :---------: |:--------------------: | :-------: |
|      Blue |           1 |                   475 |         20|
|     Green |           2 |                   560 |         20|
|       Red |           3 |                   668 |         10|
|  Red edge |       **5** |                   717 |         10|
|       NIR |       **4** |                   840 |         40|


## Photogrammetry software
Two different photogrammetry programs were used,
[OpenDroneMap](https://www.opendronemap.org/) and [Pix4D](https://www.pix4d.com/). You
can determine which was used for your specific dataset by e.g. reading the report under
the `report` subfolder.  

All the images acquired were processed using OpenDroneMap, as implemented in the SeaBee
[data processing pipeline](https://seabee-no.github.io/documentation/). Some datasets
from Vega were also processed with Pix4D.

Details regarding software settings were unfortunately not available during the writing
of this readme file. 


## Data format 
The dataset is organized as follows:

    ├── dem/
    ├── images/
    ├── orthophoto/
    ├── other/
    ├── report/
    └── config.seabee.yaml

Note that not all folders are present in every dataset. The following describes the
contents of each folder in the dataset. 

### DEM (digital elevation model)
Only included for datasets processed with ODM. The `dem` folder contains digital surface
models (DSM) and digital terrain models (DTM, models where vegetation, buildings etc.
has been removed). The two versions of the data are very similar for the MASSIMAL
datasets.

The photogrammetry software was not set up for calculating under-water
elevation/bathymetry, but in some shallow areas the models still clearly show the shape
of the seabed under water. Note, however, that the models cannot be used to accurately
measure water depth in absolute units.

### Images
The `images` folder contains the original images used for photogrammetry. They are named
with the pattern `IMG_<image number>_<band number>.tif`. A large amount of additional
information is saved in each image's EXIF metadata, including camera position and
orientation, camera calibration parameters, solar irradiance, etc. Use software capable
of reading EXIF metadata to display this information, for example
[exiftool](https://exiftool.org/). 

### Orthophoto
The `orthophoto` folder contains GeoTiffs with all the images "mosaiced" into a single
image. The file named `odm_orthophoto.original.tif` or `pix4d_orthophoto.original.tif`
corresponds to the original file produced by either ODM or Pix4D. The other GeoTiffs are
versions used for publishing the data on the [SeaBee Geo-Visualization
Portal](https://geonode.seabee.sigma2.no/).

Note that for Pix4D data, the original mosaic is saved using floating-point numbers and
has units of reflectance (see [Pix4D note on reflectance
maps](https://support.pix4d.com/hc/en-us/articles/202739409)). 

### Other
The `other` folder contains additional or non-standard files, added as part of the
original dataset or produced by the photogrammetry software.  

### Report
The `report` folder contains a PDF report describing the photogrammetry processing, and
in some cases some additional files. The report is a good source of information
regarding data quality, dataset size etc. 

### Config file
The file `config.seabee.yaml` contains metadata about the dataset. The metadata includes
information about the original data (location, time, number of images) and parameters
for controlling processing and publishing of datasets via SeaBee.


