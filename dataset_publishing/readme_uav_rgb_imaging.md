# MASSIMAL multispectral UAV imaging dataset
This file is a description of a dataset with RGB images of shallow coastal areas in
Norway, collected using a UAV (drone). The remainder of the file describes the research
project that collected the images, the UAVs and cameras used, photogrammetry
post-processing, and the structure of the dataset.


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


## UAVs and cameras
A selection of different UAVs and cameras were used for data collection: 

- DJI Phantom 4
- DJI Mavic 2 Pro
- DJI Matrice 210
- AgEagle eBee 

Unfortunately, the original images collected with the DJI Matrice 210 and AgEagle eBee
were not available for publishing at the time of writing of this readme file. However,
the datasets collected with the DJI Phantom 4 or the DJI Mavic 2 Pro do include the
original images. In these cases, the cameras were an integrated part of the UAV. Consult
the EXIF metadata of the original images for additional information about the cameras on
these UAVs.


## Photogrammetry software
Many datasets were initially processed using [Pix4D](https://www.pix4d.com/). During the
Massimal project period the SeaBee [data processing
pipeline](https://seabee-no.github.io/documentation/) was also established, opening the
possibility to process the images using [OpenDroneMap](https://www.opendronemap.org/)
(ODM), an open-source alternative to Pix4D. Some datasets have been processed only with
Pix4D or ODM, and some have been processed with both. 

For datasets processed with ODM, the original images are included, as well as the output
products of the processing. For datasets processed with Pix4D, the processing outputs
(at least the orthomosaic) is included, but the original images are not included. Open
the report under the `report` subfolder to determine which method was used for a
specific dataset.  

Details regarding software settings were unfortunately not available during the writing
of this readme file. 

## Data format 
A typical dataset is organized as follows:

    ├── dem/
    ├── images/
    ├── orthophoto/
    ├── other/
    ├── report/
    └── config.seabee.yaml

Note that not all folders are present in every dataset. The following describes the
contents of each folder in the dataset. 

### DEM (digital elevation model)
The `dem` folder contains digital surface models (DSM) and digital terrain models (DTM,
models where vegetation, buildings etc. has been removed). 

The photogrammetry software was not set up for calculating under-water
elevation/bathymetry, but in some shallow areas the models still clearly show the shape
of the seabed under water. Note, however, that the models cannot be used to accurately
measure water depth in absolute units.

### Images
The `images` folder contains the original images used for photogrammetry. A large amount
of additional information is saved in each image's EXIF metadata, including UAV position
and speed, and gimbal orientation. Use software capable of reading EXIF metadata to
display this information, for example [exiftool](https://exiftool.org/). 

### Orthophoto
The `orthophoto` folder contains RGB GeoTiffs with all the images "mosaiced" into a
single image. The file named `odm_orthophoto.original.tif` or
`pix4d_orthophoto.original.tif` corresponds to the original file produced by either ODM
or Pix4D. The other GeoTiffs are versions used for publishing the data on the [SeaBee
Geo-Visualization Portal](https://geonode.seabee.sigma2.no/).

Note that the images have been color balanced to produce a visually pleasing
orthomosaic. The colors do not directly represent radiance or reflectance
(see [Pix4D article on reflectance map vs
orthomosaic](https://support.pix4d.com/hc/en-us/articles/202739409)).  

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
