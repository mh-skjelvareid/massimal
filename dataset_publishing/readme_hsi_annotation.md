# Annotations of Massimal hyperspectral images
This files decribes a dataset of image annotations for hyperspectral images of coastal
areas collected using a UAV (drone). This is not an independent dataset - the original
hyperspectral images are required to really make use of the annotations, and these have
to be downloaded separately. The description in this file is also general, and not
connected to a specific location or hyperspectral image dataset.

The remainder of this file decribes the research project that collected the data, the
tools and taxonomy used for annotations, and the annotation data format and structure.

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


## Hyperspectral imaging system 
The hyperspectral images that were the basis for annotation are described in detail in
the datasets containing these images. However, a short summary is given here:

- **Imaging system**: The images were collected using an "[Airborne Remote Sensing
System](https://resonon.com/hyperspectral-airborne-remote-sensing-system)" manufactured
by Resonon. The system included a hyperspectral camera, an IMU measuring camera position
and orientation, and a spectrometer measuring downwelling irradiance, 
- **Hyperspectral camera**: The system used a [Pika-L](https://resonon.com/Pika-L)
hyperspectral camera with 300 spectral channels covering the a spectral range of
400-1000 nm. The camera is a push-broom sensor with 900 spatial pixels, making all
images 900 pixels wide (across-track). The height of each image is 2000 pixels or less.
- **Unmanned aerial vehicle** (UAV) / drone: A DJI MAtrice 600 Pro was used as the
  camera platform. The camera was mounted to a DJI
  [Ronin-MX](https://www.dji.com/no/ronin-mx) 3-axis gimbal. 
- **Ground sampling distance**: The UAV flying altitude and speed was coordinated with
  the camera field of view and framerate to give approximately equal sampling distance
  along- and across-track. For most images, the ground sampling distance was 3.5-4.0 cm.   

## Ground truth
Various types of ground truth, i.e., direct observations of the ground or seafloor, were
used in the annotation process. These included: 
- Geotagged images, aquired either from a boat, an unmanned surface vehicle (USV), or
  while walking (in the intertidal zone).
- Images and video from transects, acquired using a handheld camera while snorkeling, a
  camera towed behind a boat, or using an ROV. The position and shape of the transect
  was known, but images / video were not directly geotagged. In some cases markers
  placed along the transect functioned as "milestones", enabling calculation of position
  relative to the transect end points.
- Point observations done by lowering a "drop-camera" from a boat to the seafloor.
  At each point, the position was logged, and observations on seafloor substrate,
  vegetation, and depth were written down as field notes.    

Ground truth data from the project is published as separate datasets, and the
methodology for collecting ground truth are described in more detail in these.  

## Selection of images for annotation
The annotations in the dataset were done on a subset of the hyperspectral images
collected in the Massimal project. Only a subset of images was annotated because:

- Only a subset of the images were easy to interpret directly, or had good coverage of
  ground truth data. 
- Annotation is time- and labor-intensive, and there were limited resources available
  for annotation in the project.  

## Remote sensing images
The Massimal project collected two types of images using UAVs:
- **Hyperspectral images**: Images with 300 spectral channels, collected using a
  push-broom imager. An IMU was used to continuously measure the camera position and
  orientation, enabling georeferencing of the images. However, the heading and altitude
  measurements were of variable quality, and the georeferencing was only accurate to
  approximately 2-5 meters.    
- **Color images (RGB)**: Collected using small commercial drones (e.g. DJI Phantom 4,
  DJI Mavic). These images have much less spectral information than the hyperspectral
  images, but the image collection method is faster and can cover a large area in a
  short amount of time. Acquring images also allow the use of photogrammetry software
  such as [Pix4d](https://www.pix4d.com) or
  [OpenDroneMap](https://www.opendronemap.org), which accurately merge the images into a
  large mosaic. Using a few accurately measured ground control points also enables
  "warping" the mosaic to position each pixel with an accuracy within approximately 10
  cm. Such orthomosaics are useful as high-resolution base maps when interpreting ground
  truth and annotating the hyperspectral images. 

## Overlaying ground truth on remote sensing images
The [QGIS](https://qgis.org) software was used to visualize ground truth positions
overlayed on one or more layers of remote sensing images. 

When using geotagged images, the position of each image was saved in a vector point
layer. The attribute table of this layer included the file names of the image files.
This enabled direct display of the images within QGIS, either via the [identify
features](https://docs.qgis.org/3.40/en/docs/user_manual/introduction/general_tools.html#identify)
tool, defining the image as an attachment, or as a [HTML "map tip"](https://opengislab.com/blog/2020/8/23/mapping-and-viewing-geotagged-photos-in-qgis).

## Annotation methodology
The ground truth information inherently has a geospatial nature, and annotation of
remote sensing images is often done using a geospatial data type such as a vector
polygon. However, in the case of the hyperspectral images to be annotated, the accuracy
of the georeferencing was only to within a few meters. The position accuracy of ground
truth data also varied from centimeter to meter scale. If annotations were made as
vector polygons solely based on the ground truth positions, the accuracy of annotations
relative to the hyperspectral images would be only within a few meters. This is
problematic when the features to be annotated (e.g. a patch of seagrass meadow) also is
on the scale of a few meters.      

An alternatice approach was used when making annotations in this project. Rather than
defining geospatial polygons, annotations were made directly on the pixels of the
hyperspectral image.

## "Hasty" annotation tool
