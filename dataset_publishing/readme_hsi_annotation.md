# Annotations of MASSIMAL hyperspectral images
This file describes a dataset of image annotations for hyperspectral images of coastal
areas collected using a UAV (drone). This is not an independent dataset - the original
hyperspectral images are required to make use of the annotations, and these have to be
downloaded separately. The description in this file is also general, and not connected
to a specific location or hyperspectral image dataset.

The remainder of this file describes the research project that collected the data, the
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
- [SeaBee data portal with MASSIMAL
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
- **Ground sampling distance**: The UAV flying altitude and speed was adjusted to the
  camera framerate and field of view and, to give approximately equal sampling distance
  along- and across-track. For most images, the ground sampling distance was 3.5-4.0 cm.   

## Ground truth
Various types of ground truth, i.e., direct observations of the ground or seafloor, were
used in the annotation process. These included: 
- **Geotagged images**, acquired either from a boat, an unmanned surface vehicle (USV),
  or whilst walking in the intertidal zone.
- **Images and video from transects**, acquired using a handheld camera while
  snorkeling, a camera towed behind a boat, or using an ROV. The position and shape of
  the transect was known, but images / video were not directly geotagged. In some cases
  markers placed along the transect functioned as "milestones", enabling calculation of
  position relative to the transect end points.
- **Point observations** done by lowering a "drop-camera" from a boat to the seafloor.
  At each point, the position was logged, and observations on seafloor substrate,
  vegetation, and depth were written down as field notes.    

Ground truth data from the project is published as separate datasets, and the
methodology for collecting ground truth are described in more detail in these.  

## Selection of images for annotation
The annotations in the dataset were done on a subset of the hyperspectral images
collected in the MASSIMAL project. Only a subset of images was annotated because:

- Only a subset of the images had good coverage of ground truth data, or were easy to
  interpret directly. 
- Some images mainly cover land, and were not of interest in the MASSIMAL project.
- Annotation is time- and labor-intensive, and there were limited resources available
  for annotation in the project.  

## Remote sensing images
The MASSIMAL project mainly collected two types of remote sensing images using UAVs:
- **Hyperspectral images**: Images with 300 spectral channels, collected using a
  push-broom imager, as described above. The camera IMU was used to continuously measure
  the camera position and orientation, enabling georeferencing of the images. However,
  the heading and altitude measurements were of variable quality, and the georeferencing
  was only accurate to approximately 2-5 meters.    
- **Color images (RGB)**: Collected using small consumer drones (mainly DJI Phantom 4 or
  DJI Mavic 2 Pro). These images have much less spectral information than the
  hyperspectral images, but the image collection method is faster and can cover a large
  area in a short amount of time. RGB images can also be processed using photogrammetry
  software such as [Pix4d](https://www.pix4d.com) or
  [OpenDroneMap](https://www.opendronemap.org), which accurately merge the images into a
  large mosaic. Using a few accurately measured ground control points, the mosaic can be
  georeferenced with an average position accuracy of a few centimeters. Such
  orthomosaics are useful as high-resolution base maps when interpreting ground truth
  and annotating the hyperspectral images. 

## Overlaying ground truth on remote sensing images
The [QGIS](https://qgis.org) software was used to visualize multiple layers of
geospatial information. These layers were usually (numbered frow lowest to highest
layer):

1. Background map layer with global coverage, e.g. from Google Maps.
2. High-resolution RGB mosaic (if available).
3. Georeferenced hyperspectral images.
4. Ground truth reference points, e.g. geotagged image positions, transect end points,
   etc.

When using geotagged images, the position of each image was saved in a vector point
layer. The attribute table of this layer included the file names of the image files.
This enabled direct display of the images within QGIS, either via the [identify
features](https://docs.qgis.org/3.40/en/docs/user_manual/introduction/general_tools.html#identify)
tool, defining the image as an attachment, or as a [HTML "map
tip"](https://opengislab.com/blog/2020/8/23/mapping-and-viewing-geotagged-photos-in-qgis).

Additional ground truth information (images, video, notes) related to the area being
annotated was also reviewed in the process. With all this information available, the
annotator was able to get an general impression of the area, and interpret the scene
visible in the hyperspectral images. 

## Annotation methodology

### Semantic segmentation
Annotation of remote sensing images is often done using a geospatial data type such as a
[vector polygon](https://support.esri.com/en-us/gis-dictionary/polygon). However, in the
case of the hyperspectral images to be annotated, georeferencing was only accurate to
within a few meters. The positional accuracy of ground truth data also varied from
centimeter to meter scale. If annotations were made as vector polygons solely based on
the ground truth positions, the accuracy of annotations relative to the hyperspectral
images would be only within a few meters. This is problematic when the size of features
to be annotated (e.g. a patch of seagrass meadow) also are on the scale of a few meters.      

An alternative approach was used when making annotations in this project. Rather than
defining geospatial polygons, annotations were made directly on the pixels of the
hyperspectral image. This enables the annotator to use and follow the visible features
of the image, rather than "blindly" using a georeferenced polygon based only on ground
truth. Since the annotations are made in the same raster coordinate system as the
hyperspectral images, the same georeferencing can be used for both images and
annotations. Also, if georeferencing is re-run with changed parameters, this will not
change the position of the annotations relative to the image. 


### "Hasty" annotation tool
While geospatial information was viewed in QGIS, the annotation was done using
[Hasty](https://app.hasty.ai), an online annotation system that is [part of
CloudFactory's AI Data Platform](https://hasty.cloudfactory.com/). Hasty was chosen for
its

- intuitive user interface
- organization of data into projects and datasets
- annotation options, using both classes and attributes
- ergonomic annotation tools
- sharing and collaboration features
- selection of export formats

However, Hasty is *not designed for geospatial data*. Annotations were therefore made on
RGB versions of hyperspectral images, without any geospatial context.

### Annotation taxonomy and hierarchy
Initially, a few datasets were annotated based on the nature types present at the
particular locations. Later in the project, a common annotation taxonomy was developed,
inspired by the hierarchical taxonomy used by the SeaBee infrastructure project (see
https://github.com/SeaBee-no/annotation , and ["Drone and ground-truth data collection,
image annotation and machine learning: A protocol for coastal habitat mapping and
classification"](https://doi.org/10.1016/j.mex.2024.102935)  by K.Ø. Kvile et al.). 

The hierarchical structure was used to allow the specificity of annotations to be
adapted to the amount of information available:

- Using detailed annotation classes (e.g. species classes like *Fucus vesiculosus*) where
  accurate, high-quality information is available (e.g. accurately geotagged
  high-quality images, or field notes).
- Using less detailed annotations classes (e.g. "rockweed" or "algae") where there is
  less information (e.g. low-quality underwater images, or UAV images only).

Using such a hierarchy gives the annotator more flexibility to communicate "everything
they know" about the scene. However, it makes subsequent training of machine learning
models more complicated. It is typically necessary to group some classes together,
exclude some classes, etc., before using the annotations for supervised learning. Such
adaptations may vary based on the purpose of the machine learning model, and we find it
useful to separate the annotation taxonomy from the final classes used in a
classification model.  

The hierarchical taxonomy used in the project is listed below. This is based on the
3-level system used by SeaBee, but has been extended to allow additional levels and
intermediate classes where needed.

- 1 Deep water
- 2 Algae
    - 2.1 Brown algae
        - 2.1.1 Kelp
            - 2.1.1.1 Laminaria hyperborea
            - 2.1.1.2 Laminaria digitata
            - 2.1.1.3 Sacchoriza polyides
            - 2.1.1.4 Saccharina latissima
            - 2.1.1.5 Alaria esculenta
        - 2.1.2 Rockweed
            - 2.1.2.1 Rockweed, hydrolittoral
                - 2.1.2.1.1 Ascophyllum nodosum
                - 2.1.2.1.2 Fucus vesiculosus 
                - 2.1.2.1.3 Fucus serratus
                - 2.1.2.1.4 Halidrys siliquosa
            - 2.1.2.2 Rockweed, geolittoral 
                - 2.1.2.2.1 Fucus spiralis
                - 2.1.2.2.2 Pelvetia canaliculata
        - 2.1.3 Brown algae, other
            - 2.1.3.1 Chorda filum 
            - 2.1.3.2 Desmarestia aculeata
    - 2.2 Green algae
    - 2.3 Red algae
        - 2.3.1 Coralline algae 
            - 2.3.1.1 Maerl
    - 2.4 Turf
- 3 Seagrass
    - 3.1 Zostera marina
- 4 Substrate
    - 4.1 Rock
        - 4.1.1 Bedrock
        - 4.1.2 Boulder
        - 4.1.3 Cobble
        - 4.1.4 Gravel
    - 4.2 Sediment
        - 4.2.1 Sand
        - 4.2.2 Mud
- 5 Animals
    - 5.1 Mussels
        - 5.1.1 Mytilus edilus
- 6 Human activity 
    - 6.1 Trawl tracks

The selection of classes may seem somewhat arbitrary, as new classes were added as
needed during the annotation process. There are also a few classes that were included in
the taxonomy but have not (so far) been used. However, the taxonomy at least provides a
common framework for building machine learning models based on all annotated datasets in
the project. 

*Note that Hasty does not support hierarchical annotations classes. In the annotation
output from the program, all classes have a "flat" structure, and have to be
manually organized into the hierarchy listed above.*

In addition to the classes described above, some classes were defined with *attributes*
such as "density" and "turf abundance". These attributes were only used for some
annotations, where there was enough information available to set a value for the
attribute. See descriptions of JSON files below for more information.  


## Annotation data format
Each annotated dataset has three folders named `rgb_images`, `segmentation_masks`, and
`annotations_json`. The contents of these folders is described below. Note that some
annotation datasets correspond to multiple hyperspectral image datasets (multiple
flights performed on the same day and location). 

### RGB images
The `rgb_images` folder contains the original images that the annotations were made
"on". These are RGB versions of hyperspectral images, extracted at 3 wavelengths
representing red, green and blue, with each color channel being "stretched" to yield
contrast and color balance that is suitable for annotation. Some images have also been
"glint corrected", i.e. reflections from the water surface have been estimated and
subtracted from the image.  

### Segmentation masks
The `segmentation_masks` folder contains a JSON file called "label_classes.json", which
defines the different annotation classes, their names and descriptions, and their color.
The "png_index" field defines the values for red, green and blue (integer in range
0-255) used when rendering the annotations. 

The annotations themselves are formatted as PNG color images. Note that black (0,0,0)
corresponds to background (i.e., not annotated). Note also that attribute information is
not included in this representation of the annotations. If attribute information is
needed, use the Hasty JSON file in the `annotations_json` folder.

### Annotations in JSON format
The `annotations_json` folder contains two JSON files, exported in "Hasty" and "COCO"
format. The [COCO
format](https://wiki.cloudfactory.com/docs/userdocs/export-formats/coco-dataset-format#coco-dataset-format)
is well established and supported by multiple libraries and tools. The proprietary
[Hasty
format](https://wiki.cloudfactory.com/docs/userdocs/export-formats/hasty-json-v1-1#hasty-json-v1-1)
is based on the COCO format, and adds some additional metadata, including attribute
information. 

Both formats define the annotation masks directly in the JSON file using [run-length
encoding](https://en.wikipedia.org/wiki/Run-length_encoding) (RLE). Getting a boolean
mask defining a class thus requires decoding the RLE. Code examples for RLE decoding are
available via the links to the COCO and Hasty formats.
