# Massimal underwater imaging dataset (boat)
This file is a description of a dataset with geotagged underwater images collected from
a boat. The description is general and not connected to one
specific location or dataset. The remainder of the file describes the research project
that collected the data, the camera equipment used, and the image geotagging
process.

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


## Underwater imaging
A GoPro camera (model [Hero 8 Black](https://en.wikipedia.org/wiki/GoPro#HERO8)) was
used for underwater imaging. The camera was used with its watertight casing, and was
mounted to a wooden pole using a GoPro "Large Tube Mount". The pole was then lowered
into the water along the side of the boat, with the camera either pointing directly
downwards or slightly forwards. The pole was long enough to manually adjust the camera
depth approx. 1-2 meters, allowing the camera operator to place the camera closer to the
seafloor (for better visibility) in deeper waters.  

## Trimble Catalyst GNSS system
The position of the camera was measured using a [Trimble
Catalyst](https://geospatial.trimble.com/en/products/software/trimble-catalyst) GNSS
system; a lightweight GNSS receiver connected to a mobile phone. The Catalyst system has
a subscription-based system with multiple levels of accuracy, and the highest accuracy
(approx 1 cm) was used during field work. However, the GNSS receiver could not be placed
directly above the camera, and the actual accuracy of geotagging is probably approx. 1
meter. 

The Trimble Catalyst position data was logged using the Android app [Ultra GPS
Logger](https://play.google.com/store/apps/details?id=com.flashlight.ultra.gps.logger)
using a "professional" subscription (for high-accuracy logging). The log was exported to
a CSV file. 

## Time synchronization
[GoPro Labs](https://github.com/gopro/labs) offers a firmware update that enables
updating of the internal GoPro clock by simply pointing the camera at a [QR
code](https://gopro.github.io/labs/control/precisiontime/). This method was used before
collection of almost all datasets, to enable high-accuracy time synchronization between the
camera and the position log. In cases where this method was not used, the time
offset between the camera and the USV was estimated by trial-and-error; geotagging
images, looking for well-known landmarks in the images, and checking if the landmarked
images were geotagged correctly. If not, the time offset was adjusted and the process
repeated.    

## Image geotagging
The input to the geotagging process is two data streams: A video file and a CSV file
with positions. By using the time stamps in the both data streams, it is possible to
calculate which position points overlap in time with the video, and what exact time in
the video each of these points correspond to. It is then possible to extract images from
the video at these times and link them with the position information.

The positions in the CSV are sometimes very densely sampled in time (multiple times per
second). Also, if the camera does not move for a period in the video, there may be very
many CSV rows corresponding to approximately the same position. To avoid both these
problems, the positions can be filtered based on relative distance before using them for
extracting images. The pseudocode for the filtering algorithm is as follows:

    last_included_position = first position in dataset
    for every position in dataset:
        if distance(position,last_included_position) > minimum distance:
            add position to filtered positions
            set last_included_position = position

The geotagged images in the dataset were extracted using an early version of the
**vidtransgeotag** Python module (see [GitHub
](https://github.com/mh-skjelvareid/vidtransgeotag) and
[10.5281/zenodo.14974704](https://doi.org/10.5281/zenodo.14974704)). The software is
written in Python, but uses the [FFMPEG](https://www.ffmpeg.org/) library (via [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)) to read metadata and extract images from
video. 

## Underwater image quality
The depth and quality of the water between the camera and the seafloor has a very
significant effect on the image quality: In very shallow areas, about 1-2 meters, the
image is crisp and clear, while at larger depths, the image becomes green-tinted and
more blurry. Even if the camera was set to automatic color balancing, it could not fully
compensate for the green tint. 

For many of the datasets, an algorithm for color balancing and contrast stretching has
been applied to the images. The algorithm is very simple:

- for every color band in the image:
    - calculate intensity percentile values representing lower and upper edge of pixel
      intensity distribution (usually 1% and 99%)
    - "stretch" the image band so that the range between percentile values spans the full
      dynamic range of the image (0-255 for 8-bit images) 

This algorithm often results in less green tint and a more visually pleasing image.
Note, however, that for images where the contrast in the original image is very low (i.e.
at large water depths), the algorithm can result in "extreme" contrast and odd-looking
images. 
