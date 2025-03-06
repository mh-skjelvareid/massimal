# Massimal ground imaging dataset (walking)
This file is a description of a dataset with geotagged images from the intertidal zone
collected by walking. The description is general and not connected to one
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


## Ground imaging
A GoPro camera (model [Hero 8 Black](https://en.wikipedia.org/wiki/GoPro#HERO8)) was
used for imaging. The camera was mounted to a pole using a GoPro "Large Tube Mount". The
camera pointed downwards and slightly forwards so that the end of the pole was in view.
The ground was imaged by recording video while walking around in the intertidal zone.
The walking path did not follow any pre-planned "transect", but was conducted by walking
around inside a target area, while monitoring what part of the area had already been
covered. 

## Trimble Catalyst GNSS system
The position of the camera was measured using a [Trimble
Catalyst](https://geospatial.trimble.com/en/products/software/trimble-catalyst) GNSS
system; a lightweight GNSS receiver connected to a mobile phone. The Catalyst system has
a subscription-based system with multiple levels of accuracy, and the highest accuracy
(approx 1 cm) was used during field work. The GNSS receiver was placed
directly above the camera, and the accuracy of the camera position is therefore
relatively high.

The Trimble Catalyst position data was logged using the Android app [Ultra GPS
Logger](https://play.google.com/store/apps/details?id=com.flashlight.ultra.gps.logger)
using a "professional" subscription (for high-accuracy logging). The log was exported to
a CSV file. 

## Time synchronization
The time logged by the GNSS system and the camera was synchronized by starting the
camera recording and the position logging simultaneously. Note, however, that there may
be up to 1-2 seconds time offset between these, due to slight delays in the recording
process. We estimate that this translates to a systematic position error of approx. 1
meter (depending on the direction and speed of walking).
   

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

