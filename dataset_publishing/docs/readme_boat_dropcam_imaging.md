# Massimal dropcam imaging dataset (boat)
This file is a description of a dataset with underwater video and geotagged images
collected by towing a camera from a boat. The remainder of the file describes the
research project that collected the data, the camera equipment used, and the image
geotagging process.

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
used for underwater imaging. The camera was attached by a "drop camera" using a GoPro
["Jaws" clamp](https://gopro.com/en/no/shop/mounts-accessories/jaws/ACMPM-001.html). The
drop camera was a camera custom-built by NIVA, enabling live streaming of video via a
cable to a monitor in the boat. It was possible to record video with the drop camera,
but the video quality was very poor, and a GoPro camera was therefore used for
recording. However, the live drop camera feed was very useful for adjusting the depth of
the camera, keeping it at 1-3 meters above the sea floor. 

The camera was towed at approximately 0.5 m/s (1 knot). 

## Position logging
Position was logged using a mobile phone with the ["Skippo"](https://www.skippo.no/)
app, an app for marine navigation. The start and end positions of video recording were
marked with waypoints. The position log was exported as a CSV file. 

Note that the position log represents the position of the boat and not the camera. Since
the camera was towed, it was generally a few meters behind the boat, but this offset is
not consistent. The accuracy of the position is estimated to approximately 10 meters. 

## Time synchronization
Unfortunately, the clock on the camera and the clock on the mobile phone used for
position logging were not synchronized. However, the marking of waypoints at the start
of each transect corresponded to the start of the video recording. This information was
used to add appropriate time offsets when geotagging. Note, however, that there may be a
few seconds delay between the two, which translates to a small error in position when
geotagging.     

## Image geotagging
The geotagged images in the dataset were extracted using an early version of the
**vidtransgeotag** Python module (see [GitHub
](https://github.com/mh-skjelvareid/vidtransgeotag) and
[10.5281/zenodo.14974704](https://doi.org/10.5281/zenodo.14974704)). The software is
written in Python, but uses the [FFMPEG](https://www.ffmpeg.org/) library (via [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)) to read metadata and extract images from
video. 

The input to the geotagging process is two data streams: A video file and a CSV file
with positions. By using the time stamps in both data streams, it is possible to
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


