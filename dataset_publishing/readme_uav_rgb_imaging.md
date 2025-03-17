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


## Photogrammetry software
Two different photogrammetry programs were used,
[OpenDroneMap](https://www.opendronemap.org/) and [Pix4D](https://www.pix4d.com/). You
can determine which was used for your specific dataset by reading the report under
the `report` subfolder.  

The datasets were processed using OpenDroneMap as implemented in the SeaBee
[data processing pipeline](https://seabee-no.github.io/documentation/). 

Some datasets
 were also processed with Pix4D.

Details regarding software settings were unfortunately not available during the writing
of this readme file. 
