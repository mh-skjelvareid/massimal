# Massimal sonar dataset
This file is a description of a dataset with sonar data collected by an unmanned surface
vehicle (USV). The description is general and not connected to one specific location or
dataset. The remainder of the file describes the research project that collected the
data, the USV and sonar equipment used, and details regarding raw and processed data.

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

## Unmanned surface vehicle (USV)
An ["Otter"](https://www.maritimerobotics.com/otter) USV manufactured by Marine Robotics
(Trondheim, Norway) and owned by the [SeaBee](https://seabee.no/) research
infrastructure was used for data collection. The USV was usually operated in
"auto-pilot" mode, navigating using pre-planned waypoints, but it occasionally had to
be piloted manually, typically in very shallow areas.  

## Sonar equipment
The sonar used to collect this dataset is a [BioSonics MX Aquatic Habitat Echo
Sounder](https://www.biosonicsinc.com/products/mx-aquatic-habitat-echosounder/) owned by
[SeaBee](https://seabee.no/), "Norwegian infrastructure for drone-based research,
mapping monitoring in the coastal zone". 

## Raw data
Raw data for a single dataset is saved as a collection of files with the following
format: `<yyyymmdd_HHMMSS>_mx.rtpx`, where `yyyymmdd_HHMMSS` corresponds to the date and
time when the data was recorded. Each file contains up to approximately 4500 pings,
corresponding to approximately 56 MB of data. The files also contain metadata about the
position of each ping and the type of transducer. 

Raw data files can be opened using the software [Visual
Aquatic](https://www.biosonicsinc.com/download/visual-aquatic-1-0/) made by Biosonics.
Downloading the software requires user registration, but the software is free. 

## Data processing
Processing has been performed using only the default values in the Visual
Aquatic software (version
1.0.0.13146). All possible processing steps have been performed:

- Bottom detection (bathymetry)
- Submerged plant coverage and height estimation
- Bottom echo feature extraction and bottom type clustering 

## Processed data
Processed data is exported as a CSV file. Each row in the file corresponds to a "report"
which is calculated based on a small collection of consecutive sonar "pings" (typically 5). The with the following fields:
- **Latitude_deg**: Latitude in decimal degrees
- **Longitude_deg**: Longitude in decimal degrees
- **GPSQuality**: Text string describing GPS quality (accuracy), e.g. "Differential".
- **Altitude_mReMsl**: Altitude of GPS antenna (in meters) relative to mean sea level.
- **AltitudeQuality**: Altitude quality of GPS fix, 1 is valid, 0 is invalid
- **Time**: Date and exact time of report
- **FileName**: Name of raw data file containing pings in report
- **Transducer name**: Description of transducer (always "204,8 kHz 8,8° Single") in this
  dataset.
- **Transducer number**: Always 1 in this dataset.Used to identify transducers when multiple
  transducers are used in a pinging sequence.  
- **ReportNumber**: Numbering of reports, starting at 1 and increasing by 1 for each line. 
- **FirstPingNumber**: Number of first ping included in report
- **LastPingNumber**: Number of last ping included in report
- **BottomStatus**: Either "Valid" or "Invalid". If "Valid", the bottom has been detected in
  the data and the "BottomElevation_m" field has a valid number.
- **BottomElevation_m**: Depth (in meters) given as a negative elevation (0 at surface,
  negative below).
- **PlantStatus**: Either "Valid" or "Invalid". If "Valid", the top of the plant canopy has
  been detected in the data and the "PlantHeight_m" field has a valid number.
- **PlantHeight_m**: Estimated height of submerged plants, in meters.
- **PercentCoverage**: Estimated plant coverage (in %). 
- **BottomTypeStatus**: Either "Valid" or "Invalid". If "Valid", "BottomType" has a valid
  value. 
- **BottomType**: Bottom type number (integer, 1-4). See [notes on bottom type clustering](#notes-on-bottom-type-clustering).
- **BottomType1Membership, BottomType2Membership, BottomType3Membership,
  BottomType4Membership**: Membership weights after fuzzy clustering. See [notes on bottom type clustering](#notes-on-bottom-type-clustering).
  

### Notes on bottom elevation, sea level, and bathymetry
Note that the bottom elevation as given in the field "BottomElevation_m" is a measure of
the distance from the sonar to the sea bottom at a specific time. This measurement does
not take the effect of sea level into account. For the locations visited during Massimal
field work, differences between low and high tide can be up to 3 meters. 

The Norwegian Mapping Authority provides a [web service ("Se
Havnivå")](https://kartverket.no/en/at-sea/se-havniva) and an
[API](https://vannstand.kartverket.no/tideapi_en.html) for retrieving sea level data
for Norway. We recommend combining this data with the "BottomElevation_m" column to
calculate water depth relative to mean sea level. 


### Notes on bottom type clustering
The bottom echo is "clustered", i.e. classified as belonging to a unlabelled "bottom
type", by using the [fuzzy c-means clustering
  algorithm](https://en.wikipedia.org/wiki/Fuzzy_clustering#Fuzzy_C-means_clustering)

The basis for clustering is 15 features extracted from the bottom echo: "Fractal
dimension, E1, E1’, and 12 spectral moments", according to the Visual Aquatic
documentation. The features are [z-score
normalized](https://en.wikipedia.org/wiki/Standard_score), and
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is performed on the
normalized features. The 6 first principal components are kept, and these components are
the inputs used for fuzzy c-means clustering.

The number of clusters is an adjustable input parameter, and can be changed if the
clustering is rerun in Visual Aquatic (from raw data).

Clustering is an unsupervised method, and the cluster number (e.g. 1-4) is arbitrary.
The clusters are learned from the dataset, and change from one dataset to the next.
Bottom types and cluster memberships are thus not comparable across datasets.   




