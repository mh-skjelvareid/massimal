# Massimal
A collection of code related to the Massimal research project on UAV hyperspectral
imaging of underwater vegetation.

This code is still in development, and has not yet been prepared for "production use".
However, researchers / developers working on similar topics may find parts of the code
useful for their own applications. 

The repository includes modules related to:
- Reading and writing hyperspectral images (hyspec_io)
- Rendering images, hyperspectral and others (image_render)
- Preprocessing hyperspectral images, including sun/sky glint correction and inpainting
  of missing pixels (preprocess)
- Reading and processing annotation data (generated by the
  ["Hasty"](https://app.hasty.ai/) annotation tool)
- Machine learning and deep learning for hyperspectral data (hyspec_ml, hyspec_cnn)
- Generating georeferenced ground truth photos based on geoloctaion logs and GoPro video
  / images (video_transect)

## Related repository: MassiPipe
The [MassiPipe repository](https://github.com/mh-skjelvareid/massipipe) has also been
developed as part of the Massimal project. MassiPipe contains tools for processing
hyperspectral images and irradiance spectra in a data pipeline, including calibration
(conversion to radiance), sun/sky glint correction, and reflectance conversion.  

## Repository structure
The repository has two main parts;
- src/massimal: Pure python modules which contains generel classes or functions for
  processing data (hyperspectral images / underwater video / geolocation)
- dataset_specific: A collection of Jupyter Notebooks for processing specific datasets.

In addition to these, the repository contains the following folders
- annotation: Files related to a hierarchical system for coastal marine habitat
  annotation
- dataset_publishing: Files related to publishing of data from the Massimal project
- tests: A small collection of tests (pytest tests and Jupyter notebooks) related to the
  Python modules under src/massimal .


## Installation
Massimal uses a number of external libraries. Some of these are easily installed using
pip, but others (non-Python applications) are easier to install using conda. We
recommend using conda for creating a virtual environment and installing some of the
dependencies, and then installing the MassiPipe package with pip, which will install the
remaining Python dependencies (listed in pyproject.toml).

Create conda environment, installing from conda-forge channel (change "massimal"
environment name and python version to your preference):

    conda create -n massimal -c conda-forge python=3.10 graphviz gdal rasterio ffmpeg

Download massimal from the [massimal GitHub
repository](https://github.com/mh-skjelvareid/massimal) (download as zip or use git
clone). Navigate to the root folder of the repo and install using pip ("." indicates
installation of package in current directory):

    conda activate massimal
    pip install .



## Website
The [Massimal website](https://en.uit.no/project/massimal) has information about the
project, it's members, publications etc.

## Partners
 The Massimal project is a collaboration between
 - [UiT - the Arctic University of Norway](https://en.uit.no/startsida)
 - [NIVA - the Norwegian Institute for Water and the
   Environment](https://www.niva.no/en)
 - [Nord University](https://www.nord.no/en)

## Funding
The Massimal project is funded by
- The Norwegian Research Council (grant number
  [301317](https://prosjektbanken.forskningsradet.no/project/FORISS/301317))
- UiT the Arctic University of Norway (internal funding)
