# Massimal
 A collection of code related to the Massimal research project on UAV hyperspectral imaging of underwater vegetation.

## Installation
This code is still in development and has not  been organized and prepared for widespread distribution. However, if you do want to run parts of this code, you can find requirements in the [requirements_python.txt](requirements_python.txt) and [requirements_other](requirements_other.md) files. 

The requirements are quite loosely defined, with few limits on version numbers, as no obvious version conflicts have been detected yet. If you come across problems regarding package versions, please submit an [issue](https://github.com/mh-skjelvareid/massimal/issues) describing it. 

Massimal uses a number of external libraries. Some of these are easily installed using
pip, but others (non-Python applications) are easier to install using conda. We
recommend using conda for creating a virtual environment and installing some of the
dependencies, and then installing the MassiPipe package with pip, which will install the
remaining Python dependencies (listed in pyproject.toml).

Create conda environment, installing from conda-forge channel (change "massimal" environment name and python version to your preference):

    conda create -n massimal -c conda-forge python=3.10 graphviz gdal rasterio ffmpeg

Download massimal from the [massimal GitHub repository](https://github.com/mh-skjelvareid/massimal) (download as zip or use git clone). Navigate to the root folder of the repo and install using pip ("." indicates installation of package in current directory):

    conda activate massimal
    pip install .

## Tests
The Massimal project is mainly a data science project, and much of the development is done in Jupyter notebooks. When "repeated tasks" are discovered, the code is generalized and placed in the python module (under src/massimal). Ideally, all the code should have unit tests, but it's hard to write tests for some of the data science-relatated code, and unit tests have not been prioritized yet. For now, a selection of Jupyter notebooks are included in the folder "test", as a form of more "manual" testing. 

## Website
The [Massimal website](https://en.uit.no/project/massimal) has information about the project, it's members, publications etc.

## Partners
 The Massimal project is a collaboration between
 - [UiT - the Arctic University of Norway](https://en.uit.no/startsida)
 - [NIVA - the Norwegian Institute for Water and the Environment](https://www.niva.no/en)
 - [Nord University](https://www.nord.no/en)

## Funding
The Massimal project is funded by
- The Norwegian Research Council (grant number [301317](https://prosjektbanken.forskningsradet.no/project/FORISS/301317))
- UiT the Arctic University of Norway (internal funding)
