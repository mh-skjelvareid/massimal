# Massimal
 A collection of code related to the Massimal research project on UAV hyperspectral imaging of underwater vegetation.

## Installation
This code is still in development and has not  been organized and prepared for widespread distribution. However, if you do want to run parts of this code, you can find requirements in the [requirements_python.txt](requirements_python.txt) and [requirements_other.txt](requirements_other.txt) files. 

The requirements are quite loosely defined, with few limits on version numbers, as no obvious version conflicts have been detected yet. If you come across problems regarding package versions, please submit an [issue](https://github.com/mh-skjelvareid/massimal/issues) describing it. 

We recommend creating a virtual environment to install the required packages. If using conda, this can be done with e.g. 

    conda create --name massimal pip
    conda activate massimal

If you prefer a specific Python version, specify this when creating the environment, e.g. 

    conda create --name massimal python==3.11.7 pip
    conda activate massimal

Note that pip is installed when the environment is created. Some of the python packages are available through conda, but not all. We have found that it's generally easiest to install all python packages with pip:

    pip install -r requirements_python.txt

The requirements listed in [requirements_other.txt](requirements_other.txt) must be installed manually.

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
