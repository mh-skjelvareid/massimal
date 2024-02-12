# Massimal datasets task list
Tasks are organized according to the same structure as the datasets (Location -> Sensor -> Date), but a "general" task list is also included. 

## General
- [ ] Make a function for converting .lcf (and .times?) files into "world" files (.wld) for easy visualization of Massimal images without full georeferencing.
    - [ ] Read .lcf and .times files as NumPy arrays
    - [ ] Calculate average altitude
    - [ ] Calculate start and end positions, middle and corners. Linear regression?
    - [ ] Calculate world file transform according to specifications from [Wikipedia](https://en.wikipedia.org/wiki/World_file) and/or [ArcGIS](https://pro.arcgis.com/en/pro-app/3.1/help/data/imagery/world-files-for-raster-datasets.htm) 


## Smøla
- [ ] Georeference GoPro images from Otter
    - [x] Get Otter .csv file
    - [x] Write function for searching for closest Otter position based on image timestamp
    - [ ] Handle images distributed across 2 cameras and multiple folders per camera. Rename image CameraParent\_FolderParent\_ImageFilename (also referenced in GeoPackage)?
    - [ ] Use location filtering function to decrease number of images (only need 1 image per meter)
    - [ ] Batch process all images
- [ ] Georeference GoPro images from boat

## Larvik


