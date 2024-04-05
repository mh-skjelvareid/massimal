# Massimal datasets task list
Tasks are organized according to the same structure as the datasets (Location -> Sensor -> Date), but a "general" task list is also included. 

## General
- [x] Make a function for converting .lcf (and .times?) files into "world" files (.wld) for easy visualization of Massimal images without full georeferencing.
    - [x] Read .lcf and .times files as NumPy arrays (with [numpy.loadtxt](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html#numpy.loadtxt)). See also video_transect.track_csv_to_geodataframe().
    - [x] Calculate average altitude
    - [x] Calculate start and end positions, middle and corners. Linear regression?
    - [x] Calculate world file transform according to specifications from [Wikipedia](https://en.wikipedia.org/wiki/World_file) and/or [ArcGIS](https://pro.arcgis.com/en/pro-app/3.1/help/data/imagery/world-files-for-raster-datasets.htm). Note: Not able to specify CRS in sidecar file that is automatically read. However, image is by default projected using QGIS project CRS (set to UTM). 
- [ ] Make a function for reading Garmin CSV files, interpolating to a fixed distance / time step (preferrably distance) and writing the result as a CSV file compatible with video transect image extraction. 
    - [ ] Read Garmin CSV file
    - [ ] Check time zone GPS vs video
    - [ ] Interpolate CSV to denser set of points (approx 1 per meter). See [Pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) or [SciPy](https://docs.scipy.org/doc/scipy/reference/interpolate.html) interpolation methods. See also [gpx_interpolate](https://github.com/remisalmon/gpx-interpolate/blob/master/gpx_interpolate.py) using pchip_interpolate(). Note that y_input can be a 2D array, not only a vector.
- [x] Use PCA for localized sun glint correction
    - [x] Write function for applying learned PCA for sun glint correction of image
    - [x] Write function for collecting (random) spectra from set of images and fitting PCA model to these
- [x] Write code to export RGB images as PNG with transparency layer (useful for display in QGIS)
    


## Smøla
- [ ] Georeference GoPro images from Otter
    - [x] Get Otter .csv file
    - [x] Write function for searching for closest Otter position based on image timestamp
    - [x] Handle images distributed across 2 cameras and multiple folders per camera. Rename image CameraNumber\_ImageFilename (also referenced in GeoPackage)?
    - [x] Use location filtering function to decrease number of images (only need 1 image per meter)
    - [ ] Batch process all images - copy to other folder - save with lower quality(?) (see [skimage.io.imsave](https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imsave), use "quality" keyword, 1 to 100)
    - [ ] Batch color correct used images. 
- [x] Georeference GoPro video from boat
- [x] Sun glint correct?


## Larvik
- [ ] Georeference GoPro video from boat - maybe not needed
- [x] Sun glint correct?
 
