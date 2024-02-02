# Massimal - additional requirements

This is a list of requirements for the massimal repository that are not Python packages. The requirements must be installed manually (this file is not formatted for automatic installation using software tools like e.g. pip).

- GraphViz (called by tensorflow.keras.utils.plot_model()): https://www.graphviz.org/download/ 
- FFMPEG (called by ffmpeg-python in module video_transect): https://ffmpeg.org/download.html 
- Phil Harvey's ExifTool (called by module exiftool in module video_transect): https://exiftool.org/install.html 

On Ubuntu/Debian systems, it should be possible to install these using 

	sudo apt update
	sudo apt install graphviz ffmpeg libimage-exiftool-perl

Note that these tools are used for quite specific tasks, and may not be needed in your case.