# The existence of this file indicates that the containing folder is a Python
# package. The file can also contain python code.
# See https://docs.python.org/3/reference/import.html#regular-packages
#
# The easiest way to make the package available outside its own folder
# is to install it with pip. Being inside the directory containing this file 
# (__init__.py), simply run
#   pip install .
# or, alternatively, for "editable mode",
#   pip install -e .
# 
# See also:
#   https://learn.scientific-python.org/development/tutorials/packaging/ 
#   https://packaging.python.org/tutorials/packaging-projects/ 
#
# To add the package to the Python search path without installing it, 
# you can add it to the Python path at runtime (a "hack"):
#   import sys
#   sys.path.append(/path/to/package)
#
# If working in Linux, the path to the package can also be added by modifying
# the .bashrc file in the home directory (cleaner, and only has to be done
# once). Open .bashrc (for example by typing entering <gedit ~/.bashrc> in the
# terminal) and add the following line, updating it to the actual path:
#   export PYTHONPATH="{$PYTHONPATH}:/path/to/module"
#
# See https://bic-berkeley.github.io/psych-214-fall-2016/sys_path.html for a
# nice explanation of how modules and paths are handled.
#

# import annotation
# import batch_process
# import crs_tools
# import georeferencing
# import hyspec_cnn
# import hyspec_io
# import hyspec_ml
# import image_render
# import misc
# import preprocess
# import segmentation
# import video_transect
