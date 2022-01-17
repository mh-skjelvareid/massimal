# preprocess

# Imports
import json

def read_hasty_metadata(hasty_json):
    """ Read class names and indices for Hasty.ai annotation

    # Usage:
    class_dict = read_hasty_metadata(hasty_json)

    # Required arguments:
    hasty_json:     JSON file, part of Hasty export

    # Returns:
    class_dict:     Dictionary with key = class name, value = png index

    # Notes
    The annotation is assumed to be exported from Hasty.ai in "semantic
    segmentation" format. The JSON file read by this function is bundled with a
    PNG file for every annotated image. The pixels belonging to a given class
    are indexed by using the png index in class_dict.

    """

    # Read raw text
    with open(hasty_json, 'r') as textfile:
        data = textfile.read()

    # Parse JSON text into list
    ann_class_info = json.loads(data)

    #%% Build a simple dictionary with class numbers and names
    class_dict = {}
    for element in ann_class_info:
        class_dict[element['name']] = element['png_index']

    return class_dict




# def annotation_vector_to_raster(rast_im_path,ann_vec_path,ann_rast_path):
#     """ Convert vector annotations to raster
#
#     # Usage:
#
#
#     # Required arguments:
#     rast_im_path:   Path to raster file for which the ann. mask will be made
#     ann_vec_path:   Path to file with annotation polygons (?)
#     ann_rast_path:  Path for output raster file with annotation masks
#
#     # Optional arguments:
#
#
#     # Returns:
#
#     """
#
#     return 0
