# preprocess

# Imports
import json
import skimage.io
import numpy as np
import json
import copy

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


def save_class_mask(mask,image_file):
    """ Save mask with class indices to file using skimage.io.imsave

    # Usage:
    save_class_mask(mask,image_file)

    # Required arguments:
    mask:           2D numpy array with class indices
    image_file:     Path to image file where mask will be saved (typically PNG)

    # Notes:
    The mask is converted to an 8-bit integer (range 0-255) before saving.

    """

    mask_int = mask.astype(np.uint8)    # Convert to 8-bit int
    skimage.io.imsave(image_file, mask_int, check_contrast=False)


def save_class_dict(dict,json_file):
    """ Save simple class dictinary to JSON file

    # Usage:
    save_class_dict(dict,json_file)

    # Required arguments:
    dict:       Dictionary with class names as keys, and class indices (used in
                class masks) as values.
                example_dict = {"Water":1, "Land":2}
    json_path   Path to JSON file where dictionary will be saved

    # Notes:
    The JSON file is written using UTF-8 encoding, sets the ensure_ascii option
    to False to allow a larger range of symbols in class names, and uses the
    indent=4 option make the file easier to read (by humans).
    """

    with open(json_file, "w", encoding='utf-8') as write_file:
        json.dump(dict, write_file, ensure_ascii=False, indent=4)


def read_class_dict(json_file):
    """ Load class dictionary saved in JSON format

    # Usage:
    class_dict = read_class_dict(json_file)

    # Required arguments:
    json_file:      Path to JSON file where the class dictionary has been saved

    # Returns:
    class_dict:     Simple dictionary with class names as keys and class indices
                    as values. example_dict = {"Water":1, "Land":2}
    """

    # Read serialized version of dictionary (text string)
    with open(json_file, 'r') as read_file:
        json_serialized_data = read_file.read()

    # De-serialize data, converting it into a dictionary again
    return json.loads(json_serialized_data)


def extract_subset(class_dict,class_mask,classes_to_extract, reset_class_ind = True):
    """ Extract a subset from a set of annotated classes

    # Usage
    (subset_class_dict, subset_class_mask) =
        extract_subset(class_dict,class_mask,classes_to_extract)

    # Required arguments
    class_dict:             Dict, keys = class names, values = class indices
    class_mask:             Image, map of class indices (integers)
    classes_to_extract:     List, names of classes to extract

    # Optional arguments:
    reset_class_ind         Create new class indices, starting at 1 and ending
                            at len(classes_to_extract). If False, original
                            class indices are used.

    # Returns
    subset_class_dict       Dict, keys = subset class names,
                            values = updated class indices
    subset_class_mask       Image, map of class indices for extracted classes
                            Background (no annotation) pixels are set to zero.

    """

    # Create empty class dict and mask
    subset_class_dict = {}
    subset_class_mask = np.zeros(class_mask.shape, dtype=class_mask.dtype)

    # Loop over subset of classes and create new dict and mask
    for ii, class_name in enumerate(classes_to_extract):

        # Determine index for current class
        if reset_class_ind:
            class_ind = ii + 1      # Add 1 to start numbering at 1
        else:
            class_ind = class_dict[class_name]

        # Insert class and index in dict
        subset_class_dict[class_name] = class_ind

        # Add mask correspondng to current class
        subset_class_mask[class_mask == class_dict[class_name]] = class_ind

    return subset_class_dict, subset_class_mask


def merge_classes_with_mask(class_dict,class_mask,classes_to_merge,merged_class_names):
    """ Merge subsets of a set of annotated classes

    # Usage
    (merged_class_dict, merged_class_mask) =
        merge_classes(class_dict,class_mask,classes_to_merge,merged_class_names)

    # Required arguments
    class_dict:             Dict, keys = class names, values = class indices
    class_mask:             Image, map of class indices (integers)
    classes_to_merge:       List of lists, each sub-list containing classes to
                            be merged into a single new class
    merged_class_names:     List with names for each new merged class


    # Returns
    merged_class_dict       Dict, keys = subset class names,
                            values = updated class indices
    merged_class_mask       Image, map of class indices for merged classes.

    """

    assert len(classes_to_merge) == len(merged_class_names)

    # Copy original dict and mask
    merged_class_dict = {}
    merged_class_mask = np.zeros(class_mask.shape, dtype=class_mask.dtype)
    class_dict_copy = copy.deepcopy(class_dict)

    for ii,class_set in enumerate(classes_to_merge):
        # Set index for merged class
        merged_class_ind = ii+1

        # Insert index for merged class in dict
        merged_class_dict[merged_class_names[ii]] = merged_class_ind

        for class_name in class_set:
            if class_name in class_dict:
                # Remove class from copy of original dict
                class_dict_copy.pop(class_name)

                # Update mask
                merged_class_mask[class_mask == class_dict[class_name]] = merged_class_ind
            else:
                raise ValueError('Class name \"' + class_name + '\" in classes_to_merge ' +
                                'does not match any names in class_dict')

    # Loop through remaining classes and insert them
    merged_class_ind += 1
    for class_name in class_dict_copy:
        merged_class_dict[class_name] = merged_class_ind
        merged_class_mask[class_mask == class_dict[class_name]] = merged_class_ind

        merged_class_ind += 1

    return merged_class_dict, merged_class_mask


def merge_classes_in_label_vector(class_dict,y,classes_to_merge,merged_class_names):
    """ Merge subsets of a set of annotated classes

    # Usage
    (class_dict_merged, y_merged) = merge_classes_in_label_vector(
        class_dict,y,classes_to_merge,merged_class_names)

    # Required arguments
    class_dict:             Dict, keys = class names, values = class indices
    y:                      Label vector (integers)
    classes_to_merge:       List of lists, each sub-list containing classes to
                            be merged into a single new class
    merged_class_names:     List with names for each new merged class


    # Returns
    class_dict_merged       Dict, keys = merged class names,
                            values = updated class indices
    y_merged                Image, map of class indices for extracted classes

    """

    assert len(classes_to_merge) == len(merged_class_names)

    # Copy original dict and mask
    class_dict_merged = {}
    class_dict_copy = copy.deepcopy(class_dict)
    y_merged = np.zeros(y.shape)

    for ii,class_set in enumerate(classes_to_merge):
        # Set index for merged class (starts at 1, reserve 0 for background)
        merged_class_ind = ii+1

        # Insert index for merged class in dict
        class_dict_merged[merged_class_names[ii]] = merged_class_ind

        for class_name in class_set:
            if class_name in class_dict:
                # Remove class from copy of original dict
                class_dict_copy.pop(class_name)

                # Update y vector
                y_merged[y == class_dict[class_name]] = merged_class_ind

            else:
                raise ValueError('Class name \"' + class_name + '\" in classes_to_merge ' +
                                'does not match any names in class_dict')

    # Loop through remaining classes and insert them "at the end"
    merged_class_ind += 1
    for class_name in class_dict_copy:
        class_dict_merged[class_name] = merged_class_ind
        y_merged[y == class_dict[class_name]] = merged_class_ind

        merged_class_ind += 1

    # Return
    return class_dict_merged, y_merged


def annotation_data_to_matrix(data,class_dict):
    """ Convert data structure with spectra from annotation into X,y pair

    # Usage:
    (X,y) = annotation_data_to_matrix(data,class_dict)

    # Required arguments:
    data:           Data structure from batch_process.collect_annotated_data()
    class_dict:     Dict with class names as keys and class indices as values

    # Returns:
    X:      Matrix with samples along 1. dim and wavelengths along 2. dim
    y       Vector with class indices (length equal to first dim. of X)

    """
    # Find total number of spectra in data
    n_samp = 0
    for element in data:
        for class_name in element['spectra'].keys():
            n_samp += element['spectra'][class_name].shape[0]

    # Find number of spectral channels, using info from last file
    n_spec = element['spectra'][class_name].shape[1]

    # Preallocate X and y
    X = np.zeros((n_samp,n_spec))
    y = np.zeros(n_samp)

    # Loop over all images and insert spectra into X and y
    ind = 0
    for element in data:
        for class_name in element['spectra'].keys():
            n_samp_loc = element['spectra'][class_name].shape[0]

            X[ind:(ind + n_samp_loc),:] = element['spectra'][class_name]
            y[ind:(ind + n_samp_loc)] = class_dict[class_name]

            ind = ind + n_samp_loc

    return X,y
