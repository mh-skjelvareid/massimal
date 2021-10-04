
import glob
import os

def recursive_file_search(root_dir, file_pattern):
    """ Find all files matching pattern inside a "root" directory

    # Usage:
    file_list = recursive_file_search(root_dir, file_pattern)

    # Required arguments:
    root_dir:       Path to root dir
    file_pattern:   Search pattern for file.
                    Example: '*.png' (finds all png files)
                    See glob.glob() for syntax details

    # Returns:
    file_list:  List with full path to all files found.

    """
    glob_str = root_dir + os.path.sep + '**' + os.path.sep + file_pattern
    file_list = glob.glob(glob_str,recursive=True)
    file_list.sort()

    return file_list

def new_file_path_from_existing(full_file_path,new_dir,new_ext = None):
    # See https://note.nkmk.me/en/python-os-basename-dirname-split-splitext/
