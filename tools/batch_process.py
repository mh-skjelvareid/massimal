# batch_process

# Imports
import skimage, skimage.io
import numpy as np
import spectral
import misc, hyspec_io, preprocess

def detect_saturated(input_dir,recursive_src=False,**kwargs):
    """ Detect saturated pixels in image batch, save as PNG files

    # Required arguments:
    input_dir:      Directory contating input ENVI files (*.hdr).
                    All files will be included in the batch.

    # Optional arguments:
    recursive_src:  Whether to search the input_dir for .hdr files recursively
    **kwargs:       Keyword arguments are passed on to detect_saturated()
    """

    file_list = misc.file_pattern_search(input_dir, '*.hdr', recursive = recursive_src)

    for file in file_list:
        # Load image
        print('Loading ' + file)
        (im_cube,wl,rgb_ind,metadata) = hyspec_io.load_envi_image(file)

        # Detect saturated pixels in image
        mask = preprocess.detect_saturated(im_cube,**kwargs)

        # Save mask as PNG image with same base name and folder as image
        mask_file = file.split(sep='.')[0] + '_sat.png'
        print('Saving saturation mask as ' + mask_file)
        skimage.io.imsave(mask_file,skimage.img_as_ubyte(mask))


def hedley_remove_glint(input_dir,output_dir,train_cube_path,train_sat_path,recursive_src = False):
    """ Apply Hedley glint removal to image batch

    # Required arguments:
    input_dir:      Directory contating input ENVI files (*.hdr).
                    All files will be included in the batch.
    output_dir:     Directory for saving output files.
    train_cube_path: Path to ENVI file with data used for training the glint
                    model (homogenous background with variations in glint).
    train_sat_path: Path to PNG image with boolean mask, True indicating saturation

    # Optional arguments:
    recursive_src:  Whether to search the input_dir for .hdr files recursively

    """

    # Load training data
    print('Loading training data ' + train_cube_path)
    (train_im,wl,rgb_ind,metadata) = hyspec_io.load_envi_image(train_cube_path)

    # Load saturation mask
    print('Loading saturation mask ' + train_sat_path)
    sat_mask = skimage.io.imread(train_sat_path)
    train = train_im[sat_mask==0]

    # Create and train sun glint model
    print('Training Hedley sun glint correction model')
    sgm = preprocess.HedleySunGlint()
    sgm.fit(train,wl)

    # Find input files
    file_list = misc.file_pattern_search(input_dir, '*.hdr', recursive = recursive_src)
    print('Found ' + str(len(file_list)) + ' images, adding to batch.')

    # Loop over all input files
    for input_file in file_list:
        # Load data
        print('Loading input file ' + input_file)
        (im,wl,rgb_ind,metadata) = hyspec_io.load_envi_image(input_file)

        # Remove glint
        im_sgc = sgm.remove_glint(im)

        # Update metadata
        metadata['wavelength'] = [metadata['wavelength'][ii] for ii in range(len(wl)) if sgm.vis_ind[ii]]

        # Create path for output file
        output_file = misc.build_newdir_filepath([input_file],output_dir)[0]
        print('Saving sun glint corrected file as ' + output_file)

        # Save file
        hyspec_io.save_envi_image(output_file,im_sgc,metadata)

        # Copy .lcf and .times files if they exist
        misc.copy_hyspec_aux_files(input_file,output_dir)
