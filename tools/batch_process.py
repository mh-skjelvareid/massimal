# batch_process

# Imports
import skimage, skimage.io, skimage.morphology
import numpy as np
import spectral
import misc, hyspec_io, preprocess, image_render

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


def hedley_remove_glint(input_dir,output_dir,train_cube_path,recursive_src = False):
    """ Apply Hedley glint removal to image batch

    # Required arguments:
    input_dir:      Directory contating input ENVI files (*.hdr).
                    All files will be included in the batch.
    output_dir:     Directory for saving output files.
    train_cube_path: Path to ENVI file with data used for training the glint
                    model (homogenous background with variations in glint).

    # Optional arguments:
    recursive_src:  Whether to search the input_dir for .hdr files recursively

    """

    # Load training data
    print('Loading training data ' + train_cube_path)
    (train,wl,rgb_ind,metadata) = hyspec_io.load_envi_image(train_cube_path)

    # Reshape to 2D
    train = train.reshape((-1,train.shape[2]))

    # Create and train sun glint model
    print('Training Hedley sun glint correction model')
    sgm = preprocess.HedleySunGlint()
    sgm.fit(train,wl)

    # Find input files
    file_list = misc.file_pattern_search(input_dir, '*.hdr', recursive = recursive_src)
    print('Found ' + str(len(file_list)) + ' images in input folder.')

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


def absolute_stretch(input_dir,output_dir,limits,file_ext='png',recursive_src = False):
    """ Perform absolute stretch on image batch

    # Required arguments:
    input_dir:      Directory contating input image files.
                    All files will be included in the batch.
    output_dir:     Directory for saving output image files.
    limits:         Limits for stretch, see misc.absolute_stretch

    # Optional arguments:
    file_ext:      File extension (file type) to look for in input_dir
    recursive_src:  Whether to search the input_dir recursively

    """

    # Find input files
    input_files = misc.file_pattern_search(input_dir, '*.' + file_ext, recursive = recursive_src)
    print('Found ' + str(len(input_files)) + ' images in input folder.')

    # Make output file paths
    output_files = misc.build_newdir_filepath(input_files,output_dir)

    for input_file, output_file in zip(input_files, output_files):
        # Load data
        print('Loading input file ' + input_file)
        input_image = skimage.io.imread(input_file)

        # Stretch file
        output_image = image_render.absolute_stretch(input_image,limits)

        # Save file
        print('Saving stretched image file as ' + output_file)
        skimage.io.imsave(output_file,output_image.astype(input_image.dtype))


def envi_rgb_render(input_dir,output_dir=None,limits=None,recursive_src=False,inpaint=True,closing_rad=3):
    """ Create RGB renderings of hyperspectral files and save as PNG


    # Required arguments:
    input_dir:      Directory contating input image files.
                    All files will be included in the batch.

    # Optional arguments:
    output_dir:     Directory for saving output image files. If None, files are
                    saved in input directory.
    limits:         Limits for stretch, see misc.absolute_stretch
                    If not specified, each image will be stretched to its
                    (2,92) percentiles, separatey for each R,G,B band
    recursive_src:  Whether to search the input_dir recursively
    inpaint:        Interpolate (inpaint) for pixels inside image
    closing_rad:    Radius used for morphology "closing" used to determine which
                    pixels will be inpainted. The larger the value, the more
                    pixels are included.

    """

    # Find input files
    input_files = misc.file_pattern_search(input_dir, '*.hdr', recursive = recursive_src)
    print('Found ' + str(len(input_files)) + ' images in input folder.')

    # Make output file paths
    if output_dir is None:
        output_dir = input_dir

    output_files = misc.build_newdir_filepath(input_files,output_dir,new_ext='.png')

    # Iterate over batch
    for input_file, output_file in zip(input_files,output_files):
        # Load data
        print('Loading input file ' + input_file)
        (im,wl,rgb_ind,metadata) = hyspec_io.load_envi_image(input_file)

        # Extrct RGB
        im_rgb = im[:,:,rgb_ind]

        # Stretch
        if limits is None:
            im_rgb_sc = image_render.percentile_stretch(im_rgb)
        else:
            im_rgb_sc = image_render.absolute_stretch(im_rgb,limits)

        # Convert to 8-bit int
        im_rgb_sc = np.uint8(im_rgb_sc*255)

        if inpaint:
            zero_mask = ~np.all(im_rgb_sc == 0,axis=2)
            mask_closed = skimage.morphology.binary_closing(
                            zero_mask,skimage.morphology.disk(closing_rad))
            mask_diff = (zero_mask != mask_closed)
            im_rgb_sc = preprocess.inpaint_masked(im_rgb_sc,mask_diff)

        # Save
        print('Saving RGB render of hyperspectral file as ' + output_file)
        skimage.io.imsave(output_file,im_rgb_sc)


# def save_envi_rgb_image(input_dir,output_dir):
#     """
#
#     """
#
#     # Find input files
#     file_list = misc.file_pattern_search(input_dir, '*.hdr', recursive = recursive_src)
#     print('Found ' + str(len(file_list)) + ' images in input folder.')
#
#     # Loop over all input files
#     for input_file in file_list:
#         # Load data
#         print('Loading input file ' + input_file)
#         (im,wl,rgb_ind,metadata) = hyspec_io.load_envi_image(input_file)
#
#         # Extract default RGB bands
#         im_rgb = im[:,:,rgb_ind]
#
#         # Update metadata
#         metadata['wavelength'] = [metadata['wavelength'][ii] for
#                                     ii in range(len(wl)) if ii in rgb_ind]
#
#         # Create path for output file
#         # NOTE: Should ideally add a short "RGB" tag to the file name
#         output_file = misc.build_newdir_filepath([input_file],output_dir)[0]
#         print('Saving RGB version of file as ' + output_file)
#
#         # Save file
#         hyspec_io.save_envi_image(output_file,im_rgb,metadata)
