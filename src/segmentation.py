from skimage.draw import random_shapes
from skimage.measure import label    
import numpy as np

def segmentation_data_generator(n_images,
                                data_shape,
                                class_mean,
                                class_std = 0,
                                background_mean = 0,
                                background_std = 0,
                                min_size = 5,
                                max_size = None,
                                shape = 'ellipse'):
    """ Generate synthetic segmentation image data 
    
    # Input parameters:
    n_images:     Number of images to generate
    data_shape:   (n_rows, n_cols, n_channels)
    class_mean:   (class1_mean, class2_mean, ...)
                  Nested tuple wth mean value(s) for each class.
                  Each element contains mean values for all channels
    class_std:    (class1_mean, class2_mean, ...)
                  Nested tuple wth standard dev. value for each class.
                  Each element contains std values for all channels
                  If None, class_std is set to zero for each class
                  If non-zero, values for each class follow a Gaussian
                  distribution.
    min_size:     Minimum size (pixels) of each object
    max_size:     Maximum size of each object
    shape:        {rectangle, circle, triangle, ellipse, None} str, optional
                  Shape to generate, 'None' to pick random ones
    
    
    # Returns:
    image:        Image with dimensions (n_rows, n_cols, n_channels)
                  dtype = float64
    label         Image with dimensions (n_rows, n_cols)
                  dtype = int
                  Classes are labelled 1, 2, ... according to their
                  order in class_mean / class_std
                  Index 0 corresponds to "background" (no class)
                  
    # Notes:
    - Only generates a single object per class
    - This function relies heavily on the random_shapes function 
    from SciKit Image: https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.random_shapes
    
    """
    
    n_rows, n_cols, n_channels = data_shape
    n_classes = len(class_mean)
    
    image_count = 0
    while image_count < n_images:
        # Make random shapes and label them, retrying if random_shapes() generates too few
        n_labels = None
        while n_labels != n_classes:
            # Generate random image with random intensities
            random_im, _ = random_shapes((n_rows,n_cols),
                                max_shapes=n_classes,
                                min_shapes=n_classes,
                                min_size=min_size,
                                max_size=max_size,
                                intensity_range=(0,127),
                                shape = shape,
                                num_trials=200,
                                channel_axis=None)
            # Label the random shapes (0=background, 1=first class, etc.)
            labels, n_labels = label(random_im,background=255,return_num=True)


        noise_generator = np.random.default_rng()
        

        im = np.zeros(data_shape)
        
        if background_mean != 0 or background_std != 0:
            background_mask = np.where(labels==0)
            if len(background_mean) == 1:
                background_mean = (background_mean for _ in range(data_shape[-1]))
            if len(background_std) == 1:
                background_std = (background_std for _ in range(data_shape[-1]))
                
            background_noise = noise_generator.multivariate_normal(
                mean=background_mean,cov=np.diag(background_std),size=background_mask[0].shape)
            im[background_mask] = background_noise
        
        for i in range(n_labels):
            class_mask = np.where(labels==(i+1))
            if len(class_std) == 1:
                class_std = (class_std for _ in range(data_shape[-1]))
            class_signal = noise_generator.multivariate_normal(
                mean=class_mean[i],cov=np.diag(class_std[i]),size=class_mask[0].shape)
            im[class_mask] = class_signal
            
        im[im<0] = 0
        im[im>1] = 1

        yield im, labels
        image_count += 1