# batch_process

# Imports
import glob
import numpy as np
import os

def batch_inpaint_saturated_pixels(input_dir,output_dir):

    input_files = glob(input_dir)
