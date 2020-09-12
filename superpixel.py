"""The superpixel generator for the support image format

The fast way to generate superpixel with target images. It default run 
multiprocessing for the single node. Also, it supports distributed multiprocessing 
for multi nodes.

Author: Weikun Han <weikunhan@gmail.com>

Please install: 
- pip install scikit-image
- pip install tqdm
"""

import argparse
import json
import os
import random
import skimage
import skimage.color as color
import skimage.filters as filters
import skimage.segmentation as segmentation
import numpy as np
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

CONFIG = json.loads(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                                      'config_superpixel.json')).read())
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class FelzenszwalbMethod:
    
    # TODO: you could add more config setting by changing this function
    def convert(self, image, scale=100, sigma=0.5, min_size=50):

        return segmentation.felzenszwalb(image, scale=scale, 
                                         sigma=sigma, min_size=min_size)


class SlicMethod:
    
    # TODO: you could add more config setting by changing this function
    def convert(self, image, n_segments=250, compactness=10, sigma=1):

        return segmentation.slic(image, n_segments=n_segments,
                                 compactness=compactness, sigma=sigma)


class QuickshiftMethod:
    
    # TODO: you could add more config setting by changing this function
    def convert(self, image, kernel_size=3, max_dist=6, ratio=0.5):
        
        return segmentation.quickshift(image, kernel_size=kernel_size, 
                                       max_dist=max_dist, ratio=ratio)


class WatershedMethod:
    
    # TODO: you could add more config setting by changing this function
    def convert(self, image, markers=250, compactness=0.001):
        gradient = filters.sobel(color.rgb2gray(image))

        return segmentation.watershed(gradient, markers=markers, 
                                      compactness=compactness) 


def get_superpixel(method, **kwargs):
    if method == 'watershed':

        return WatershedMethod()
    elif method == 'quick':

        return QuickshiftMethod()
    elif method == 'fz':

        return FelzenszwalbMethod()
    elif method == 'slic':

        return SlicMethod()
    else:
        
        raise ValueError

def has_file_allowed_extension(filename, extensions=IMG_EXTENSIONS):
    """ Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def save_superpixel(image_path_in, image_path_out, superpixel_method, **kwargs):
    """ Generate the superpixel ans save as .png
    
    Args:
        image_path_in (string): input path to a image 
        image_path_out (string): output path to a image
        superpixel_method (string): watershed, quick, fz, slic
        kwargs (dict): the config setting for target method

    Returns:
        Nothing
    """
    if os.path.isfile(image_path_out):
        try:
            Image.open(image_path_out).verify()

            return 
        except Exception as e:
            pass

    image = skimage.util.img_as_float(Image.open(image_path_in))
    superpixel = get_superpixel(superpixel_method)
    segments = superpixel.convert(image, **kwargs)
    segments = Image.fromarray(segments.astype(np.uint32))
    segments.save(image_path_out)

def generate_superpixels(image_dir_in, image_dir_out, superpixel_method, 
                         multiprocessing_distributed, **kwargs):
    """Generate superpixels for input image.

    Args:
        image_dir_in (string): input directory to a image 
        image_dir_out (string): output directory to a image
        superpixel_method (string): watershed, quick, fz, slic
        multiprocessing_distributed (bool): for distributed computing
        kwargs (dict): the config setting for target method

    Returns:
        Nothing
    """
    classes_list = [x.name for x in os.scandir(image_dir_in) if x.is_dir()]

    if multiprocessing_distributed:
        random.shuffle(classes_list)
    
    pool = Pool() 
    
    for classes in tqdm(classes_list, total= len(classes_list), 
                        desc='Start generating superpixels'):
        temp_dir_out = os.path.join(image_dir_out, classes)
        temp_dir_in = os.path.join(image_dir_in, classes)

        if not os.path.exists(temp_dir_out):
            os.mkdir(temp_dir_out)

        images_list = [x.name for x in os.scandir(temp_dir_in) if x.is_file()]
        
        for image in images_list:
            if not has_file_allowed_extension(image):
                continue

            image_path_in = os.path.join(temp_dir_in, image)
            image_path_out = os.path.join(temp_dir_out, '{}.png'.format(image.split('.')[0]))
            pool.apply_async(save_superpixel, 
                             (image_path_in, image_path_out, superpixel_method), 
                             (kwargs))

    pool.close()
    pool.join()

def main():
    config_dict = {}
    
    print('Start generate superpixels use method "{}"'.format(args.method))

    if args.config:
        config_dict = CONFIG[args.method]
        print('the config info: {}'.format(CONFIG[args.method]))

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    generate_superpixels(args.input, args.output, args.method, 
                         args.multiprocessing_distributed, **config_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Superpixels generator')
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='please input the path of image folder '
                             'e.g. "image_folder/train"')
    parser.add_argument('-o', '--output', 
                        type=str,
                        required=True,
                        help='please output the path of image folder '
                             'e.g. "image_folder/train_sp"')
    parser.add_argument('-m', '--method', 
                        type=str,
                        required=True,
                        help='please select superpixel method: '
                             '"watershed", "quick", "fz", "slic"')
    parser.add_argument('--config', action='store_true',
                        help='use config in file ./config_superpixel.json '
                             'to generate superpixel based on select method')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='use multi-processing distributed to generate '
                             'superpixel N processes per node, which has N CPUs')
    args = parser.parse_args()

    main()