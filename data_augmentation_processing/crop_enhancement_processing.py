import os

import cv2
import numpy as np
import random
import scipy.ndimage as ndi

    # Crop the image randomly
    # translation_factor：transformation factor
    # Image np type

def do_random_crop(translation_factor, image_array,zoom_range):
    height = image_array.shape[0] #Obtain the height of the image
    width = image_array.shape[1]  #Obtain the width of the image
    x_offset = np.random.uniform(0, translation_factor * width)
    y_offset = np.random.uniform(0, translation_factor * height)
    offset = np.array([x_offset, y_offset])
    scale_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    crop_matrix = np.array([
        [scale_factor,0],
        [0,scale_factor]
    ])
    image_channel = [ndi.interpolation.affine_transform(image_channel, crop_matrix, offset=offset, order=0, mode = 'nearest',cval = 0.0)
                         for image_channel in image_array]

    image_array = np.stack(image_channel, axis=0)
    return image_array

def do_random_rotation(translation_factor, image_array,zoom_range):
    height = image_array.shape[0]
    width = image_array.shape[1]
    x_offset = np.random.uniform(0, translation_factor * width)
    y_offset = np.random.uniform(0, translation_factor * height)
    offset = np.array([x_offset, y_offset])
    scale_factor = np.random.uniform(zoom_range[0],
                                        zoom_range[1])
    crop_matrix = np.array([
        [scale_factor,0],[0,scale_factor]
    ])
    image_array = np.rollaxis(image_array,axis = -1,start = 0)
    image_channel = [ndi.interpolation.affine_transform(image_channel, crop_matrix, offset=offset, order=0, mode='nearest', cval=0.0)
                         for image_channel in image_array]
    image_array = np.stack(image_channel, axis = 0)
    image_array = np.rollaxis(image_array,0,3)
    return image_array

# img_path:Path of image
# save_path：Saved Path
# data_augmentation：Magnification factor

def generator_dataset(img_dir_path, save_path, data_augmentation,translation_factor):
    img_dirs = os.listdir(img_dir_path)
    zoom_range = [0.75, 1.25]
    for img_name in img_dirs:
        img = cv2.imread(img_dir_path + "/" + img_name)
        for i in range(0, data_augmentation):
            crop_img = do_random_crop(translation_factor,img,zoom_range)
            rotate_img = do_random_rotation(translation_factor,crop_img,zoom_range)
            cv2.imwrite(save_path + "/" + str(i) + "_" + img_name, rotate_img)


generator_dataset("./data/split","./data/generator",10,0.3)


