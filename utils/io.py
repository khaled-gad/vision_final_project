"""
from photo_wct.py of https://github.com/NVIDIA/FastPhotoStyle
Copyright (C) 2018 NVIDIA Corporation.
Licensed under the CC BY-NC-SA 4.0

This module provides input/output utilities for the style transfer process.
"""
import os
import datetime

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class Timer:
    """
    A context manager that measures execution time of code blocks.
    
    Attributes:
        msg (str): Message template to display with elapsed time.
        verbose (bool): Whether to print timing information.
        start_time: Timestamp when the timer starts.
    """
    def __init__(self, msg='Elapsed time: {}', verbose=True):
        self.msg = msg
        self.start_time = None
        self.verbose = verbose

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(self.msg.format(datetime.datetime.now() - self.start_time))


def open_image(image_path, image_size=None):
    """
    Opens an image file and prepares it for neural processing.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int, optional): Target size for resizing the image.
        
    Returns:
        torch.Tensor: Image tensor ready for neural network processing.
    """
    img_source = Image.open(image_path)
    transformation_list = []
    if image_size is not None:
        img_source = transforms.Resize(image_size)(img_source)
        # transformation_list.append(transforms.Resize(image_size))
    width, height = img_source.size
    transformation_list.append(transforms.CenterCrop((height // 16 * 16, width // 16 * 16)))
    transformation_list.append(transforms.ToTensor())
    transform_pipeline = transforms.Compose(transformation_list)
    return transform_pipeline(img_source).unsqueeze(0)


def change_seg(seg):
    """
    Converts a segmentation image with RGB colors to label indices.
    
    Args:
        seg (PIL.Image): Segmentation image with color-coded regions.
        
    Returns:
        numpy.ndarray: Segmentation map with numerical labels.
    """
    color_mapping = {
        (0, 0, 255): 3,    # blue
        (0, 255, 0): 2,    # green
        (0, 0, 0): 0,      # black
        (255, 255, 255): 1,  # white
        (255, 0, 0): 4,    # red
        (255, 255, 0): 5,  # yellow
        (128, 128, 128): 6,  # grey
        (0, 255, 255): 7,  # lightblue
        (255, 0, 255): 8   # purple
    }
    seg_array = np.asarray(seg)
    result_map = np.zeros(seg_array.shape[:-1])
    
    for row in range(seg_array.shape[0]):
        for col in range(seg_array.shape[1]):
            pixel_color = tuple(seg_array[row, col, :])
            if pixel_color in color_mapping:
                result_map[row, col] = color_mapping[pixel_color]
            else:
                closest_color_idx = 0
                min_distance = 99999
                for color_key in color_mapping:
                    color_distance = np.sum(np.abs(np.asarray(color_key) - seg_array[row, col, :]))
                    if color_distance < min_distance:
                        min_distance = color_distance
                        closest_color_idx = color_mapping[color_key]
                    elif color_distance == min_distance:
                        try:
                            closest_color_idx = result_map[row, col-1, :]
                        except Exception:
                            pass
                result_map[row, col] = closest_color_idx
    return result_map.astype(np.uint8)


def load_segment(image_path, image_size=None):
    """
    Loads a segmentation image and preprocesses it for style transfer.
    
    Args:
        image_path (str): Path to the segmentation image.
        image_size (int, optional): Target size for resizing the segmentation.
        
    Returns:
        numpy.ndarray: Processed segmentation map.
    """
    if not image_path:
        return np.asarray([])
    
    seg_img = Image.open(image_path)
    if image_size is not None:
        resize_transform = transforms.Resize(image_size, interpolation=Image.NEAREST)
        seg_img = resize_transform(seg_img)
    
    width, height = seg_img.size
    crop_transform = transforms.CenterCrop((height // 16 * 16, width // 16 * 16))
    seg_img = crop_transform(seg_img)
    
    if len(np.asarray(seg_img).shape) == 3:
        seg_img = change_seg(seg_img)
    
    return np.asarray(seg_img)


def compute_label_info(content_segment, style_segment):
    """
    Analyzes content and style segmentation maps to determine valid transfer labels.
    
    Args:
        content_segment (numpy.ndarray): Content image segmentation map.
        style_segment (numpy.ndarray): Style image segmentation map.
        
    Returns:
        tuple: (label_set, label_indicator) where label_set contains unique labels
              and label_indicator is a boolean array indicating valid transfer labels.
    """
    if not content_segment.size or not style_segment.size:
        return None, None
    
    max_label_value = np.max(content_segment) + 1
    unique_labels = np.unique(content_segment)
    valid_label_flags = np.zeros(max_label_value)
    
    for label in unique_labels:
        content_pixels = np.where(content_segment.reshape(content_segment.shape[0] * content_segment.shape[1]) == label)
        style_pixels = np.where(style_segment.reshape(style_segment.shape[0] * style_segment.shape[1]) == label)

        content_area = content_pixels[0].size
        style_area = style_pixels[0].size
        
        # Check if both areas are significant and their ratio is reasonable
        if (content_area > 10 and 
            style_area > 10 and 
            content_area / style_area < 100 and 
            style_area / content_area < 100):
            valid_label_flags[label] = True
        else:
            valid_label_flags[label] = False
            
    return unique_labels, valid_label_flags


def mkdir(directory_name):
    """
    Creates a directory if it doesn't exist.
    
    Args:
        directory_name (str): Path of directory to create.
        
    Raises:
        AssertionError: If a file with the same name already exists.
    """
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    else:
        assert os.path.isdir(directory_name), f'already exists filename {directory_name}'
