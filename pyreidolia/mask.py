import os
import cv2
from numba import njit, jit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Union, List, Tuple

def rle_to_mask(rle: str, 
                size: Tuple[int]  = (1400, 2100)):
    """decode rle to mask
    
    Parameters
    ----------
    rle : str
        the rle string
    size : Tuple[int], optional
        target mask size, by default (1400, 2100)

    Returns
    -------
    np.array
        mask
    """
    array = np.fromstring(rle, dtype=int, sep=' ')
    return array_to_mask(array=array, size=size) 

@jit(nopython=True)
def array_to_mask(array: np.array, size: Tuple[int]  = (1400, 2100)):
    """Converting an RLE encoding to a mask

    Parameters
    ----------
    rle : str
        the rle string
    size : Tuple[int], optional
        target mask size, by default (1400, 2100)

    Returns
    -------
    np.array
        mask
    """

    width, height = size[:2]
    mask = np.zeros(width*height).astype(np.uint8)
    # array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def mask_to_rle(image):
    """Encode a masked image to rle string

    Parameters
    ----------
    image : np.array
        the masked image

    Returns
    -------
    str
        the rle string
    """
    pixels = image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

def make_mask(df: pd.DataFrame,
              mask_path: str, 
              image_name: str, 
              shape: Tuple[int] = (350, 525)):
    """Create mask based on df, image name and shape.

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing the image names and patterns
    mask_path : str
        images folder path
    image_name : str
        image name
    shape : Tuple[int]
        mask size, by default (350, 525)

    Returns
    -------
    _type_
        _description_
    """
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
    df = df[df["im_id"] == image_name]
    for idx, im_name in enumerate(df["im_id"].values):
        for classidx, classid in enumerate(["Fish", "Flower", "Gravel", "Sugar"]):
            mask = cv2.imread(mask_path + classid + im_name)
            if mask is None:
                continue
            if mask[:, :, 0].shape != (350, 525):
                mask = cv2.resize(mask, (525, 350))
            masks[:, :, classidx] = mask[:, :, 0]
    masks = masks / 255
    return masks 

def bounding_box(img):
    # return max and min of a mask to draw bounding box
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def get_mask_cloud(img_path, img_id, label, mask):
    img = cv2.imread(os.path.join(img_path, img_id), 0)
    mask_decoded = rle_to_mask(mask, size=img.shape)
    mask_decoded = (mask_decoded > 0.0).astype(int)
    img = np.multiply(img, mask_decoded)
    return img

def get_binary_mask_sum(encoded_mask):
    """Count the number of pixels in a mask (surface)

    Args:
        encoded_mask (np.array): pattern mask

    Returns:
        int: number of pixels (surface)
    """
    mask_decoded = rle_to_mask(encoded_mask, size=(2100, 1400))
    binary_mask = (mask_decoded > 0.0).astype(int)
    return binary_mask.sum()

