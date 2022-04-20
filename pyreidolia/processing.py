import numpy as np
import cv2
from typing import Union, Callable, Optional, Any, List, Tuple


def resize(img: Union[np.ndarray, np.array], 
           img_size: Tuple[int, int] = (350, 525)):
    """resize an image, if user supplied size is different from original size

    Parameters
    ----------
    img : np.array
        original image to resize
    img_size : tuple, optional
        target image size, by default ((350, 525)

    Returns
    -------
    np.array
        resized image
    """
    if img.shape != img_size:
        return cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
    else:
        return img


def threshold(img: Union[np.array, np.ndarray]):
    """Threshold an image, returns a black and white image based on the threshold

    Parameters
    ----------
    img : Union[np.array, np.ndarray]
        image to be thresholded

    Returns
    -------
    Union[np.array, np.ndarray]
        thresholded image
    """
    _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    return img

def erode(img: Union[np.array, np.ndarray], 
          gb_ksize: Tuple[int, int] = (3, 3), 
          erosion_ksize: Tuple[int, int] = (5, 5)):
    """Erode an image using GaussianBlur

    Parameters
    ----------
    img : Union[np.array, np.ndarray]
        the image to be eroded
    gb_ksize : Tuple[int, int], optional
        the GaussianBlur kernel size, by default (3, 3)
    erosion_ksize : Tuple[int, int], optional
        the erosion kernel size, by default (5, 5)

    Returns
    -------
    Union[np.array, np.ndarray]
        The eroded image
    """
    # same kernel size than for the erosion
    gb_filter = cv2.GaussianBlur(img, ksize=gb_ksize, sigmaX=0, sigmaY=0)
    kernel = np.ones(erosion_ksize, np.uint8)
    return cv2.erode(gb_filter, kernel)

def dilate(img: Union[np.array, np.ndarray],
           ksize: Tuple[int, int] = (7, 7)):
    """dilate an image

    Parameters
    ----------
    img : Union[np.array, np.ndarray]
        image to dilate
    ksize : Tuple[int, int], optional
        the dilatation kernel size, by default (7, 7)

    Returns
    -------
    Union[np.array, np.ndarray]
        dilated image
    """
    kernel = np.ones(ksize, np.uint8)
    return cv2.dilate(img, kernel)

def remove_bckground(img: Union[np.array, np.ndarray]):
    """Remove background from the image

    Parameters
    ----------
    img : Union[np.array, np.ndarray]
        image with background

    Returns
    -------
    Union[np.array, np.ndarray]
        image without background
    """
    backSub = cv2.createBackgroundSubtractorMOG2() #cv2.createBackgroundSubtractorKNN()
    return backSub.apply(img)

def draw_convex_hull(mask: Union[np.array, np.ndarray], 
                     mode: str = 'convex'):
    """Draw a convex hull or a rectangular hull

    Parameters
    ----------
    mask : Union[np.array, np.ndarray]
        the convex hull or a rectangular hull mask
    mode : str, optional
        rectangular of convex hull, by default 'convex'

    Returns
    -------
    Union[np.array, np.ndarray]
        image with the convex/rectangular hull
    """
    img = np.zeros(mask.shape)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        if mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255.

def post_process(probability: np.array,
                 threshold: Union[float, int], 
                 min_size: Union[float, int]):
    """Threshold the probability. Take the probability of a pixel
    of being of certain type and threshold it, draw the convex hull
    of the area for comparing to the original mask.

    Parameters
    ----------
    probability : 
        predicted probability of being of certain type
    threshold : 
        the threshold, if probability > threshold --> 1
    min_size : 
        the minimum size of the area

    Returns
    -------
    np.array, int
        the mask of each component and the number of components in the image
    """
    # threshold the probability
    mask = (cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1])
    # draw the convex hull
    mask = draw_convex_hull(mask.astype(np.uint8))
    # compute the connected components labeled image of boolean image
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    # attach the label to the predicted area
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num