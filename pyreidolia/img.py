import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def pd_get_resolution_sharpness(df: pd.DataFrame, train_path: str, test_path: str, im_id_col: str = 'image_id') -> pd.DataFrame:
    """Get the image resolution and sharpness

    Parameters
    ----------
    df : pd.DataFrame
        the input data frame
    train_path : str
        path of the train data folder
    test_path : str
        path of the test data folder
    im_id_col : str
        the column name corresponding to the image id

    Returns
    -------
    df : pd.DataFrame
        the dataframe augmented with two columns, resolution and sharpness
    """
    n_img = len(df)
    res, sharp = np.empty((n_img,), dtype=object), np.empty((n_img,), dtype=np.float)
    progress_bar = tqdm(df[im_id_col])
    for i, im in enumerate(progress_bar):
        progress_bar.set_description('Processing {0:<30}'.format(im))
        if os.path.exists(train_path+im):
            img = cv2.imread(train_path+im, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(test_path+im, cv2.IMREAD_UNCHANGED)
        res[i] = img.shape
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        gnorm = np.sqrt(laplacian**2)
        sharp[i] = np.average(gnorm)
    df = df.assign(**{'resolution': res, 'sharpness': sharp})
    return df

def get_resolution_sharpness(im, train_path: str, test_path: str):
    """Get the image resolution and sharpness

    Parameters
    ----------
    df : pd.DataFrame
        the input data frame
    train_path : str
        path of the train data folder
    test_path : str
        path of the test data folder
        

    Returns
    -------
    pd.Series
        the series with two elements, resolution and sharpness
    """
    if os.path.exists(train_path+im):
        img = cv2.imread(train_path+im, cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(test_path+im, cv2.IMREAD_UNCHANGED)
    res = img.shape
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    gnorm = np.sqrt(laplacian**2)
    sharp = np.average(gnorm)
    return pd.Series([res, sharp])