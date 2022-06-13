import os
from typing import Union, Callable, Optional, Any, List, Tuple
import cv2


def get_img(
    img_name: str, main_folder: str, folder: str = "train_images_525/train_images_525"
):
    """Return image based on image name and folder.

    Parameters
    ----------
    img_name : str
        the image name, with extension
    main_folder : str
        the parent folder, with train and test images folder
    folder : str
        the image folder, by default "train_images_525/train_images_525"

    Returns
    -------
    Union[np.ndarray, np.array]
        image
    """
    data_folder = f"{main_folder}/{folder}"
    image_path = os.path.join(data_folder, img_name)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
