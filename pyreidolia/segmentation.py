import os
import gc
import cv2
import time
import tqdm
import random
import numpy as np
import pandas as pd
import os
import albumentations as albu
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from PIL import Image

from typing import Union, Callable
from pyreidolia.mask import make_mask

def seed_everything(seed):
    """Set the seed of components using random generators

    Parameters
    ----------
    seed : int
        the seed of the random number generator, for numpy and torch
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def to_tensor(x: Union[np.array, np.ndarray],
              **kwargs):
    """Convert image or mask to tensor format

    Parameters
    ----------
    x : Union[np.array, np.ndarray]
        image or mask to convert

    Returns
    -------
    Union[np.array, np.ndarray]
        to tensor format
    """
    return x.transpose(2, 0, 1).astype("float32")

# Dataset class
# dataset code to be decoupled from our model training code for better readability and modularity
# Pytorch provides two primitive Dataset and Dataloader
# Dataset stores the samples and their corresponding labels, 
# and DataLoader wraps an iterable around the Dataset to enable easy access to the samples
# see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CloudDataset(Dataset):
    """Image augmentation using albumentations, horizontal flip for data augmentation

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with the image name and masks, by default None
    datatype : str
        if train or test data, by default "train"
    img_ids : np.array
        array of image ids to be used, by default None
    transforms : _type_
        image transform function for image augmentation, by default albu.Compose([albu.HorizontalFlip()])
    subfolder : str
        where to store the images, by default "train_images_525/"
    """
    def __init__(
        self,
        df: pd.DataFrame = None,
        datatype: str = "train",
        img_ids: np.array = None,
        transforms=albu.Compose([albu.HorizontalFlip()]), #, AT.ToTensor()
        img_dir = None,
        subfolder = "train_images_525/",
        mask_subfolder = "train_masks_525/"
    ):
        self.df = df
        self.data_folder = f"{img_dir}{subfolder}"
        self.mask_folder = f'{img_dir}{mask_subfolder}'
        self.img_ids = img_ids
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        # make mask
        mask = make_mask(df=self.df, image_name=image_name, mask_path=self.mask_folder)
        # load the image in RGB
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Blur the image
        img_blurred = cv2.blur(img, ksize=(51, 51))
        # Take the difference with the original image
        # Weight with a factor of 4x to increase contrast
        img = cv2.addWeighted(img, 4, img_blurred, -4, 0)
        # augmentation
        augmented = self.transforms(image=img, mask=mask)
        img = np.transpose(augmented["image"], [2, 0, 1])
        mask = np.transpose(augmented["mask"], [2, 0, 1])
        return img, mask

    def __len__(self):
        return len(self.img_ids)
    

def get_training_augmentation():
    """retrieve the augmented training images

    Returns
    -------
    object
        the albumentation transform
    """
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(
            scale_limit=0.5,
            rotate_limit=0,
            shift_limit=0.1,
            p=0.5,
            border_mode=0
        ),
        albu.GridDistortion(p=0.5),
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(320, 640),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn: Callable):
    """Construct preprocessing transform

    Parameters
    ----------
    preprocessing_fn : 
        data normalization function
            (can be specific for each pretrained neural network)

    Returns
    -------
    albumentations.Compose
        albumentations compose method
    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def dice(img1: np.array, img2: np.array):
    """Compute the dice metric between two images. The images are thresholded
    (binary pixel values)

    Parameters
    ----------
    img1 : 
        reference image
    img2 : 
        predicted image

    Returns
    -------
    float
        the dice metric value
    """
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2.0 * intersection.sum() / (img1.sum() + img2.sum())

def dice_no_threshold(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
):
    """Non threshold dice metric

    Parameters
    ----------
    outputs : 
        the predicted torch tensor (image)
    targets : 
        the target tensor (image)
    eps : 
        tolerance, by default 1e-7
    threshold : 
        threshold, if any, for binarizing the image, by default None

    Returns
    -------
    float
        dice metric value
    
    Reference
    ---------
    https://catalyst-team.github.io/catalyst/_modules/catalyst/dl/utils/criterion/dice.html
    """
    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    dice = 2 * intersection / (union + eps)

    return dice

def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()


    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., 
                           eps=self.eps, threshold=None, 
                           activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid', lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if self.activation is None:
            self.bce = nn.BCELoss(reduction='mean')
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)
    
