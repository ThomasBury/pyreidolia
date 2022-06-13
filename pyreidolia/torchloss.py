import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation="sigmoid"):
    """Compute the Intersection over Union (f-score or Jaccard index)

    Parameters
    ----------
    pr : torch.Tensor
        A list of predicted elements
    gt : torch.Tensor
        A list of elements that are to be predicted
    beta : int, optional
        is chosen such that recall is considered beta  times as important as precision, is, by default 1
    eps : _type_, optional
        avoiding division by zero, by default 1e-7
    threshold : _type_, optional
        probability threshold for getting classes, by default None
    activation : str, optional
        activation function name, by default 'sigmoid'

    Returns
    -------
    float
        IoU (Jaccard) score

    Raises
    ------
    NotImplementedError
        wrong name for the activation function
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + eps
    )

    return score


class DiceLoss(nn.Module):
    """DiceLoss custom implementation of the Dice Loss function.
    includes the F_beta score.

    Parameters
    ----------
    nn : torch nn.Module
        parent torch class

    Returns
    -------
    float
        the dice coefficient
    """

    __name__ = "dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid"):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(
            y_pr,
            y_gt,
            beta=1.0,
            eps=self.eps,
            threshold=None,
            activation=self.activation,
        )


class BCEDiceLoss(DiceLoss):
    """BCEDiceLoss , equal weighting of the BCE and DICE losses

    Parameters
    ----------
    DiceLoss : torch loss
        parent class

    Returns
    -------
    float
        value of the BCED
    """

    __name__ = "bce_dice_loss"

    def __init__(self, eps=1e-7, activation="sigmoid", lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if self.activation is None:
            self.bce = nn.BCELoss(reduction="mean")
        else:
            self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice * dice) + (self.lambda_bce * bce)


class DiceLossTorch(nn.Module):
    """DiceLossTorch

    a pure PyTorch implementation of the Dice loss

    Parameters
    ----------
    nn : torch nn module
        parent class
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLossTorch, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    """DiceBCELoss

    a pure PyTorch implementation of the DiceBCELoss

    Parameters
    ----------
    nn : torch nn module
        parent class
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        intersection = torch.sum(inputs * targets)
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            torch.sum(inputs) + torch.sum(targets) + smooth
        )
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        # unsafe to cast
        BCE = nn.BCEWithLogitsLoss(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    """IoULoss _summary_

    a pure PyTorch implementation of the Intersection over Union (Jaccard) loss

    Parameters
    ----------
    nn : torch nn module
        parent class
    """

    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    """FocalLoss _summary_

    a pure PyTorch implementation of the Focal loss with ALPHA = 0.8 and GAMMA = 2

    Parameters
    ----------
    nn : torch nn module
        parent class
    """

    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss
