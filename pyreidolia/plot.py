import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

from typing import Any, Callable, Union, List, Tuple, Optional
from palettable.cartocolors.qualitative import Bold_10
from matplotlib import patches as patches

from pyreidolia.mask import rle_to_mask, bounding_box, get_mask_cloud


def set_my_plt_style(
    height: int = 3, width: int = 5, linewidth: Union[float, int] = 2
) -> plt.figure:
    """This set the style of matplotlib to fivethirtyeight with some modifications (colours, axes)

    Parameters
    ----------
    height :
        fig height in inches (yeah they're still struggling with the metric system) (default ``3``)
    width :
        fig width in inches (yeah they're still struggling with the metric system) (default ``5``)
    linewidth :
         (default ``2``)

    """
    plt.style.use("fivethirtyeight")
    my_colors_list = Bold_10.hex_colors
    myorder = [2, 3, 4, 1, 0, 6, 5, 8, 9, 7]
    my_colors_list = [my_colors_list[i] for i in myorder]
    bckgnd_color = "#f5f5f5"
    params = {
        "figure.figsize": (width, height),
        "axes.prop_cycle": plt.cycler(color=my_colors_list),
        "axes.facecolor": bckgnd_color,
        "patch.edgecolor": bckgnd_color,
        "figure.facecolor": bckgnd_color,
        "axes.edgecolor": bckgnd_color,
        "savefig.edgecolor": bckgnd_color,
        "savefig.facecolor": bckgnd_color,
        "grid.color": "#d2d2d2",
        "lines.linewidth": linewidth,
        "grid.alpha": 0.5,
    }  # plt.cycler(color=my_colors_list)
    mpl.rcParams.update(params)


def plot_cloud(
    img_path: str,
    img_id: str,
    label_mask: str,
    figsize: Tuple[int] = (20, 10),
    ax: Optional[matplotlib.axes.Axes] = None,
):
    """Plot a cloud with a mask

    Parameters
    ----------
    img_path :
        image path
    img_id :
        image id
    label_mask :
        mask label
    figsize :
        the figure size, by default (20,10)
    ax :
        existing axes, if any, by default None

    Returns
    -------
    matplotlib.axes.Axes
        matplotlib axes
    """
    img = cv2.imread(os.path.join(img_path, img_id))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    cmaps = {"Fish": "Blues", "Flower": "Reds", "Gravel": "Greens", "Sugar": "Oranges"}
    colors = {"Fish": "Blue", "Flower": "Red", "Gravel": "Green", "Sugar": "Orange"}
    for label, mask in label_mask:
        mask_decoded = rle_to_mask(mask, size=img.shape)
        if mask is not None:
            rmin, rmax, cmin, cmax = bounding_box(mask_decoded)
            bbox = patches.Rectangle(
                (cmin, rmin),
                cmax - cmin,
                rmax - rmin,
                linewidth=1.5,
                edgecolor=colors[label],
                facecolor="none",
            )
            ax.add_patch(bbox)
            ax.text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))
            ax.imshow(mask_decoded, alpha=0.5, cmap=cmaps[label])
            ax.text(cmin, rmin, label, bbox=dict(fill=True, color=colors[label]))
    return ax


def plot_rnd_cloud(
    img_path: str,
    grouped_masks: pd.Series,
    n_samples: int = 9,
    figsize: Tuple[int] = (20, 10),
    show: bool = True,
):
    """Plot random images from the dataset

    Parameters
    ----------
    img_path :
        image path
    grouped_masks :
        the series with tuple (label, mask)
    n_samples :
        number of images to plot, by default 9
    figsize :
        figure size, by default (20,10)
    show :
        display or not, by default True

    Returns
    -------
    matplotlib.pyplot.figure
        the matplotlib figure
    """

    n_cols = int(np.floor(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    n_subplots = n_rows * n_cols

    fig, axs = plt.subplots(figsize=figsize, ncols=n_cols, nrows=n_rows)

    if n_samples > 1:
        axs = axs.flatten()

    count = 0
    for image_id, label_mask in grouped_masks.sample(n_samples).iteritems():
        _ = plot_cloud(
            img_path=img_path, img_id=image_id, label_mask=label_mask, ax=axs[count]
        )
        count += 1

    if n_subplots > n_samples > 1:
        for i in range(n_samples, n_subplots):
            ax = axs[i]
            ax.set_axis_off()

    # Display the figure
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def draw_label_only(
    label: str, train_df: pd.DataFrame, train_path: str, figsize: Tuple[int] = (16, 6)
):
    """Draw only the clipped part of the image corresponding to the label

    Parameters
    ----------
    label :
        cloud label to draw
    train_df :
        the dataframe with image info
    train_path :
        the images path
    figsize :
        figure size, by default (16,6)
    """
    samples_df = train_df[
        (~train_df["encoded_pixels"].isnull()) & (train_df["label"] == label)
    ].sample(2)
    count = 0
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize)
    for idx, sample in samples_df.iterrows():
        img = get_mask_cloud(
            train_path, sample["image_id"], sample["label"], sample["encoded_pixels"]
        )
        ax[count].imshow(img, cmap="gray")
        count += 1
    plt.suptitle(f"Illustration of {label} cloud pattern")
    plt.tight_layout()


def plot_cloud_column(
    image: Union[np.ndarray, np.array],
    mask: Union[np.ndarray, np.array],
    original_image: Optional[Union[np.ndarray, np.array]] = None,
    original_mask: Optional[Union[np.ndarray, np.array]] = None,
):
    """Plot image and masks in different charts (columns)
    If two pairs of images and masks are passes, show both.

    Parameters
    ----------
    image :
        images to plot
    mask :
        mask to plot
    original_image :
        original image to plot, if any, by default None
    original_mask :
        original mask to plot, if any, by default None
    """
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f"Mask {class_dict[i]}", fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title("Original image", fontsize=fontsize)

        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title("Transformed image", fontsize=fontsize)

        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(
                f"Transformed mask {class_dict[i]}", fontsize=fontsize
            )


def visualize_with_raw(
    image: Union[np.ndarray, np.array],
    mask: Union[np.ndarray, np.array],
    original_image: Optional[Union[np.ndarray, np.array]] = None,
    original_mask: Optional[Union[np.ndarray, np.array]] = None,
    raw_image: Optional[Union[np.ndarray, np.array]] = None,
    raw_mask: Optional[Union[np.ndarray, np.array]] = None,
):
    """Plot image and masks in different charts (columns)
    If two pairs of images and masks are passes, show both.

    Parameters
    ----------
    image :
        images to plot
    mask :
        mask to plot
    original_image :
        original image to plot, if any, by default None
    original_mask :
        original mask to plot, if any, by default None
    raw_image :
        raw image to plot, if any, by default None
    raw_mask :
        raw mask to plot, if any, by default None
    """
    fontsize = 14
    class_dict = {0: "Fish", 1: "Flower", 2: "Gravel", 3: "Sugar"}

    f, ax = plt.subplots(3, 5, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
        ax[0, i + 1].imshow(original_mask[:, :, i])
        ax[0, i + 1].set_title(f"Original mask {class_dict[i]}", fontsize=fontsize)

    ax[1, 0].imshow(raw_image)
    ax[1, 0].set_title("Original image", fontsize=fontsize)

    for i in range(4):
        ax[1, i + 1].imshow(raw_mask[:, :, i])
        ax[1, i + 1].set_title(f"Raw predicted mask {class_dict[i]}", fontsize=fontsize)

    ax[2, 0].imshow(image)
    ax[2, 0].set_title("Transformed image", fontsize=fontsize)

    for i in range(4):
        ax[2, i + 1].imshow(mask[:, :, i])
        ax[2, i + 1].set_title(
            f"Predicted mask with processing {class_dict[i]}", fontsize=fontsize
        )


def plot_with_augmentation(image, mask, augment):
    """
    Wrapper for `plot_cloud_column` function.
    """
    augmented = augment(image=image, mask=mask)
    image_flipped = augmented["image"]
    mask_flipped = augmented["mask"]
    plot_cloud_column(
        image_flipped, mask_flipped, original_image=image, original_mask=mask
    )
