from random import choice, randint
# from typing import TYPE_CHECKING
import cv2
import numpy as np

from detection.yolov3.utils.datasets import LoadImagesAndLabels
from detection.yolov3.utils.utils import plot_one_box, xywh2xyxy

# if TYPE_CHECKING:
from typing import Dict, List, Union

from matplotlib import axes
from matplotlib.image import AxesImage

def show_image(image: np.ndarray, ax: axes.Axes, **kwargs) -> AxesImage:
    return ax.imshow(image[:, :, ::-1], interpolation="nearest", **kwargs)

def show_random_image(dataset: LoadImagesAndLabels, ax: axes.Axes, classes: List[str]) -> AxesImage:
    n = randint(0, len(dataset))

    # Repeat till we have images with labels
    while len(list(dataset.labels)[n]) == 0:
        n = randint(0, len(dataset))

    image = cv2.imread(dataset.img_files[n])

    xyxy = xywh2xyxy(dataset.labels[n][:, 1:])
    xyxy *= [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]

    for i, lbl in enumerate(dataset.labels[n]):
        plot_one_box(
            xyxy[
                i,
            ],
            image,
            label=classes[int(lbl[0])],
        )

    return show_image(image, ax)