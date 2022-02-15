from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import torch

from constants import DATA_DIR


@dataclass
class State:
    frame: torch.Tensor
    sp: float  # stamina points
    zoom: int


@dataclass
class Action:
    direction: int
    time_steps: int


def see_plot(pict, size=(6, 6), title: str = None):
    plt.figure(figsize=size)
    plt.imshow(pict, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.show()


def load_image(video, frame):
    path_to_video = f'{DATA_DIR}/sample_rgb_96/{video}/'
    path_to_frame = f'{path_to_video}/f_{frame}.jpg'
    p = cv2.imread(path_to_frame)
    return p[:, :, ::-1]


def set_device(device: str):
    if 'cuda' in device:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device

