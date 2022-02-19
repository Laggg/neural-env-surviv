from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torchvision.transforms import transforms
from constants import DATA_DIR, WEIGHTS_DIR


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


def load_image(path_to_frame):
    # path_to_video = f'{DATA_DIR}/sample_rgb_96/{video}/'
    # path_to_frame = f'{path_to_video}/f_{frame}.jpg'
    p = cv2.imread(path_to_frame)
    return p[:, :, ::-1]


def load_model(model, model_name, device='cpu'):
    model.load_state_dict(torch.load(f'{WEIGHTS_DIR}/{model_name}.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model


def apply_aug(p0, aug):
    if aug == 0:
        p = p0.copy()
    elif aug == 1:
        p = cv2.rotate(p0, cv2.ROTATE_90_CLOCKWISE)
    elif aug == 2:
        p = cv2.rotate(p0, cv2.ROTATE_180)
    elif aug == 3:
        p = cv2.rotate(p0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug == 4:
        p = cv2.flip(p0, 1)
    elif aug == 5:
        p = cv2.rotate(p0, cv2.ROTATE_90_CLOCKWISE)
        p = cv2.flip(p, 1)
    elif aug == 6:
        p = cv2.rotate(p0, cv2.ROTATE_180)
        p = cv2.flip(p, 1)
    elif aug == 7:
        p = cv2.rotate(p0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        p = cv2.flip(p, 1)
    return p


def get_next_state(model, stone_cls, p, d, sp, zoom, n, reward_confidence):
    """
    input params (on pre-chosen device):
        model - in gpu in eval mode
        p     - tensor of frame with variables in [-1,1]
        d     - direction, one of {1,2,3,4,5,6,7,8}
        sp    - sp in current frame, int/float
        zoom  - zoom in current frame, one of {1} (обучал только для zoom=1)
        n     - number of timestamps, one of {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
    output params:
        p - next state frame with variables in [-1, 1]
    """
    p = torch.clone(p)
    d = F.one_hot(d, num_classes=8)

    # (1, 11)
    dd2 = torch.cat([d.unsqueeze(0),
                     sp.unsqueeze(0),
                     zoom.unsqueeze(0),
                     n.unsqueeze(0)], dim=1).float()
    with torch.no_grad():
        p = model((p.unsqueeze(0), dd2))[0]

    reward_frame_transform = transforms.Compose([transforms.CenterCrop(24)])
    state = reward_frame_transform(p)   #.to(device)
    state = state[None]
    with torch.no_grad():
        r = stone_cls(state)[:, 1]
    r = (r > reward_confidence).float().detach()#.cpu()

    return p, r


def set_device(device: str):
    if 'cuda' in device:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device
