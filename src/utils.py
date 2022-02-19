from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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


def load_image(video, frame):
    path_to_video = f'{DATA_DIR}/sample_rgb_96/{video}/'
    path_to_frame = f'{path_to_video}/f_{frame}.jpg'
    p = cv2.imread(path_to_frame)
    return p[:, :, ::-1]


def load_resunet_v5(resunet_v5, device='cpu'):
    resunet_v5.load_state_dict(torch.load(f'{WEIGHTS_DIR}/resunet_v5.pth', map_location=device))
    resunet_v5 = resunet_v5.to(device)
    resunet_v5.eval();
    return resunet_v5


def load_stone_classifier(stone_classifier, device='cpu'):
    stone_classifier.load_state_dict(torch.load(f'{WEIGHTS_DIR}/nostone_stone_classifier.pth', map_location=device))
    stone_classifier = stone_classifier.to(device)
    stone_classifier.eval();
    return stone_classifier


def get_next_state(model, p, d, sp, zoom, n, device='cpu'):
    '''
    model - in gpu in eval mode
    p     - tensor of frame with variables in [-1,1]
    d     - direction, one of {1,2,3,4,5,6,7,8}
    sp    - sp in current frame, int/float
    zoom  - zoom in current frame, one of {1} (обучал только для zoom=1)
    n     - number of timestamps, one of {1,2,3,4,5,6,7,8,9,10,11,12,13,14}
    '''
    p = torch.clone(p).to(device)
    d = F.one_hot(torch.tensor(d), num_classes=8)
    sp = torch.tensor(sp)/100
    zoom = torch.tensor(zoom)/15
    n = torch.tensor(n/14)
    dd2 = torch.cat([d,
                     sp.unsqueeze(0),
                     zoom.unsqueeze(0),
                     n.unsqueeze(0)]).unsqueeze(0).float().to(device)
    with torch.no_grad():
        p = model((p.unsqueeze(0), dd2))[0]
    return p


def set_device(device: str):
    if 'cuda' in device:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')

    return device

