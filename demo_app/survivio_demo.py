import logging
import albumentations as A
import keyboard
import pandas as pd
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from constants import WEIGHTS_DIR, DATA_DIR
from src.models import ResNetUNet_v2, StoneClassifier
from src.neural_env import NeuralEnv

from src.utils import (see_plot, load_image, set_device,
                       get_next_state, load_stone_classifier, load_resunet_v5)

logging.getLogger().setLevel(logging.INFO)

params = {'DEVICE': 'cuda:0',
          'BATCH': 1}

# if user really wants to specify device
params['DEVICE'] = set_device(params['DEVICE'])


def demo_app():
    dg = pd.read_csv(f'{DATA_DIR}/dataset_inventory_v2.csv')
    dg = dg[dg['video'] == 'RU2h8oKpuZA']   # DEBUG
    dg = dg[dg.zoom == 1].reset_index()

    stone_classifier = load_stone_classifier(StoneClassifier(), device=params['DEVICE'])
    resunet_v5 = load_resunet_v5(ResNetUNet_v2(3), device=params['DEVICE'])

    # preprocess data
    i = 10
    p = load_image(dg['video'][i], dg['frame'][i])
    cv2.imshow('Play the game (wasd)!', cv2.resize(p, (96*4, 96*4), interpolation=cv2.INTER_NEAREST))

    sp = dg['sp'][i]
    zoom = dg['zoom'][i]
    n = 4

    train_aug = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)),
                           ToTensorV2(transpose_mask=False),
                           ])
    p = train_aug(image=p)['image']


    # 0. build environment
    env = NeuralEnv(f'{WEIGHTS_DIR}/resunet_v5.pth',
                    f'{WEIGHTS_DIR}/nostone_stone_classifier.pth',
                    params['DEVICE'],
                    params['BATCH'])

    # 1. start session
    s_curr, supp_curr = env.reset()
    print('Init state:', s_curr.size(), supp_curr.size())

    # 2. choose action with some algorithm
    chosen_action = torch.tensor([1] * params['BATCH']).unsqueeze(1).to(params['DEVICE'])  # 1 = GO UP
    print('Actions:', chosen_action.size())

    # 3. get next state with environment
    s_next, supp_next, reward = env.step(s_curr, supp_curr, chosen_action)
    print('Next state:', s_next.size(), supp_next.size(), reward.size())

    # 4. go to [2] and repeat

    for k in range(params['BATCH']):
        s0 = (s_curr[k].permute(1, 2, 0).detach().cpu() + 1) / 2
        s1 = (s_next[k].permute(1, 2, 0).detach().cpu() + 1) / 2
        print('Reward:', reward[k].detach().cpu())
        print()
        # see_plot(torch.cat([s0, s1], dim=1), size=(8, 4))

    # these lines below is currently the main game:
    directions_map = {
        'W':    0,
        'WD':   1,
        'D':    2,
        'SD':   3,
        'S':    4,
        'SA':   5,
        'A':    6,
        'WA':   7,
    }

    logging.info('Now you can play the game with wasd')
    logging.info('To close the game press "e"')

    while True:
        cv2.waitKey(100)

        if keyboard.is_pressed('e'):
            exit()
        elif keyboard.is_pressed('w'):
            if keyboard.is_pressed('w+d'):
                direction = directions_map['WD']
            elif keyboard.is_pressed('w+a'):
                direction = directions_map['WA']
            else:
                direction = directions_map['W']

        elif keyboard.is_pressed('s'):
            if keyboard.is_pressed('s+a'):
                direction = directions_map['SA']
            elif keyboard.is_pressed('s+d'):
                direction = directions_map['SD']
            else:
                direction = directions_map['S']

        elif keyboard.is_pressed('a'):
            direction = directions_map['A']
        elif keyboard.is_pressed('d'):
            direction = directions_map['D']

        else:
            continue

        p = get_next_state(resunet_v5, p, direction, sp, zoom, n, device=params['DEVICE'])

        temp = p.permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        temp = cv2.resize(temp, (96*4, 96*4), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Play the game (wasd)!', temp)
        cv2.waitKey(1)
