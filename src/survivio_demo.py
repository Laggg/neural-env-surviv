import logging
import os
import albumentations as A
import keyboard
import numpy as np
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from src.models import ResNetUNetV2, DQN, StoneClassifier
from src.neural_env import agent_choice

from src.utils import (load_image, set_device,
                       get_next_state, load_model)

from constants import DATA_DIR

logging.getLogger().setLevel(logging.INFO)

params = {'DEVICE': set_device('cuda:0'),
          'BATCH': 1,
          'reward_confidence': 0.95}


def demo_app():
    agent_icon = cv2.imread(f'{DATA_DIR}/assets/DefaultSurvivr39.png', cv2.IMREAD_UNCHANGED)
    agent_icon = cv2.resize(agent_icon, (24, 24), interpolation=cv2.INTER_AREA)
    agent_mask = np.ones((24, 24, 3))
    agent_mask[:, :, 0] = agent_icon[:, :, 3]
    agent_mask[:, :, 1] = agent_icon[:, :, 3]
    agent_mask[:, :, 2] = agent_icon[:, :, 3]
    agent_mask = agent_mask / 255.
    agent_icon = agent_icon[:, :, 0:3][:, :, ::-1]

    users_model = load_model(ResNetUNetV2(3), 'resunet_v5', device=params['DEVICE'])
    agent_model = load_model(DQN(), 'dqn_v7', device=params['DEVICE'])
    stone_classifier = load_model(StoneClassifier(), 'nostone_stone_classifier_v2', device=params['DEVICE'])

    # preprocess data
    init_frame_name = np.random.choice(os.listdir(f'{DATA_DIR}/initial_frames'))
    p = load_image(f'{DATA_DIR}/initial_frames/{init_frame_name}')

    first_show = cv2.resize(p[..., ::-1], (96*4, 96*4), interpolation=cv2.INTER_NEAREST)
    first_show[46*4:52*4, 46*4:52*4, :] = agent_icon[..., ::-1] * agent_mask + \
                                          first_show[46*4:52*4, 46*4:52*4, :] * (1 - agent_mask)
    first_show = np.hstack((first_show, first_show))

    cv2.imshow('Play the game (wasd)!', first_show)
    cv2.waitKey(300)

    sp = torch.tensor([0]).to(params['DEVICE'])
    zoom = torch.tensor([1]).to(params['DEVICE'])/15
    n = torch.tensor([2]).to(params['DEVICE'])/14

    reward_user = 0
    reward_agent = 0

    train_aug = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)),
                           ToTensorV2(transpose_mask=False),
                           ])
    p = train_aug(image=p)['image']     # [0, 255] -> [-1, 1] on CPU
    p = p.to(params['DEVICE'])
    p_user = p.clone()
    p_agent = p.clone()

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
        color = (255, 255, 255)
        text = 'UNKNOWN'

        if keyboard.is_pressed('e'):
            color = (55, 255, 255)
            cv2.putText(game_img, 'PRESSED `e` - closing...', (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow('Play the game (wasd)!', game_img)
            cv2.waitKey(3500)
            exit()
        elif keyboard.is_pressed('w'):
            if keyboard.is_pressed('w+d'):
                text = 'STRAIGHT+RIGHT'
                color = (155, 55, 255)
                direction = directions_map['WD']
            elif keyboard.is_pressed('w+a'):
                text = 'STRAIGHT+LEFT'
                color = (155, 55, 255)
                direction = directions_map['WA']
            else:
                text = 'STRAIGHT'
                color = (55, 255, 55)
                direction = directions_map['W']

        elif keyboard.is_pressed('s'):
            if keyboard.is_pressed('s+a'):
                text = 'DOWN+LEFT'
                color = (255, 155, 155)
                direction = directions_map['SA']
            elif keyboard.is_pressed('s+d'):
                text = 'DOWN+RIGHT'
                color = (255, 155, 155)
                direction = directions_map['SD']
            else:
                text = 'DOWN'
                color = (55, 255, 55)
                direction = directions_map['S']

        elif keyboard.is_pressed('a'):
            text = 'LEFT'
            color = (55, 155, 255)
            direction = directions_map['A']
        elif keyboard.is_pressed('d'):
            text = 'RIGHT'
            color = (55, 155, 255)
            direction = directions_map['D']

        else:
            continue

        agent_direction = agent_choice(agent_model, p_agent)  # ready on device
        user_direction = torch.tensor(direction).to(params['DEVICE'])
        agent_direction = agent_direction.squeeze()

        p_user, r_user = get_next_state(users_model, stone_classifier, p_user,
                                        user_direction, sp, zoom, n,
                                        params['reward_confidence'])
        p_agent, r_agent = get_next_state(users_model, stone_classifier, p_agent,
                                          agent_direction, sp, zoom, n,
                                          params['reward_confidence'])

        reward_user += int(r_user.item())
        reward_agent += int(r_agent.item())

        p_user_img = p_user.permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        p_agent_img = p_agent.permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        p_user_img = (p_user_img*255).astype(np.uint8)
        p_agent_img = (p_agent_img*255).astype(np.uint8)
        p_user_img = cv2.resize(p_user_img, (96*4, 96*4), interpolation=cv2.INTER_NEAREST)
        p_agent_img = cv2.resize(p_agent_img, (96*4, 96*4), interpolation=cv2.INTER_NEAREST)

        rectangle = p_user_img.copy()
        cv2.rectangle(rectangle, (0, 0), (190, 30), (0, 0, 0), -1)
        p_user_img = cv2.addWeighted(rectangle, 0.6, p_user_img, 0.4, 0)
        cv2.putText(p_user_img, f'User reward: {str(reward_user)}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # -------------------

        rectangle = p_agent_img.copy()
        cv2.rectangle(rectangle, (0, 0), (190, 30), (0, 0, 0), -1)
        p_agent_img = cv2.addWeighted(rectangle, 0.6, p_agent_img, 0.4, 0)
        cv2.putText(p_agent_img, f'Agent reward: {str(reward_agent)}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        p_user_img[46*4:52*4, 46*4:52*4, :] = agent_icon * agent_mask + p_user_img[46*4:52*4, 46*4:52*4, :] * (1 - agent_mask)
        p_agent_img[46*4:52*4, 46*4:52*4, :] = agent_icon * agent_mask + p_agent_img[46*4:52*4, 46*4:52*4, :] * (1 - agent_mask)

        game_img = np.hstack((p_user_img, p_agent_img))
        game_img = game_img[..., ::-1]

        cv2.imshow('Play the game (wasd)!', game_img)
        cv2.waitKey(1)
