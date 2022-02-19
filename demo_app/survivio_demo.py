import logging
import albumentations as A
import keyboard
import numpy as np
import pandas as pd
import cv2
import torch
from albumentations.pytorch import ToTensorV2

from src.models import ResNetUNetV2, DQN
from src.neural_env import NeuralEnv, inference_agent, agent_choice

from src.utils import (load_image, set_device,
                       get_next_state, load_model)

from constants import WEIGHTS_DIR, DATA_DIR

logging.getLogger().setLevel(logging.INFO)

params = {'DEVICE': set_device('cuda:0'),
          'BATCH': 1,
          'reward_confidence': 0.95}


def demo_app():
    dg = pd.read_csv(f'{DATA_DIR}/dataset_inventory_v2.csv')
    dg = dg[dg['video'] == 'RU2h8oKpuZA']   # DEBUG
    dg = dg[dg.zoom == 1].reset_index()

    users_model = load_model(ResNetUNetV2(3), 'resunet_v5', device=params['DEVICE'])
    agent_model = load_model(DQN(), 'dqn_v7', device=params['DEVICE'])

    # preprocess data
    i = 10
    p = load_image(dg['video'][i], dg['frame'][i])
    first_show = np.hstack((p, p))
    cv2.imshow('Play the game (wasd)!', cv2.resize(first_show[..., ::-1],
                                                   (96*8, 96*4),
                                                   interpolation=cv2.INTER_NEAREST))

    sp = torch.tensor([dg['sp'][i]]).to(params['DEVICE'])/100
    zoom = torch.tensor([dg['zoom'][i]]).to(params['DEVICE'])/15
    n = torch.tensor([4]).to(params['DEVICE'])/14

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
        cv2.waitKey(100)
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

        p_user = get_next_state(users_model, p_user, user_direction, sp, zoom, n)
        p_agent = get_next_state(users_model, p_agent, agent_direction, sp, zoom, n)

        game_img = torch.cat([p_user.permute(1, 2, 0), p_agent.permute(1, 2, 0)], dim=1).detach().cpu().numpy() / 2 + 0.5
        game_img = (game_img*255).astype(np.uint8)
        game_img = cv2.resize(game_img, (96*8, 96*4), interpolation=cv2.INTER_NEAREST)
        game_img = game_img[..., ::-1]


        # s_curr, supp_curr, reward, act = inference_agent(q_model,
        #                                                  env,
        #                                                  torch.rot90(s_curr, 0, [1, 2]),
        #                                                  supp_curr,
        #                                                  device=params['DEVICE'])
        # s_current = s_curr.squeeze(0)
        # q_model_action = list(directions_map.keys())[act]


        # img = p.permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        # img = cv2.resize(img, (96*4, 96*4), interpolation=cv2.INTER_NEAREST)
        # img = img[..., ::-1]
        # img = (img*255).astype(np.uint8)
        #
        # rectangle = img.copy()
        # cv2.rectangle(rectangle, (0, 0), (190, 30), (0, 0, 0), -1)
        # img = cv2.addWeighted(rectangle, 0.6, img, 0.4, 0)
        # cv2.putText(img, 'Action:', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(img, text, (80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        #
        # # ----- q_model predictions -----
        # img_q_model = s_current.permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5
        # img_q_model = cv2.resize(img_q_model, (96*4, 96*4), interpolation=cv2.INTER_NEAREST)
        # img_q_model = img_q_model[..., ::-1]
        # img_q_model = (img_q_model*255).astype(np.uint8)
        #
        # rectangle = img_q_model.copy()
        # cv2.rectangle(rectangle, (0, 0), (190, 30), (0, 0, 0), -1)
        # img_q_model = cv2.addWeighted(rectangle, 0.6, img_q_model, 0.4, 0)
        # cv2.putText(img_q_model, 'Action:', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(img_q_model, q_model_action, (80, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        #
        # # сделать reset
        # # сделать rl агента справа (concat)
        #
        # game_img = np.hstack((img, img_q_model))
        cv2.imshow('Play the game (wasd)!', game_img)
        cv2.waitKey(1)
