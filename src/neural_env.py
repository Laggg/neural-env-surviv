import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms

from constants import DATA_DIR
from src.models import ResNetUNet_v2, StoneClassifier
from src.utils import load_image


class NeuralEnv:
    def __init__(self,
                 env_model_path: str,
                 reward_model_path: str,
                 device: str,
                 batch_size=16,
                 reward_confidence=0.5,
                 stone_frac=0.0,
                 step_size=4,
                 max_step=14):
        '''
        input params:
            env_model_path    [str] : path to model s_next=model(s_curr,action)
            reward_model_path [str] : path to model reward=model(s_curr)
            device            [str] : one of {'cpu', 'cuda:0', 'cuda:1'}
            batch_size        [int] : len of batch
            reward_confidence [flt] : classificator's confidence
            stone_frac        [flt] : part of the initial states with guaranteed stones
            step_size         [int] :
            max_step          [int] :
        output params:
            all output-variables will be torch.tensors in the selected DEVICE
            all input-variables have to be torch.tensors in the selected DEVICE
        '''
        self.device = device
        self.batch_size = batch_size
        self.reward_confidence = reward_confidence
        self.stone_frac = stone_frac
        self.step_size = step_size
        self.max_step = max_step
        self.reward_frame_transform = transforms.Compose([transforms.CenterCrop(24)])
        self.frame_transform = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)),
                                          ToTensorV2(transpose_mask=False)])

        self.model = ResNetUNet_v2(3)
        self.model.load_state_dict(torch.load(env_model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.stone_classifier = StoneClassifier()
        self.stone_classifier.load_state_dict(torch.load(reward_model_path, map_location=self.device))
        self.stone_classifier = self.stone_classifier.to(self.device)
        self.stone_classifier.eval()

        self.df = pd.read_csv(f'{DATA_DIR}/dataset_inventory_v2.csv')
        self.df = self.df[self.df.zoom == 1].reset_index()

    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        '''
        output params:
            init_s     [float torch tensor [-1...1]] : batch of initial states (batch,3,96,96)
            init_supp  [float torch tensor]          : batch of initial support vector (batch,2)
        '''
        init_s = torch.zeros(self.batch_size, 3, 96, 96).float()
        init_supp = torch.zeros(self.batch_size, 2).float()

        for i in range(self.batch_size):
            # j = np.random.randint(len(self.df))
            # frame = load_image(self.df["video"][j], self.df["frame"][j])
            j = np.random.randint(len(self.df[self.df['video'] == 'RU2h8oKpuZA']))  # DEBUG
            frame = load_image('RU2h8oKpuZA', self.df['frame'][j])                  # DEBUG
            frame = self.frame_transform(image=frame)['image']
            supp = torch.tensor([self.df["sp"][j] / 100, self.df['zoom'][j] / 15]).float()
            # if check_frame(frame)==True:
            #    init_s[i] = frame
            #    init_supp[i] = supp
            init_s[i] = frame
            init_supp[i] = supp
        return init_s.to(self.device), init_supp.to(self.device)

    # ----------------------------------------------------------------------------------------------------

    def get_reward(self, state):
        '''
        input params:
            state [float torch.tensor [-1...1]] : batch of states (batch,3,96,96)
        output params:
            r      [float torch.tensor [0...1]]  : batch of rewards (batch,1)
        '''
        state = self.reward_frame_transform(state)
        with torch.no_grad():
            r = self.stone_classifier(state)[:, 1].unsqueeze(1)
        r = (r > self.reward_confidence).float().detach()
        return r

    # ----------------------------------------------------------------------------------------------------

    def step(self, s_curr, supp_curr, action):
        '''
        input params:
            s_curr    [float torch.tensor [-1...1]] : batch of current states (batch,3,96,96)
            supp_curr [float torch tensor]          : batch of current support vector (batch,2)
            action    [int torch tensor {1,...,8}]  : batch of chosen direction (batch,1)
        output params:
            s_next    [float torch.tensor [-1...1]] : batch of next states (batch,3,96,96)
            supp_next [float torch tensor]          : batch of next support vector =supp_curr (batch,2)
            reward    [float torch.tensor [0...1]]  : batch of rewards (batch,1)
        '''
        # we need action_ohe to be (batch, 8)
        action_ohe = F.one_hot(action.squeeze() - 1, num_classes=8).float()
        if len(action_ohe.size()) == 1:  # if we don't have batch dim
            action_ohe = action_ohe.unsqueeze(0)

        n = torch.tensor([self.step_size / self.max_step] * self.batch_size)
        n = n.unsqueeze(1).float().to(self.device)  # (batch,1)
        v = torch.cat([action_ohe, supp_curr, n], dim=1)  # (batch,8+2+1)
        with torch.no_grad():
            s_next = self.model((s_curr, v)).detach()
        reward = self.get_reward(s_next)
        return s_next, supp_curr, reward
