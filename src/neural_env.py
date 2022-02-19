import numpy as np
import pandas as pd
import albumentations as A
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms

from constants import DATA_DIR
from src.models import ResNetUNetV2, StoneClassifier
from src.utils import load_image, apply_aug, load_model


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
        """
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
        """
        self.device = device
        self.batch_size = batch_size
        self.reward_confidence = reward_confidence
        self.stone_frac = stone_frac
        self.step_size = step_size
        self.max_step = max_step
        self.reward_frame_transform = transforms.Compose([transforms.CenterCrop(24)])
        self.frame_transform = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)),
                                          ToTensorV2(transpose_mask=False)])

        self.model = ResNetUNetV2(3)
        self.model = load_model(self.model, 'resunet_v5', device=self.device)

        self.stone_classifier = StoneClassifier()
        self.stone_classifier = load_model(self.stone_classifier, 'nostone_stone_classifier_v2.pth', device=self.device)

        self.df = pd.read_csv(f'{DATA_DIR}/dataset_inventory_v2.csv')
        self.df = self.df[self.df.zoom == 1].reset_index()

    # ----------------------------------------------------------------------------------------------------

    def reset(self):
        """
        output params:
            init_s     [float torch tensor [-1...1]] : batch of initial states (batch,3,96,96)
            init_supp  [float torch tensor]          : batch of initial support vector (batch,2)
        """
        init_s = torch.zeros(self.batch_size, 3, 96, 96).float()
        init_supp = torch.zeros(self.batch_size, 2).float()

        for i in range(self.batch_size):
            # j = np.random.randint(len(self.df))
            # frame = load_image(self.df["video"][j], self.df["frame"][j])
            j = np.random.randint(len(self.df[self.df['video'] == 'RU2h8oKpuZA']))  # DEBUG
            frame = load_image('RU2h8oKpuZA', self.df['frame'][j])                  # DEBUG

            aug = np.random.choice(np.arange(8), p=np.array([0.125] * 8))
            frame = apply_aug(frame, aug)
            frame = self.frame_transform(image=frame)['image']
            supp = torch.tensor([self.df['sp'][j] / 100, self.df['zoom'][j] / 15]).float()
            reward = self.get_reward(frame.unsqueeze(0).to(self.device))

            while reward > 0:
                frame = frame.unsqueeze(0).expand(8, 3, 96, 96).to(self.device)
                supp = supp.unsqueeze(0).expand(8, 2).to(self.device)
                act = torch.tensor([x for x in range(8)]).to(self.device)
                frame, supp, reward = self.step(frame, supp, act)
                k = torch.argmin(reward)
                reward = reward[k]
                frame = frame[k].detach().cpu()
                supp = supp[k].detach().cpu()
            # +условие, чтобы агент c вер-тью p ресался на кадре, на котором где-то есть камень
            init_s[i] = frame
            init_supp[i] = supp
        return init_s.to(self.device), init_supp.to(self.device)

    # ----------------------------------------------------------------------------------------------------

    def get_reward(self, state):
        """
        input params:
            state [float torch.tensor [-1...1]] : batch of states (batch,3,96,96)
        output params:
            r      [float torch.tensor [0...1]]  : batch of rewards (batch,1)
        """
        state = self.reward_frame_transform(state).to(self.device)
        with torch.no_grad():
            r = self.stone_classifier(state)[:, 1].unsqueeze(1)
        r = (r > self.reward_confidence).float().detach()
        return r

    # ----------------------------------------------------------------------------------------------------

    def step(self, s_curr, supp_curr, action):
        """
        input params:
            s_curr    [float torch.tensor [-1...1]] : batch of current states (batch,3,96,96)
            supp_curr [float torch tensor]          : batch of current support vector (batch,2)
            action    [int torch tensor {0,...,7}]  : batch of chosen direction (batch,)
        output params:
            s_next    [float torch.tensor [-1...1]] : batch of next states (batch,3,96,96)
            supp_next [float torch tensor]          : batch of next support vector =supp_curr (batch,2)
            reward    [float torch.tensor [0...1]]  : batch of rewards (batch,1)
        """

        action_ohe = F.one_hot(action, num_classes=8).float()   # (batch, 8)
        if len(action) == 1:
            action_ohe.unsqueeze(0)
        n = torch.tensor([self.step_size / self.max_step] * action.size()[0])
        n = n.unsqueeze(1).float().to(self.device)          # (batch, 1)
        v = torch.cat([action_ohe, supp_curr, n], dim=1)    # (batch, 8+2+1)
        with torch.no_grad():
            s_next = self.model((s_curr, v)).detach()
        reward = self.get_reward(s_next)
        return s_next, supp_curr, reward


# ----------------------------------------------------------------------------------------------------
def agent_choice(agent_model, pict):
    """
    input params:
        agent_model in eval mode - on chosen device
        pict (96, 96, 3) [-1, 1] - on chosen device

    output params:
        agent_action int on of {0, 1, ..., 7} - on chosen device
    """
    s_curr = pict.clone().unsqueeze(0)
    with torch.no_grad():
        agent_action = torch.argmax(agent_model(s_curr), dim=1)
    return agent_action


def inference_agent(model, env, s_init, supp_init, device='cpu'):
    """
    s_init (3,96,96) cuda
    supp_init (2,) cuda
    """
    model.eval()
    s_curr = s_init.clone().unsqueeze(0).to(device)
    print(f'1: {s_curr.size()=}')

    supp_curr = supp_init.clone().unsqueeze(0)

    with torch.no_grad():
        act = torch.argmax(model(s_curr), dim=1)
    s_curr, supp_curr, reward = env.step(s_curr, supp_curr, act)
    print(f'2: {s_curr.size()=}')

    return s_curr, supp_curr, reward, act



