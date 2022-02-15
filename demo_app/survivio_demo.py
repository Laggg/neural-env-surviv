import torch

from constants import WEIGHTS_DIR
from src.neural_env import NeuralEnv
from src.utils import see_plot, set_device

params = {'DEVICE': 'cuda:0',
          'BATCH': 1}

# if user really wants to specify device
params['DEVICE'] = set_device(params['DEVICE'])


def demo_app():
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
        see_plot(torch.cat([s0, s1], dim=1), size=(8, 4))
