import numpy as np
import torch
from Src.Utils.utils import NeuralNet


class Policy(NeuralNet):
    def __init__(self, state_dim, config, action_dim=None):
        super(Policy, self).__init__()

        self.config = config
        self.state_dim = state_dim
        if action_dim is None:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = action_dim

    def init(self):
        temp, param_list = [], []
        for name, param in self.named_parameters():
            temp.append((name, param.shape))
            if 'var' in name:
                param_list.append(
                    {'params': param, 'lr': self.config.actor_lr / 100})  # Keep learning rate of variance much lower
            else:
                param_list.append({'params': param})
        self.optim = self.config.optim(param_list, lr=self.config.actor_lr)

        print("Policy: ", temp)


