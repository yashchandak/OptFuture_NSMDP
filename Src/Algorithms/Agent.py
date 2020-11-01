import numpy as np
import torch

"""
"""

class Agent:
    # Parent Class for all algorithms

    def __init__(self, config):
        self.state_low, self.state_high = config.env.observation_space.low, config.env.observation_space.high
        self.state_diff = self.state_high - self.state_low

        try:
            if config.env.action_space.dtype == np.float32:
                self.action_low, self.action_high = config.env.action_space.low, config.env.action_space.high
                self.action_diff = self.action_high - self.action_low
        except:
            print('-------------- Warning: Possible action type mismatch ---------------')

        self.state_dim = config.env.observation_space.shape[0]

        if len(config.env.action_space.shape) > 0:
            self.action_dim = config.env.action_space.shape[0]
        else:
            self.action_dim = config.env.action_space.n

        self.config = config
        self.entropy, self.tracker = 0, 0

        # Abstract class variables
        self.modules = None

    def init(self):
        if self.config.restore:
            self.load()

        for name, m in self.modules:
            m.to(self.config.device)

    def clear_gradients(self):
        for _, module in self.modules:
            module.optim.zero_grad()

    def save(self):
        if self.config.save_model:
            for name, module in self.modules:
                module.save(self.config.paths['ckpt'] + name+'.pt')

    def load(self):
        try:
            for name, module in self.modules:
                module.load(self.config.paths['ckpt'] + name + '.pt')
            print('Loaded model from last checkpoint...')
        except ValueError as error:
            print("Loading failed: ", error)

    def step(self, loss, clip_norm=False):
        self.clear_gradients()
        loss.backward()
        for _, module in self.modules:
            module.step(clip_norm)

    def reset(self):
        for _, module in self.modules:
            module.reset()

    def get_grads(self):
        grads = []
        if self.config.debug:
            for _, module in self.modules:
                temp = []
                for param in module.parameters():
                    try:
                        temp.append(np.mean(np.abs(param.grad.data.cpu().numpy())))
                        # temp.append(np.mean(np.abs(param.data.cpu().numpy())))
                    except:
                        pass
                grads.append(temp)
        return grads

    def track_entropy(self, act_probs, action):
        if self.config.debug:
            if self.config.cont_actions:
                # Not tracking entropy, rather just short term deviation from long term mean
                # More useful than normal entropy as it also indicates whether the mean is changing or not.
                # Also, in case of diagonal Gaussian, often the variance is kept constant, thus entropy is const as well
                self.entropy = 0.5 * self.entropy + 0.5 * np.sum((action - self.tracker)**2)
                self.tracker = 0.99 * self.tracker + 0.01*action
            else:
                act_probs = act_probs.cpu().data.numpy()
                curr_entropy = - np.sum(act_probs * np.log(act_probs + 1e-8))
                self.entropy = 0.9 * self.entropy + 0.1 * curr_entropy


    def track_entropy_cont(self, action):
        if self.config.debug:
            # Not tracking entropy, rather just short term deviation from long term mean
            # More useful than normal entropy as it also indicates whether the mean is changing or not.
            # Also, in case of diagonal Gaussian, often the variance is kept constant, thus entropy is const as well
            self.entropy = 0.5 * self.entropy + 0.5 * np.sum((action - self.tracker)**2)
            self.tracker = 0.99 * self.tracker + 0.01*action
