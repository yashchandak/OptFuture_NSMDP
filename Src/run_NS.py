#!~miniconda3/envs/pytorch/bin python
# from __future__ import print_function

import numpy as np
import Src.Utils.utils as utils
from Src.NS_parser import Parser
from Src.config import Config
from time import time
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env
        self.state_dim = np.shape(self.env.reset())[0]

        if len(self.env.action_space.shape) > 0:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n
        print("Actions space: {} :: State space: {}".format(self.action_dim, self.state_dim))

        self.model = config.algo(config=config)


    def train(self):
        # Learn the model on the environment
        return_history = []
        true_rewards = []
        action_prob = []

        ckpt = self.config.save_after
        rm_history, regret, rm, start_ep = [], 0, 0, 0
        # if self.config.restore:
        #     returns = list(np.load(self.config.paths['results']+"rewards.npy"))
        #     rm = returns[-1]
        #     start_ep = np.size(returns)
        #     print(start_ep)

        steps = 0
        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode

            state = self.env.reset()
            self.model.reset()

            step, total_r = 0, 0
            done = False
            while not done:
                # self.env.render(mode='human')
                action, extra_info, dist = self.model.get_action(state)
                new_state, reward, done, info = self.env.step(action=action)
                self.model.update(state, action, extra_info, reward, new_state, done)
                state = new_state

                # Tracking intra-episode progress
                total_r += reward
                # regret += (reward - info['Max'])
                step += 1
                if step >= self.config.max_steps:
                    break

            # track inter-episode progress
            # returns.append(total_r)
            steps += step
            # rm = 0.9*rm + 0.1*total_r
            rm += total_r
            if episode%ckpt == 0 or episode == self.config.max_episodes-1:
                rm_history.append(rm)
                return_history.append(total_r)
                if self.config.debug and self.config.env_name == 'NS_Reco':
                    action_prob.append(dist)
                    true_rewards.append(self.env.get_rewards())

                print("{} :: Rewards {:.3f} :: steps: {:.2f} :: Time: {:.3f}({:.5f}/step) :: Entropy : {:.3f} :: Grads : {}".
                      format(episode, rm, steps/ckpt, (time() - t0)/ckpt, (time() - t0)/steps, self.model.entropy, self.model.get_grads()))

                # self.model.save()
                utils.save_plots(return_history, config=self.config, name='{}_rewards'.format(self.config.seed))

                t0 = time()
                steps = 0


        if self.config.debug and self.config.env_name == 'NS_Reco':

            fig1, fig2 = plt.figure(figsize=(8, 6)), plt.figure(figsize=(8, 6))
            ax1, ax2 = fig1.add_subplot(1, 1, 1), fig2.add_subplot(1, 1, 1)

            action_prob = np.array(action_prob).T
            true_rewards = np.array(true_rewards).T

            for idx in range(len(dist)):
                ax1.plot(action_prob[idx])
                ax2.plot(true_rewards[idx])

            plt.show()


# @profile
def main(train=True, inc=-1, hyper='default', base=-1):
    t = time()
    args = Parser().get_parser().parse_args()

    # Use only on-policy method for oracle
    if args.oracle >= 0:
            args.algo_name = 'ONPG'

    if inc >= 0 and hyper != 'default' and base >= 0:
        args.inc = inc
        args.hyper = hyper
        args.base = base

    config = Config(args)
    solver = Solver(config=config)

    # Training mode
    if train:
        solver.train()

    print("Total time taken: {}".format(time()-t))

if __name__ == "__main__":
        main(train=True)

