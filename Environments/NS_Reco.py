from __future__ import print_function
import numpy as np
from Src.Utils.utils import Space
import matplotlib.pyplot as plt

"""
"""

class NS_Reco(object):
    def __init__(self,
                 speed=2,
                 oracle=-1,
                 debug=True):

        self.debug = debug
        self.n_max_actions = 5
        self.state_dim = 1
        self.max_horizon = 1
        self.speed = speed
        self.oracle = oracle

        # The state and action space of the domain.
        self.action_space = Space(size=self.n_max_actions)
        self.observation_space = Space(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.state = np.array([1])          # State is always 1

        # Time counter
        self.episode = 0

        # Reward associated with each arm is computed based on
        # sinusoidal wave of varying amplitude and frequency
        rng = np.random.RandomState(1)
        self.amplitude = rng.rand(self.n_max_actions)

        rng = np.random.RandomState(0)
        self.frequency = rng.rand(self.n_max_actions) * self.speed * 0.005

        # Add noise of different variances to each arm
        rng = np.random.RandomState(0)
        self.stds = rng.rand(self.n_max_actions) * 0.01

        if self.oracle >= 0:
            self.amplitude = self.amplitude * np.sin(self.oracle * self.frequency)
            self.speed = 0

        print("Reward Amplitudes: {} :: Avg {} ".format(self.amplitude, np.mean(self.amplitude)))

        self.reset()

    def seed(self, seed):
        self.seeding = seed

    def reset(self):
        return self.state

    def step(self, action):
        assert 0 <= action < self.n_max_actions

        if self.speed == 0:
            all = self.amplitude + np.random.randn(self.n_max_actions) * self.stds
        else:
            all = self.amplitude * np.sin(self.episode * self.frequency) + np.random.randn(self.n_max_actions) * self.stds

        reward = all[action]

        # Compute reward
        # reward = self.amplitude[action] * np.sin(self.episode * self.frequency[action]) \
        #          + np.random.randn()*self.stds[action]

        # The episode always ends in one step.
        self.episode += 1
        return self.state, reward, True, {'Max': np.max(all)}

    def get_rewards(self):
        if self.speed == 0:
            return self.amplitude + np.random.randn(self.n_max_actions) * self.stds
        else:
            return self.amplitude * np.sin(self.episode * self.frequency) + np.random.randn(self.n_max_actions) * self.stds


if __name__=="__main__":
    # Plotting Agent
    rewards_list = []
    n_actions = 5
    epochs = 1000
    speed = 4
    all_rewards = np.zeros((n_actions, epochs))
    env = NS_Reco(speed=speed, debug=True)

    for a in range(n_actions):
        env.episode = 0
        for i in range(epochs):
            state = env.reset()
            _, r, _, _ = env.step(a)
            all_rewards[a][i] = r



    import matplotlib as mpl
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 26

    # mpl.style.use('seaborn')  # https://matplotlib.org/users/style_sheets.html

    plt.rc('font', size=BIGGER_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    mpl.rc('legend', fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.set_title('Itemized Rewards(speed={})'.format(speed))
    # ax1.set_ylim(max_y)
    ax1.locator_params(nbins=5)
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0), useMathText=True)
    # plt.figure()


    colors = plt.cm.get_cmap('tab10', n_actions + 5)  # https://matplotlib.org/gallery/color/colormap_reference.html
    # ax1.set_prop_cycle(colors)

    for a in range(n_actions):
        ax1.plot(all_rewards[a], alpha=0.7, label=str(a+1), color=colors(a+5))

    # plt.title('Speed: {}'.format(speed))
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.plot(np.max(all_rewards + 0.05, axis=0), color='black')
    # plt.show()


    fig1.savefig('Recommender' + str(speed) + "_true.png", bbox_inches="tight")

    figLegend1 = plt.figure(figsize=(8, 0.5))
    l1 = plt.figlegend(*ax1.get_legend_handles_labels(), loc='upper center', fancybox=True, shadow=True, ncol=n_actions)
    for line in l1.get_lines():
        line.set_linewidth(5.0)
    figLegend1.savefig('item_legend.png', bbox_inches="tight")

# factor = 0.003
# for speed in range(9, 11):
#     x = np.arange(1000)
#     xx = x * factor * speed
#     plt.plot(np.cos(xx))
# plt.plot()
# plt.show()