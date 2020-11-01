from __future__ import print_function
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
from Src.Utils.utils import Space


class NS_Reacher(object):
    def __init__(self,
                 action_type='discrete',  # 'discrete' {0,1} or 'continuous' [0,1]
                 n_actions=4,
                 oracle=-1,
                 speed=4,
                 debug=True,
                 max_step_length=0.15,
                 max_steps=2):

        n_actions = 4
        self.debug = debug

        # NS Specific settings
        self.oracle = oracle
        self.speed = speed
        self.frequency = self.speed * 0.001
        self.episode = 0

        self.n_actions = n_actions
        self.action_space = Space(size=n_actions)
        self.observation_space = Space(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        self.disp_flag = False

        self.motions = self.get_action_motions(self.n_actions)

        self.wall_width = 0.05
        self.step_unit = self.wall_width - 0.005
        self.repeat = int(max_step_length / self.step_unit)

        self.max_horizon = int(max_steps / max_step_length)
        self.step_reward = -0.5
        self.collision_reward = 0  # -0.05
        self.movement_reward = 0  # 1
        self.randomness = 0.25

        # No lidars used
        self.n_lidar = 0
        self.angles = np.linspace(0, 2 * np.pi, self.n_lidar + 1)[:-1]  # Get 10 lidar directions,(11th and 0th are same)
        self.lidar_angles = list(zip(np.cos(self.angles), np.sin(self.angles)))
        self.static_obstacles = self.get_static_obstacles()

        if debug:
            self.heatmap_scale = 99
            self.heatmap = np.zeros((self.heatmap_scale + 1, self.heatmap_scale + 1))

        self.reset()

    def seed(self, seed):
        self.seed = seed

    def render(self):
        x, y = self.curr_pos

        # ----------------- One Time Set-up --------------------------------
        if not self.disp_flag:
            self.disp_flag = True
            # plt.axis('off')
            self.currentAxis = plt.gca()
            plt.figure(1, frameon=False)                            #Turns off the the boundary padding
            self.currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
            self.currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
            plt.ion()                                               #To avoid display blockage

            self.circle = Circle((x, y), 0.01, color='red')
            for coords in self.static_obstacles:
                x1, y1, x2, y2 = coords
                w, h = x2-x1, y2-y1
                self.currentAxis.add_patch(Rectangle((x1, y1), w, h, fill=True, color='gray'))
            print("Init done")
        # ----------------------------------------------------------------------

        for key, val in self.dynamic_obs.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True, color='black')
                self.currentAxis.add_patch(self.objects[key])


        for key, val in self.reward_states.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True)
                self.currentAxis.add_patch(self.objects[key])

        if len(self.angles) > 0:
            r = self.curr_state[-10:]
            coords = zip(r * np.cos(self.angles), r * np.sin(self.angles))

            for i, (w, h) in enumerate(coords):
                self.objects[str(i)] = Arrow(x, y, w, h, width=0.01, fill=True, color='lightgreen')
                self.currentAxis.add_patch(self.objects[str(i)])

        self.objects['circle'] = Circle((x, y), 0.01, color='red')
        self.currentAxis.add_patch(self.objects['circle'])

        # remove all the dynamic objects
        plt.pause(1e-7)
        for _, item in self.objects.items():
            item.remove()
        self.objects = {}

    def set_rewards(self):
        # All rewards
        self.G1_reward = 100 #100

    def reset(self):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.episode += 1
        self.set_rewards()
        self.steps_taken = 0
        self.reward_states = self.get_reward_states()
        self.dynamic_obs = self.get_dynamic_obstacles()
        self.objects = {}

        #x = 0.25
        #x = np.clip(x + np.random.randn()/30, 0.15, 0.35) # Add noise to initial x position
        self.curr_pos = np.array([0.5, 0.5])
        self.curr_state = self.make_state()

        return self.curr_state


    def get_action_motions(self, n_actions):
        shape = (n_actions, 2)
        motions = np.zeros(shape)

        motions[0] = [0, 1]
        motions[1] = [1, 0]
        motions[2] = [0, -1]
        motions[3] = [-1, 0]

        # Normalize to make maximium distance covered at a step be 1
        max_dist = np.max(np.linalg.norm(motions, ord=2, axis=-1))
        motions /= max_dist

        return motions

    def step(self, action):
        # action = binaryEncoding(action, self.n_actions) # Done by look-up table instead
        self.steps_taken += 1
        reward = 0

        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term = self.is_terminal()
        if term:
            return self.curr_state, 0, term, {'No INFO implemented yet'}

        motion = self.motions[action]  # Table look up for the impact/effect of the selected action
        reward += self.step_reward

        for i in range(self.repeat):
            if np.random.rand() < self.randomness:
                # Add noise some percentage of the time
                noise = np.random.rand(2)/1.415  # normalize by max L2 of noise
                delta = noise * self.step_unit  # Add noise some percentage of the time
            else:
                delta = motion * self.step_unit

            new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action

            if self.valid_pos(new_pos):
                dist = np.linalg.norm(delta)
                reward += self.movement_reward * dist  # small reward for moving
                if dist >= self.wall_width:
                    print("ERROR: Step size bigger than wall width", new_pos, self.curr_pos, dist, delta, motion, self.step_unit)

                self.curr_pos = new_pos
                reward += self.get_goal_rewards(self.curr_pos)
                # reward += self.open_gate_condition(self.curr_pos)
            else:
                reward += self.collision_reward
                break

            # To avoid overshooting the goal
            if self.is_terminal():
                break

            # self.update_state()
            self.curr_state = self.make_state()

        if self.debug:
            # Track the positions being explored by the agent
            x_h, y_h = self.curr_pos*self.heatmap_scale
            self.heatmap[min(int(y_h), 99), min(int(x_h), 99)] += 1

            ## For visualizing obstacle crossing flaw, if any
            # for alpha in np.linspace(0,1,10):
            #     mid = alpha*prv_pos + (1-alpha)*self.curr_pos
            #     mid *= self.heatmap_scale
            #     self.heatmap[min(int(mid[1]), 99)+1, min(int(mid[0]), 99)+1] = 1

        return self.curr_state.copy(), reward, self.is_terminal(), {'No INFO implemented yet'}

    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        # Append lidar values
        for cosine, sine in self.lidar_angles:
            r, r_prv = 0, 0
            pos = (x+r*cosine, y+r*sine)
            while self.valid_pos(pos) and r < 0.5:
                r_prv = r
                r += self.step_unit
                pos = (x+r*cosine, y+r*sine)
            state.append(r_prv)

        # Append the previous action chosen
        # state.extend(self.curr_action)

        return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if reward and self.in_region(pos, region):
                self.reward_states[key] = (region, 0)  # remove reward once taken
                if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))

                return reward
        return 0

    def get_reward_states(self):
        if self.oracle >= 0:
            x = (0.9 * np.sin(self.oracle * self.frequency) + 1) / 2.0
            y = (0.9 * np.cos(self.oracle * self.frequency) + 1) / 2.0
        else:
            x = (0.9 * np.sin(self.episode * self.frequency) + 1) / 2.0
            y = (0.9 * np.cos(self.episode * self.frequency) + 1) / 2.0

        self.G1 = (x - 0.025, y - 0.025, x + 0.025, y + 0.025)
        return {'G1': (self.G1, self.G1_reward)}

    def get_dynamic_obstacles(self):
        """
        :return: dict of objects, where key = obstacle shape, val = on/off
        """
        return {}


    def get_static_obstacles(self):
        """
        Each obstacle is a solid bar, represented by (x,y,x2,y2)
        representing bottom left and top right corners,
        in percentage relative to board size

        :return: list of objects
        """
        obstacles = []
        return obstacles

    def valid_pos(self, pos):
        flag = True

        # Check boundary conditions
        if not self.in_region(pos, [0,0,1,1]):
            flag = False

        # Check collision with static obstacles
        for region in self.static_obstacles:
            if self.in_region(pos, region):
                flag = False
                break

        # Check collision with dynamic obstacles
        for key, val in self.dynamic_obs.items():
            region, cond = val
            if cond and self.in_region(pos, region):
                flag = False
                break

        return flag

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G1):
            return 1
        elif self.steps_taken >= self.max_horizon:
            return 1
        else:
            return 0

    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False


if __name__=="__main__":
    # Random Agent
    rewards_list = []
    env = NS_Reacher(debug=True, n_actions=4)
    for i in range(1000):
        rewards = 0
        done = False
        env.reset()
        while not done:
            env.render()
            action = np.random.randint(env.n_actions)
            next_state, r, done, _ = env.step(action)
            rewards += r
        rewards_list.append(rewards)

    print("Average random rewards: ", np.mean(rewards_list), np.sum(rewards_list))