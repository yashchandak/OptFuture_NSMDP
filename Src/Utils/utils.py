from __future__ import print_function
import numpy as np
import torch
from torch import tensor, float32, int32
from torch.autograd import Variable
import torch.nn as nn
import shutil
import matplotlib.pyplot as plt
from os import path, mkdir, listdir, fsync
import importlib
from time import time
import sys
import pickle

np.random.seed(0)
torch.manual_seed(0)
dtype = torch.FloatTensor

class Logger(object):
    fwrite_frequency = 1800  # 30 min * 60 sec
    temp = 0

    def __init__(self, log_path, restore, method):
        self.terminal = sys.stdout
        self.file = 'file' in method
        self.term = 'term' in method

        if self.file:
            if restore:
                self.log = open(path.join(log_path, "logfile.log"), "a")
            else:
                self.log = open(path.join(log_path, "logfile.log"), "w")


    def write(self, message):
        if self.term:
            self.terminal.write(message)

        if self.file:
            self.log.write(message)

            # Save the file frequently
            if (time() - self.temp) > self.fwrite_frequency:
                self.flush()
                self.temp = time()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.

        # Save the contents of the file without closing
        # https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        # WARNING: Time consuming process, Makes the code slow if too many writes
        if self.file:
            self.log.flush()
            fsync(self.log.fileno())



def save_plots(rewards, config, name='rewards'):
    np.save(config.paths['results'] + name, rewards)
    if config.debug:
        if 'Grid' in config.env_name or 'room' in config.env_name:
            # Save the heatmap
            plt.figure()
            plt.title("Exploration Heatmap")
            plt.xlabel("100x position in x coordinate")
            plt.ylabel("100x position in y coordinate")
            plt.imshow(config.env.heatmap, cmap='hot', interpolation='nearest', origin='lower')
            plt.savefig(config.paths['results'] + 'heatmap.png')
            np.save(config.paths['results'] + "heatmap", config.env.heatmap)
            config.env.heatmap.fill(0)  # reset the heat map
            plt.close()

        plt.figure()
        plt.ylabel("Total return")
        plt.xlabel("Episode")
        plt.title("Performance")
        plt.plot(rewards)
        plt.savefig(config.paths['results'] + "performance.png")
        plt.close()


def plot(rewards):
    # Plot the results
    plt.figure(1)
    plt.plot(list(range(len(rewards))), rewards)
    plt.xlabel("Trajectories")
    plt.ylabel("Reward")
    plt.title("Baseline Reward")
    plt.show()


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.ctr = 0
        self.nan_check_fequency = 10000

    def custom_weight_init(self):
        # Initialize the weight values
        for m in self.modules():
            weight_init(m)

    def update(self, loss, retain_graph=False, clip_norm=False):
        self.optim.zero_grad()  # Reset the gradients
        loss.backward(retain_graph=retain_graph)
        self.step(clip_norm)

    def step(self, clip_norm):
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        self.optim.step()
        self.check_nan()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def check_nan(self):
        # Check for nan periodically
        self.ctr += 1
        if self.ctr == self.nan_check_fequency:
            self.ctr = 0
            # Note: nan != nan  #https://github.com/pytorch/pytorch/issues/4767
            for name, param in self.named_parameters():
                if (param != param).any():
                    raise ValueError(name + ": Weights have become nan... Exiting.")

    def reset(self):
        return




def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)



class Space:
    def __init__(self, low=[0], high=[1], dtype=np.uint8, size=-1):
        if size == -1:
            self.shape = np.shape(low)
        else:
            self.shape = (size, )
        self.low = np.array(low)
        self.high = np.array(high)
        self.dtype = dtype
        self.n = len(self.low)

def get_var_w(shape, scale=1):
    w = torch.Tensor(shape[0], shape[1])
    w = nn.init.xavier_uniform(w, gain=nn.init.calculate_gain('sigmoid'))
    return Variable(w.type(dtype), requires_grad=True)


def get_var_b(shape):
    return Variable(torch.rand(shape).type(dtype) / 100, requires_grad=True)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def weight_init(m):
    if isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = 0#.1/ np.sqrt((fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        # m.weight.data.normal_(0.0, 0.03)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()




def save_training_checkpoint(state, is_best, episode_count):
    """
    Saves the models, with all training parameters intact
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    filename = str(episode_count) + 'checkpoint.path.rar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



def search(dir, name, exact=False):
    all_files = listdir(dir)
    for file in all_files:
        if exact and name == file:
            return path.join(dir, name)
        if not exact and name in file:
            return path.join(dir, name)
    else:
        # recursive scan
        for file in all_files:
            if file == 'Experiments':
                continue
            _path = path.join(dir, file)
            if path.isdir(_path):
                location = search(_path, name, exact)
                if location:
                    return location

def dynamic_load(dir, name, load_class=False):
    try:
        abs_path = search(dir, name).split('/')[1:]
        pos = abs_path.index('OptFuture')
        module_path = '.'.join([str(item) for item in abs_path[pos + 1:]])
        print("Module path: ", module_path, name)
        if load_class:
            obj = getattr(importlib.import_module(module_path), name)
        else:
            obj = importlib.import_module(module_path)
        print("Dynamically loaded from: ", obj)
        return obj
    except:
        raise ValueError("Failed to dynamically load the class: " + name )

def check_n_create(dir_path, overwrite=False):
    try:
        if not path.exists(dir_path):
            mkdir(dir_path)
        else:
            if overwrite:
               shutil.rmtree(dir_path)
               mkdir(dir_path)
    except FileExistsError:
        print("\n ##### Warning File Exists... perhaps multi-threading error? \n")

def create_directory_tree(dir_path):
    dir_path = str.split(dir_path, sep='/')[1:-1]  #Ignore the blank characters in the start and end of string
    for i in range(len(dir_path)):
        check_n_create(path.join('/', *(dir_path[:i + 1])))


def remove_directory(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)


def clip_norm(params, max_norm=1):
    # return params
    norm_param = []
    for param in params:
        norm = np.linalg.norm(param, 2)
        if norm > max_norm:
            norm_param.append(param/norm * max_norm)
        else:
            norm_param.append(param)
    return norm_param




class TrajectoryBuffer:
    """
    Pre-allocated memory interface for storing and using Off-policy trajectories
    Note: slight abuse of notation.
          sometimes Code treats 'dist' as extra variable and uses it to store other things, like: prob, etc.
    """
    def __init__(self, buffer_size, state_dim, action_dim, atype, config, dist_dim=1, stype=float32):

        max_horizon = config.env.max_horizon

        self.s = torch.zeros((buffer_size, max_horizon, state_dim), dtype=stype, requires_grad=False, device=config.device)
        self.a = torch.zeros((buffer_size, max_horizon, action_dim), dtype=atype, requires_grad=False, device=config.device)
        self.beta = torch.ones((buffer_size, max_horizon), dtype=float32, requires_grad=False, device=config.device)
        self.mask = torch.zeros((buffer_size, max_horizon), dtype=float32, requires_grad=False, device=config.device)
        self.r = torch.zeros((buffer_size, max_horizon), dtype=float32, requires_grad=False, device=config.device)
        self.ids = torch.zeros(buffer_size, dtype=int32, requires_grad=False, device=config.device)

        self.buffer_size = buffer_size
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

        self.atype = atype
        self.stype = stype
        self.config = config

    @property
    def size(self):
        return self.valid_len

    def reset(self):
        self.episode_ctr = -1
        self.timestep_ctr = 0
        self.buffer_pos = -1
        self.valid_len = 0

    def next(self):
        self.episode_ctr += 1
        self.buffer_pos += 1

        # Cycle around to the start of buffer (FIFO)
        if self.buffer_pos >= self.buffer_size:
            self.buffer_pos = 0

        if self.valid_len < self.buffer_size:
            self.valid_len += 1

        self.timestep_ctr = 0
        self.ids[self.buffer_pos] = self.episode_ctr

        # Fill rewards vector with zeros because episode overwriting this index
        # might have shorter horizon than the previous episode cached in this vector.
        self.r[self.buffer_pos].fill_(0)
        self.mask[self.buffer_pos].fill_(0)

    def add(self, s1, a1, beta1, r1):
        pos = self.buffer_pos
        step = self.timestep_ctr

        self.s[pos][step] = torch.tensor(s1, dtype=self.stype)
        self.a[pos][step] = torch.tensor(a1, dtype=self.atype)
        self.beta[pos][step] = torch.tensor(beta1, dtype=float32)
        self.r[pos][step] = torch.tensor(r1, dtype=float32)
        self.mask[pos][step] = torch.tensor(1.0, dtype=float32)

        self.timestep_ctr += 1

    def _get(self, idx):
        # ids represent the episode number
        # idx represents the buffer index
        # Both are not the same due to use of wrap around buffer
        ids = self.ids[idx]

        if self.valid_len >= self.buffer_size:
            # Subtract off the minimum value idx (as the idx has wrapped around in buffer)
            if self.buffer_pos + 1 == self.buffer_size:
                ids -= self.ids[0]
            else:
                ids -= self.ids[self.buffer_pos + 1]

        return ids, self.s[idx], self.a[idx], self.beta[idx], self.r[idx], self.mask[idx]

    def sample(self, batch_size):
        count = min(batch_size, self.valid_len)
        return self._get(np.random.choice(self.valid_len, count))

    def get_all(self):
        return self._get(np.arange(self.valid_len))

    def batch_sample(self, batch_size, randomize=True):
        raise NotImplementedError

    def save(self, path, name):
        dict = {
                's': self.s,
                'a': self.a,
                'beta': self.beta,
                'mask': self.mask,
                'r': self.r,
                'ids': self.ids,
                'time': self.timestep_ctr, 'pos': self.buffer_pos, 'val': self.valid_len
        }
        with open(path + name + '.pkl', 'wb') as f:
            pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path, name):
        with open(path + name + '.pkl', 'rb') as f:
            dict = pickle.load(f)

        self.s = dict['s']
        self.a = dict['a']
        self.beta = dict['beta']
        self.mask = dict['mask']
        self.r = dict['r']
        self.ids = dict['ids']
        self.timestep_ctr, self.buffer_pos, self.valid_len = dict['time'], dict['pos'], dict['val']

        print('Memory buffer loaded..')
