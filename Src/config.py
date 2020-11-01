import sys
from yaml import dump
from os import path
import Src.Utils.utils as utils
import numpy as np
import torch
from collections import OrderedDict

class Config(object):
    def __init__(self, args):

        # SET UP PATHS
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..'))

        # Do Hyper-parameter sweep, if needed
        self.idx = args.base + args.inc
        if self.idx >= 0 and args.hyper != 'default':
            self.hyperparam_sweep = utils.dynamic_load(self.paths['root'], "random_search_" + str(args.hyper), load_class=False)
            self.hyperparam_sweep.set(args, self.idx)
            del self.hyperparam_sweep  # *IMP: CanNOT deepcopy an object having reference to an imported library (DPG)

        # Make results reproducible
        seed = args.seed
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Frequency of saving results and models.
        self.save_after = args.max_episodes // args.save_count if args.max_episodes > args.save_count else args.max_episodes

        # add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name, args.algo_name, folder_suffix)

        if args.swarm:
            # DO not create different folder for each seed in swarm
            # Takes up a lot of space and makes data aggregation slow
            self.paths['logs'] = self.paths['experiment'] + '/'
            self.paths['ckpt'] = self.paths['experiment'] + '/'
            self.paths['results'] = self.paths['experiment'] + '/'
        else:
            path_prefix = [self.paths['experiment'], str(args.seed)]
            self.paths['logs'] = path.join(*path_prefix, 'Logs/')
            self.paths['ckpt'] = path.join(*path_prefix, 'Checkpoints/')
            self.paths['results'] = path.join(*path_prefix, 'Results/')

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                utils.create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        # Output logging
        sys.stdout = utils.Logger(self.paths['logs'], args.restore, args.log_output)

        # Get the domain and algorithm
        self.env, self.gym_env, self.cont_actions = self.get_domain(args.env_name, args=args, debug=args.debug,
                                                               path=path.join(self.paths['root'], 'Environments'))
        try:
            self.env.seed(seed.item())
        except:
            self.env.seed(seed)

        # Set Model
        self.algo = utils.dynamic_load(path.join(self.paths['root'], 'Src', 'Algorithms'), args.algo_name, load_class=True)

        self.feature_dim = [int(size) for size in args.NN_basis_dim.split(',')]
        self.policy_basis_dim = [int(size) for size in args.Policy_basis_dim.split(',')]

        # GPU
        self.device = torch.device('cuda' if args.gpu else 'cpu')

        # optimizer
        if args.optim == 'adam':
            self.optim = torch.optim.Adam
        elif args.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop
        elif args.optim == 'sgd':
            self.optim = torch.optim.SGD
        else:
            raise ValueError('Undefined type of optmizer')


        print("=====Configurations=====\n", args)


    def get_domain(self, tag, args, path, debug=True):

        if tag == 'NS_Reco' or tag == 'NS_Reacher':
            obj = utils.dynamic_load(path, tag, load_class=True)
            env = obj(speed=args.speed, oracle=args.oracle, debug=debug)
            return env, False, env.action_space.dtype == np.float32

        else:
            import gym
            from gym.spaces.box import Box

            # Register gym environment. By specifying kwargs,
            # you are able to choose which patient to simulate.
            # patient_name must be 'adolescent#001' to 'adolescent#010',
            # or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
            from gym.envs.registration import register
            register(
                id='NS_SimGlucose-v0',
                entry_point='Environments.SimGlucose.simglucose.envs:NS_T1DSimEnv',
                kwargs={'patient_name': 'adolescent#002', 'speed': self.speed, 'oracle': self.oracle, 'seed': self.seed}
            )

            env = gym.make(tag)
            return env, True, isinstance(env.action_space, Box)


