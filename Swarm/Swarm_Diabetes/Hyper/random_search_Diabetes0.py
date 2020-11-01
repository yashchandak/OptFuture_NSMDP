import itertools
from collections import OrderedDict
import numpy as np

choice = np.random.choice
uniform = np.random.uniform

def loguniform(low=0.0, high=1.0, size=None, base=10):
    return np.power(base, np.random.uniform(low, high, size))

def set(obj, idx):
    # The names should be the same as argument names in parser.py
    args = OrderedDict()
    args['experiment'] = 'NS'
    seeds = np.arange(10)

    # Any hyper-parameter combination, should be evaluated exactly once for all the above seeds.
    n_seeds = len(seeds)

    rem = idx % n_seeds  # Determines hyper-param setting. Once all the seeds have been used, this increments by one.
    q = idx // n_seeds   # Determines seed for the current hyper-param

    np.random.seed(q)  # seed for choosing hyper-param
    args['seed'] = seeds[rem]  # seed for training the model parameters

    # Hyper-parma search
    args['speed'] = choice([0, 1, 2, 3, 4])
    args['oracle'] = -1
    args['env_name'] = choice(['NS_SimGlucose-v0'])
    args['algo_name'] = choice(['ProOLS'])

    args['extrapolator_basis'] = choice(['Fourier'])
    args['batch_size'] = choice([1000])
    args['delta'] = choice([1, 3, 5])
    args['max_inner'] = args['delta'] * choice([10, 20, 30])
    args['fourier_k'] = choice([2, 3, 4])
    args['importance_clip'] = choice([5.0, 10.0])
    # args['entropy_lambda'] = uniform(0.5, 0.05)

    args['actor_lr'] = loguniform(-4, -2)
    args['gamma'] = 0.99 #uniform(0.999, 1)
    # args['gauss_std'] = choice([-1, uniform(0.5, 1.5)])
    args['gauss_std'] = uniform(0.5, 2.5)

    # Fixed Hyper-params
    args['raw_basis'] = True
    args['fourier_order'] = 3
    # args['NN_basis_dim'] = '256'
    args['buffer_size'] = int(1e3)
    args['max_episodes'] = int(1e3)
    args['save_count'] = 100
    args['optim'] = 'rmsprop'

    # Logging and checkpoints
    args['swarm'] = True
    args['debug'] = False
    args['restore'] = False
    args['save_model'] = False
    args['log_output'] = 'term' #''file_term'
    args['gpu'] = 0

    print('Experiment: {} :: Seed: {}'.format(q, rem))

    # Create command
    folder_suffix = str(q)
    for name, value in args.items():
        if type(value) == float:
            value = round(value, 5)
        setattr(obj, name, value)
        if name != 'algo_name' and name != 'seed':
            folder_suffix += "_" + str(value)
    setattr(obj, 'folder_suffix', folder_suffix)

