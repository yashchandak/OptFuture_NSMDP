import argparse
from datetime import datetime


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Parameters for Hyper-param sweep
        parser.add_argument("--base", default=0, help="Base counter for Hyper-param search", type=int)
        parser.add_argument("--inc", default=1, help="Increment counter for Hyper-param search", type=int)
        parser.add_argument("--hyper", default='default', help="Which Hyper param settings")
        parser.add_argument("--seed", default=2, help="seed for variance testing", type=int)

        # General parameters
        parser.add_argument("--save_count", default=10, help="Number of ckpts for saving results and model", type=int)
        parser.add_argument("--optim", default='rmsprop', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",
                            choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model ckpts")
        parser.add_argument("--summary", default=True, type=self.str2bool,
                            help="--UNUSED-- Visual summary of various stats")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)
        parser.add_argument("--swarm", default=False, help="Running on swarm?", type=self.str2bool)

        # Book-keeping parameters
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
        parser.add_argument("--experiment", default='Test_runfolder', help="Name of the experiment")

        self.Env_n_Agent_args(parser)  # Decide the Environment and the Agent
        self.Main_AC_args(parser)  # General Basis, Policy, Critic
        self.NS(parser)  # Settings for stochastic action set

        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser

    def Env_n_Agent_args(self, parser):
        # parser.add_argument("--algo_name", default='OFPG', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='ONPG', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='ProOLS', help="Learning algorithm")
        parser.add_argument("--algo_name", default='ProWLS', help="Learning algorithm")
        # parser.add_argument("--env_name", default='NS_SimGlucose-v0', help="Environment to run the code")
        parser.add_argument("--env_name", default='NS_Reco', help="Environment to run the code")
        # parser.add_argument("--env_name", default='NS_Reacher', help="Environment to run the code")

        parser.add_argument("--max_episodes", default=int(1000), help="maximum number of episodes (75000)", type=int)
        parser.add_argument("--max_steps", default=500, help="maximum steps per episode (500)", type=int)

    def NS(self, parser):
        parser.add_argument("--buffer_size", default=int(1e3), help="Size of memory buffer (3e5)", type=int)
        parser.add_argument("--extrapolator_basis", default='Poly', help="Basis for least-square", choices=['Linear', 'Poly', 'Fourier'])
        parser.add_argument("--batch_size", default=1000, help="Batch size", type=int)
        parser.add_argument("--fourier_k", default=7, help="Terms in extrapolator fourier basis", type=int)
        parser.add_argument("--max_inner", default=150, help="Iterations per update", type=int)
        parser.add_argument("--delta", default=5, help="Time steps in future for optimization", type=int)
        parser.add_argument("--entropy_lambda", default=0.1, help="Lagrangian for policy's entropy", type=float)
        parser.add_argument("--importance_clip", default=10.0, help="Clip value for importance ratio", type=float)
        parser.add_argument("--oracle", default=-1000, help="NS Fixed at given episode", type=int)
        parser.add_argument("--speed", default=2, help="Speed of non-stationarity", type=int)

    def Main_AC_args(self, parser):
        parser.add_argument("--gamma", default=0.99, help="Discounting factor", type=float)
        parser.add_argument("--actor_lr", default=1e-2, help="Learning rate of actor", type=float)
        parser.add_argument("--state_lr", default=1e-3, help="Learning rate of state features", type=float)
        parser.add_argument("--gauss_std", default=1.5, help="Variance for gaussian policy", type=float)

        parser.add_argument("--raw_basis", default=True, help="No basis fn.", type=self.str2bool)
        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis", type=self.str2bool)
        parser.add_argument("--fourier_order", default=-1, help="Order of fourier basis, " +
                                                               "(if > 0, it overrides neural nets)", type=int)
        parser.add_argument("--NN_basis_dim", default='32', help="Shared Dimensions for Neural network layers")
        parser.add_argument("--Policy_basis_dim", default='32', help="Dimensions for Neural network layers for policy")
