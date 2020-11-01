import numpy as np
import torch
from torch import tensor, float32
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, utils
from Src.Algorithms import NS_utils
from Src.Algorithms.Extrapolator import WLS

"""

"""
class ProWLS(Agent):
    def __init__(self, config):
        super(ProWLS, self).__init__(config)
        # Get state features and instances for Actor and Value function
        self.state_features = Basis.get_Basis(config=config)
        self.actor, self.atype, self.action_size = NS_utils.get_Policy(state_dim=self.state_features.feature_dim, config=config)

        self.memory = utils.TrajectoryBuffer(buffer_size=config.buffer_size, state_dim=self.state_dim,
                                             action_dim=self.action_size, atype=self.atype, config=config, dist_dim=1)
        self.extrapolator = WLS(max_len=config.buffer_size, delta=config.delta, basis_type=config.extrapolator_basis, k=config.fourier_k)

        self.modules = [('actor', self.actor), ('state_features', self.state_features)]
        self.counter = 0
        self.init()

    def reset(self):
        super(ProWLS, self).reset()
        self.memory.next()
        self.counter += 1
        self.gamma_t = 1

    def get_action(self, state):
        state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device)
        state = self.state_features.forward(state.view(1, -1))
        action, prob, dist = self.actor.get_action_w_prob_dist(state)

        # if self.config.debug:
        #     self.track_entropy(dist, action)

        return action, prob, dist

    def update(self, s1, a1, prob, r1, s2, done):
        # Batch episode history
        self.memory.add(s1, a1, prob, self.gamma_t * r1)
        self.gamma_t *= self.config.gamma

        if done and self.counter % self.config.delta == 0:
            try:
                self.optimize()
            except RuntimeError as error:
                print('Runtime Error: ', error)

    def optimize(self):
        if self.memory.size <= self.config.fourier_k:
            # If number of rows is less than number of features (columns), it wont have full column rank.
            return

        # Inner optimization loop
        for iter in range(self.config.max_inner):
            # IMP: This is important for it to work
            # Do not sub-sample(w or w/o replacement), like in ProOLS. The gradients are NOT the same in expectation.
            # Take ALL samples once.
            id, s, a, beta, r, mask = self.memory.get_all()            # B, BxHxD, BxHxA, BxH, BxH

            # Batch, horizon, dimension
            B, H, D = s.shape
            _, _, A = a.shape

            # create state features
            s_feature = self.state_features.forward(s.view(B*H, D))             # BxHxD -> (BxH)xd

            # Get action probabilities
            log_pi, dist_all = self.actor.get_logprob_dist(s_feature, a.view(B * H, -1))     # (BxH)xd, (BxH)xA
            log_pi = log_pi.view(B, H)                                                       # (BxH)x1 -> BxH

            log_pi[mask == 0] = 0  # For summing in log, equate these to zero
            rho_num = torch.exp(torch.sum(log_pi, dim=-1, keepdim=True))

            beta[mask == 0] = 1     # For taking raw product, equate these to 1
            rho_denom = torch.prod(beta, dim=-1, keepdim=True)


            rho = rho_num / rho_denom
            rho = torch.clamp(rho, 0, self.config.importance_clip)            # Clipped Importance sampling (Biased)

            # Create total return
            total_return = torch.sum(r, dim=-1, keepdim=True)                   # sum(BxH) -> Bx1

            # Note: forecasting for the mean performance of the next batch when learning actively
            forecast = self.extrapolator.forward(x=id, y=total_return, weights=rho, max_len=self.memory.size,
                                                 delta=self.config.delta)

            # Compute the final loss
            loss = - 1.0 * forecast

            # Discourage very deterministic policies.
            if self.config.entropy_lambda > 0:
                if self.config.cont_actions:
                    entropy = torch.sum(dist_all.entropy().view(B, H, -1).sum(dim=-1) * mask) / torch.sum(mask)  # (BxH)xA -> BxH
                else:
                    log_pi_all = dist_all.view(B, H, -1)
                    pi_all = torch.exp(log_pi_all)                                      # (BxH)xA -> BxHxA
                    entropy = torch.sum(torch.sum(pi_all * log_pi_all, dim=-1) * mask) / torch.sum(mask)

                loss = loss + self.config.entropy_lambda * entropy

            # Compute the total derivative and update the parameters.
            self.step(loss)


