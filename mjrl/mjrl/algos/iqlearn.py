import logging
from re import match
from typing import override

logging.disable(logging.CRITICAL)
import numpy as np
import scipy as sp
import scipy.sparse.linalg as spLA
import copy
import time as timer
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical, Normal
import copy

# samplers
import mjrl.samplers.trajectory_sampler as trajectory_sampler
import mjrl.samplers.batch_sampler as batch_sampler
import mjrl.algos.batch_reinforce as batch_reinforce

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog


class IQLearn(batch_reinforce.BatchREINFORCE):
    def __init__(
        self,
        env,
        policy,
        baseline,
        alpha,  # lr
        gamma=0.99,
        seed=None,
        save_logs=False,
    ):
        super().__init__(env, policy, baseline, alpha, seed, save_logs)
        self.q_params = torch.randn(env.spec.observation_dim, env.spec.action_dim)
        self.gamma = gamma
        self.Q_opt = torch.optim.Adam(self.q_params, lr=alpha)
        self.policy_opt = torch.optim.Adam(
            self.policy.model.parameters(), lr=alpha, maximize=True
        )

    @override
    def train_from_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = (
            mean_return
            if self.running_score is None
            else 0.9 * self.running_score + 0.1 * mean_return
        )  # approx avg of last 10 iters
        if self.save_logs:
            self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Q-function update
        # --------------------------
        # Initialize Q-function parameters randomly

        ts = timer.time()
        self.iq_update(observations, actions)  # change if needed
        t_gLL += timer.time() - ts

        # Log information
        if self.save_logs:
            self.logger.log_kv("alpha", self.alpha)
            self.logger.log_kv("time_vpg", t_gLL)
            self.logger.log_kv("running_score", self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv("success_rate", success_rate)
                except:
                    pass

        return base_stats

    def iq_update(
        self,
        obs_samp,
        act_samp,
        rew_samp,
        next_obs_samp,
        obs_exp,
        act_exp,
        rew_exp,
        next_obs_exp,
    ):
        # define the loss J
        # case of forward KL divergence
        # phi(x) = -exp-(x+1)

        # building J
        # 1)expert part

        # calculate Q(s,a)
        self.policy.model.requires_grad = False
        self.q_params.requires_grad = True

        Q = self.q_params[obs_samp, act_samp]

        # calculate V^pi(s')
        action = self.policy.get_action(next_obs_samp)
        log_prob = self.policy.log_likelihood(next_obs_samp, action)
        next_Q = self.q_params[next_obs_samp, action]
        Vs_next = next_Q - log_prob

        # argument of phi
        update = Q - self.gamma * (Vs_next).mean()

        # calculate phi
        # with torch.no_grad():
        #    phi = 1/(1+ reward)**2
        phi = update / (1 - update)

        # 2)observation part
        # calculate V^pi(s_0)
        action = self.policy.get_action(next_obs_exp)
        log_prob = self.policy.log_likelihood(next_obs_exp, action)
        current_Q = self.q_params[obs_exp, action]
        V = current_Q - log_prob

        # calculate J
        J = -1 * (phi.mean() - (1 - self.gamma) * V.mean())

        J.backward()
        self.Q_opt.step()
        self.Q_opt.zero_grad()

        self.policy.model.requires_grad = True
        self.q_params.requires_grad = False

        Q = self.q_params[obs_samp, act_samp]

        action = self.policy.get_action(obs_samp)
        log_prob = self.policy.log_likelihood(obs_samp, action)

        # calculate V^pi(s)
        V = (Q - log_prob).mean()

        V.backward()
        self.policy_opt.step()
        self.policy_opt.zero_grad()
