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
        seed=None,
        save_logs=False,
    ):
        super().__init__(env, policy, baseline, alpha, seed, save_logs)

    @override
    def train_from_paths(self, paths):
        
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

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
        
        ts = timer.time()
        gradient = self.iq_update(observations, actions, advantages) # change if needed
        t_gLL += timer.time() - ts
        
        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + self.alpha * gradient
        
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)
        
        # Log information
        if self.save_logs:
            self.logger.log_kv("alpha", self.alpha)
            self.logger.log_kv("time_vpg", t_gLL)
            self.logger.log_kv("kl_dist", kl_dist)
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
    
    def iq_update(self, observations, actions, advantages):
        
        pass