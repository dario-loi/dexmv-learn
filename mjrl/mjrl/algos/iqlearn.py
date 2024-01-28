import logging
from re import match

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

# utility functions
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog


class IQLearn:
    # we do not inherit since we want total control over the training loop
    # however we will still mock a BatchREINFORCE-like API.

    def __init__(
        self,
        env,
        policy,
        baseline,
        alpha,  # lr
        seed=None,
        save_logs=False,
    ):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = alpha
        self.seed = seed
        self.save_logs = save_logs
        self.running_score = None
        if save_logs:
            self.logger = DataLog()

    def CPI_surrogate(self, observations, actions, advantages):
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)  # type: ignore
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        LR = self.policy.likelihood_ratio(new_dist_info, old_dist_info)
        surr = torch.mean(LR * adv_var)
        return surr

    def kl_old_new(self, observations, actions):
        old_dist_info = self.policy.old_dist_info(observations, actions)
        new_dist_info = self.policy.new_dist_info(observations, actions)
        mean_kl = self.policy.mean_kl(new_dist_info, old_dist_info)
        return mean_kl

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = self.CPI_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(cpi_surr, self.policy.trainable_params)
        vpg_grad = np.concatenate(
            [g.contiguous().view(-1).data.numpy() for g in vpg_grad]
        )
        return vpg_grad

    def train_step(
        self,
        N,
        sample_mode="trajectories",
        env_name=None,
        T=1e6,
        gamma=0.995,
        gae_lambda=0.98,
        num_cpu="max",
    ):
        env_name = self.env.env_id if env_name is None else env_name

        assert (
            sample_mode == "trajectories" or sample_mode == "samples"
        ), "invalid sample mode"

        ts = timer.time()

        if sample_mode == "trajectories":
            paths = trajectory_sampler.sample_paths_parallel(
                N, self.policy, T, env_name, self.seed, num_cpu
            )
        else:
            paths = batch_sampler.sample_paths(
                N,
                self.policy,
                T=T,
                env_name=env_name,
                pegasus_seed=self.seed,
                num_cpu=num_cpu,
            )

        if self.save_logs:
            self.logger.log_kv("sampling_time", timer.time() - ts)

        # the algo then obtains these "paths" that are some kind of data representation
        # of trajectories, they are probably stored on disk(?) to handle large datasets
        # it calls train_from_paths(paths) to actually train the policy.

    def train_from_paths(self, paths):
        # get the paths and process them
        pass

    def log_rollout_statistics(self, paths):
        pass
