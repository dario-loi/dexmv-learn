import logging
from re import match

# from typing import override

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

# nn modules
from torch import nn
from torch.nn import functional as F



def dump_tensors(only_gpu=True):
    print("Tensor objects dump")
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if only_gpu and obj.device.type == "cpu":
                    continue
                print(type(obj), obj.size(), obj.device)
        except:
            pass
    print("Memory summary")
    print(torch.cuda.memory_summary())


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim, device="cpu"):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.device = device
        self.fc1.to(device)
        self.fc2.to(device)
        self.fc3.to(device)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)

    def forward(self, x):
        x.to(self.device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return F.leaky_relu(self.fc3(x))

    def get_q(self, obs, act):
        obs = obs.to(self.device)
        act = act.to(self.device)
        x = torch.cat([obs, act], dim=-1)
        return self.forward(x).squeeze()


class IQLearn(batch_reinforce.BatchREINFORCE):
    def __init__(
        self,
        env,
        policy,
        baseline,
        alpha,  # lr
        demo_paths=None,
        gamma=0.99,
        seed=None,
        save_logs=False,
    ):
        super().__init__(env, policy, baseline, alpha, seed, save_logs)
        self.Qnet = QNet(env.observation_dim, env.action_dim, device="cuda")
        self.gamma = gamma
        self.demo_paths = demo_paths
        self.Q_opt = torch.optim.Adam(self.Qnet.parameters(), lr=alpha)
        self.policy_opt = torch.optim.Adam(
            self.policy.model.parameters(), lr=alpha
        )  # maximize=True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # @override
    def train_from_paths(self, paths):
        # Sampled trajectories
        act_t = np.concatenate([path["actions"][:-1] for path in paths])
        act_t = torch.from_numpy(act_t).float().to(self.device)

        # Retrieve the (st, at, st1) triplets
        obs_t = np.concatenate([path["observations"][:-1] for path in paths])
        obs_t1 = np.concatenate([path["observations"][1:] for path in paths])
        obs_t = torch.from_numpy(obs_t).float().to(self.device)
        obs_t1 = torch.from_numpy(obs_t1).float().to(self.device)

        # Expert trajectories
        demo_obs_t = np.concatenate(
            [path["observations"][:-1] for path in self.demo_paths]
        )
        demo_obs_t1 = np.concatenate(
            [path["observations"][1:] for path in self.demo_paths]
        )

        demo_act_t = np.concatenate([path["actions"][:-1] for path in self.demo_paths])

        demo_obs_t = torch.from_numpy(demo_obs_t).float().to(self.device)
        demo_obs_t1 = torch.from_numpy(demo_obs_t1).float().to(self.device)
        demo_act_t = torch.from_numpy(demo_act_t).float().to(self.device)

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
        self.iq_update(obs_t, obs_t1, act_t, demo_obs_t, demo_obs_t1)
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
        obs_t,
        obs_t1,
        act_t,
        demo_obs_t,
        demo_obs_t1,
    ):
        # define the loss J
        # case of forward KL divergence
        # phi(x) = -exp-(x+1)

        # building J
        # 1)sampled part

        # calculate Q(s,a)
        self.policy.is_rollout = False
        self.policy.model.requires_grad = False
        self.Qnet.requires_grad = True

        Q = self.Qnet.get_q(obs_t, act_t)

        # calculate V^pi(s')
        print("obs_t1", obs_t1.shape)
        action = self.policy.get_action(obs_t1)
        action = action[0]  # unpack the action from the info dict
        log_prob = self.policy.log_likelihood(obs_t1, action)
        next_Q = self.Qnet.get_q(obs_t1, action)
        Vs_next = next_Q - log_prob

        # argument of phi
        update = Q - self.gamma * (Vs_next).mean()

        # calculate phi
        phi = update / (1 - update)

        # 2)expert part
        # calculate V^pi(s_0)
        action = self.policy.get_action(demo_obs_t1)[0]
        log_prob = self.policy.log_likelihood(demo_obs_t1, action)
        current_Q = self.Qnet.get_q(demo_obs_t, action)
        V = current_Q - log_prob

        # calculate J
        J = -1 * (phi.mean() - (1 - self.gamma) * V.mean())

        J.backward()
        self.Q_opt.step()
        self.Q_opt.zero_grad()

        self.policy.model.requires_grad = True
        self.Qnet.requires_grad = False

        Q = self.Qnet.get_q(obs_t, act_t)

        action = self.policy.get_action(obs_t)[0]
        log_prob = self.policy.log_likelihood(obs_t, action)

        # calculate V^pi(s)
        V = -(Q - log_prob).mean()  # THE MINUS HAS BEEN ADDED BECAUSE TO
        # MAXIMIZE THE ORIGINAL OBJECTIVE

        V.backward()
        self.policy_opt.step()
        self.policy_opt.zero_grad()
