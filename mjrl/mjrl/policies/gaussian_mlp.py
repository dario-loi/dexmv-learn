import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tpi.core.config import cfg


class MLP:
    def __init__(
        self, env_spec, hidden_sizes=(64, 64), min_log_std=-3, init_log_std=0, seed=None
    ):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim  # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        self.model = MuNet(self.n, self.m, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
            param.data = 1e-2 * param.data
        if cfg.POLICY_LEARN_LOG_STD:
            self.log_std = Variable(
                torch.ones(self.m) * init_log_std, requires_grad=True
            )
            self.trainable_params = list(self.model.parameters()) + [self.log_std]
        else:
            self.log_std = Variable(
                torch.ones(self.m) * init_log_std, requires_grad=False
            )
            self.trainable_params = list(self.model.parameters())

        # Old Policy network
        # ------------------------
        self.old_model = MuNet(self.n, self.m, hidden_sizes)
        if cfg.POLICY_LEARN_LOG_STD:
            self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        else:
            self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
            self.old_params = list(self.old_model.parameters())
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        self.is_rollout = True

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate(
            [p.contiguous().view(-1).data.numpy() for p in self.trainable_params]
        )
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx : current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = torch.clamp(
                self.trainable_params[-1], self.min_log_std
            ).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx : current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = torch.clamp(
                self.old_params[-1], self.min_log_std
            ).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        if type(observation) == np.ndarray:
            observation = (
                torch.from_numpy(observation).float().to(self.model.fc0.weight.device)
            )

        std_val = torch.from_numpy(np.float32(self.log_std_val)).to(observation.device)

        mean = self.model(observation)
        noise = torch.exp(std_val) * torch.randn(self.m).to(observation.device)
        action = mean + noise

        np_mean = mean.detach().cpu().numpy()
        if self.is_rollout:
            return [
                action.detach().cpu().numpy(),
                {
                    "mean": np_mean,
                    "log_std": self.log_std_val,
                    "evaluation": np_mean,
                },
            ]

        return [
            action,
            {"mean": np_mean, "log_std": self.log_std_val, "evaluation": np_mean},
        ]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std

        log_std = log_std.to(observations.device)
        action = actions[0]
        mean = model(observations)

        zs = (action - mean) / torch.exp(log_std)
        LL = (
            -0.5 * torch.sum(zs**2, dim=1)
            + -torch.sum(log_std)
            + -0.5 * self.m * np.log(2 * np.pi)
        )
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std**2 - new_std**2
        Dr = 2 * new_std**2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl)


class MuNet(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes=(64, 64),
        in_shift=None,
        in_scale=None,
        out_shift=None,
        out_scale=None,
    ):
        super(MuNet, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        self.fc0 = nn.Linear(obs_dim, hidden_sizes[0])
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], act_dim)

    def set_transformations(
        self, in_shift=None, in_scale=None, out_shift=None, out_scale=None
    ):
        # store native scales that can be used for resets
        self.transformations = dict(
            in_shift=in_shift,
            in_scale=in_scale,
            out_shift=out_shift,
            out_scale=out_scale,
        )
        self.in_shift = (
            torch.from_numpy(np.float32(in_shift))
            if in_shift is not None
            else torch.zeros(self.obs_dim)
        )
        self.in_scale = (
            torch.from_numpy(np.float32(in_scale))
            if in_scale is not None
            else torch.ones(self.obs_dim)
        )
        self.out_shift = (
            torch.from_numpy(np.float32(out_shift))
            if out_shift is not None
            else torch.zeros(self.act_dim)
        )
        self.out_scale = (
            torch.from_numpy(np.float32(out_scale))
            if out_scale is not None
            else torch.ones(self.act_dim)
        )
        self.in_shift = Variable(self.in_shift, requires_grad=False)
        self.in_scale = Variable(self.in_scale, requires_grad=False)
        self.out_shift = Variable(self.out_shift, requires_grad=False)
        self.out_scale = Variable(self.out_scale, requires_grad=False)

    def forward(self, x):
        self.in_shift = self.in_shift.to(x.device)
        self.in_scale = self.in_scale.to(x.device)
        self.out_shift = self.out_shift.to(x.device)
        self.out_scale = self.out_scale.to(x.device)
        self.to(x.device)
        out = (x - self.in_shift) / (self.in_scale + 1e-8)
        out = torch.tanh(self.fc0(out))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        out = out * self.out_scale + self.out_shift
        return out
