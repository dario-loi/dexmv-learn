import numpy as np
import copy


class QuadraticBaseline:
    def __init__(self, env_spec, reg_coeff=1e-5, obs_dim=None):
        self.n = obs_dim if obs_dim is not None else env_spec.observation_dim
        self._reg_coeff = reg_coeff
        self._coeffs = None

    def _features(self, paths):
        o = np.concatenate([path["observations"] for path in paths])
        o = np.clip(o, -10, 10) / 10.0
        if o.ndim > 2:
            o = o.reshape(o.shape[0], -1)
        N, n = o.shape
        num_feat = int(
            n + n * (n + 1) / 2 + 1 + 4
        )  # linear + full quad (symmetric matrix) + bias (1.0) + time till pow 4
        feat_mat = np.ones((N, num_feat))  # memory allocation

        # linear features
        feat_mat[:, :n] = o

        # quadratic features
        k = n  # starting from this column in feat_mat
        for i in range(n):
            for j in range(i, n):
                feat_mat[:, k] = o[:, i] * o[:, j]  # element-wise product
                k += 1

        k = 0  # start from this row
        for i in range(len(paths)):
            l = len(paths[i]["rewards"])
            al = np.arange(l) / 1000.0
            for j in range(4):
                feat_mat[k : k + l, -4 + j] = al ** (j + 1)
            k += l

        return feat_mat

    def fit(self, paths, return_errors=False):
        # featmat = np.concatenate([self._features(path) for path in paths])
        featmat = self._features(paths)
        returns = np.concatenate([path["returns"] for path in paths])

        if return_errors:
            predictions = (
                featmat.dot(self._coeffs)
                if self._coeffs is not None
                else np.zeros(returns.shape)
            )
            errors = returns - predictions
            error_before = np.sum(errors**2) / np.sum(returns**2)

        reg_coeff = copy.deepcopy(self._reg_coeff)
        for _ in range(10):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns),
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

        if return_errors:
            predictions = featmat.dot(self._coeffs)
            errors = returns - predictions
            error_after = np.sum(errors**2) / np.sum(returns**2)
            return error_before, error_after

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features([path]).dot(self._coeffs)
