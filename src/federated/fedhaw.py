"""Original FedHAW (Nakai-Kasai & Wadayama, arXiv:2605.00458).

The server keeps three pieces of state across rounds, as `run_serverside.py`
does: gamma, lambda, and the previous round's client parameter vectors
(`params_candidates_pre`). They are set once in `reset` and never re-initialised
during training.
"""

from .aggregation import fedhaw_init, fedhaw_update, softmax_vec


class FedHAW:
    name = "fedhaw"
    mu_prox = 0.0

    def __init__(self, lr_gamma: float, lr_lambda: float):
        self.lr_gamma = float(lr_gamma)
        self.lr_lambda = float(lr_lambda)

    def reset(self, sizes):
        self.sizes = list(sizes)
        self.gammat, self.lambdat = fedhaw_init(self.sizes)
        self.params_candidates_pre = None

    def aggregate(self, params_vector_pre, params_candidates, round_idx, lr):
        """`lr` is the local learning rate mu, which appears in the hypergradients."""
        g, self.gammat, self.lambdat, w = fedhaw_update(
            params_vector_pre, params_candidates, self.params_candidates_pre,
            self.gammat, self.lambdat, round_idx, lr, self.lr_gamma, self.lr_lambda)
        self.params_candidates_pre = params_candidates          # kept for the next round
        return g, {"weights": [float(v) for v in w], "gamma": float(self.gammat),
                   "lambdas": [float(v) for v in self.lambdat]}

    @property
    def weights(self):
        return softmax_vec(self.lambdat)
