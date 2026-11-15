"""FedAvg (McMahan et al., 2017): a_k = N_k / N, theta = sum_k a_k theta_k."""

from .aggregation import fedavg_aggregate


class FedAvg:
    name = "fedavg"
    mu_prox = 0.0

    def reset(self, sizes):
        self.sizes = list(sizes)

    def aggregate(self, params_vector_pre, params_candidates, round_idx, lr):
        g, w = fedavg_aggregate(params_candidates, self.sizes)
        return g, {"weights": [float(v) for v in w]}
