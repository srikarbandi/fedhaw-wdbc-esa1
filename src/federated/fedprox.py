"""FedProx (Li et al., 2020).

Local objective: L_k(theta) + mu_prox/2 * ||theta - theta_global_previous||^2
(implemented in client.local_train). Aggregation: a_k = N_k / N.

Note: the original FedHAW repository's `train_fedprox` weights clients by the
number of mini-batches (len(train_loader)). We use N_k/N, as in the FedProx
paper and our FedAvg, so FedProx differs from FedAvg only in the proximal
term. With equal batch sizes the two weightings differ only through rounding
up of partial batches.
"""

from .aggregation import fedavg_aggregate


class FedProx:
    name = "fedprox"

    def __init__(self, mu_prox: float):
        self.mu_prox = float(mu_prox)

    def reset(self, sizes):
        self.sizes = list(sizes)

    def aggregate(self, params_vector_pre, params_candidates, round_idx, lr):
        g, w = fedavg_aggregate(params_candidates, self.sizes)
        return g, {"weights": [float(v) for v in w]}
