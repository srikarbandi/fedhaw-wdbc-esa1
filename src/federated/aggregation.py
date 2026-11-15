"""Server-side aggregation rules (pure functions, no training).

`fedhaw_update` reproduces the server part of the original `train_fedhaw`
(external/FedHAW/functions/training.py, lines 104-122 at commit 6f56552)
statement by statement. Only the variable layout differs.
"""

from __future__ import annotations

import numpy as np
import torch


def softmax_vec(x):
    """Identical to the original (`np.exp(x) / np.sum(np.exp(x))`).

    A max-shifted softmax would be mathematically equivalent but would change
    the last floating-point bits. We keep the original form so the bit-exact
    test against the original repository holds. Lambdas stay O(1) here, so
    overflow is not a concern.
    """
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp)


def fedavg_aggregate(params_candidates, sizes):
    """theta = sum_k (N_k/N) theta_k  (original `train_fedavg`)."""
    pk = np.array(sizes) / sum(sizes)
    with torch.no_grad():
        g = sum(pk[k] * params_candidates[k] for k in range(len(params_candidates)))
    return g.detach().clone(), pk


def fedhaw_init(sizes):
    """Original initialisation (run_serverside.py): gamma = log(1.0) = 0, lambda_k = log(N_k/N).

    An empty client (N_k = 0) would get lambda_k = -inf, i.e. weight 0. The
    default partition settings never produce empty clients.
    """
    N = sum(sizes)
    with np.errstate(divide="ignore"):
        lambdat = [np.log(sizes[k] / N) for k in range(len(sizes))]
    return np.log(1.0), lambdat


def fedhaw_update(params_vector_pre, params_candidates, params_candidates_pre, gammat, lambdat,
                  loop, mu, lr_gamma, lr_lambda):
    """Original FedHAW server step.

    if loop > 0:                                    # round 0 skips the hypergradient
        a_pre  = softmax(lambda)                    # previous weights
        d      = theta^t - sum_k a_pre_k theta_k^t  # global minus provisional aggregate
        gamma  = gamma - lr_gamma * e^gamma / mu * <d, theta^t>
        lambda_k = lambda_k - lr_lambda * e^(2 gamma) * a_pre_k (1 - a_pre_k) / mu
                              * <d, theta_k^(t-1)>   # NEW gamma, PREVIOUS-round candidates
    a = softmax(lambda)
    theta^(t+1) = e^gamma * sum_k a_k theta_k^t

    Returns (theta^(t+1), gamma, lambda, a).
    """
    K = len(params_candidates)
    with torch.no_grad():
        if loop > 0:
            sm_lambdatpre = softmax_vec(lambdat)
            params_tmpsum = sum(sm_lambdatpre[node] * params_candidates[node] for node in range(K))
            tmpvec = params_vector_pre - params_tmpsum
            gammat = gammat - lr_gamma * np.exp(gammat) / mu * torch.sum(tmpvec * params_vector_pre).item()
            lambdat = [
                lambdat[node] - lr_lambda * np.exp(2 * gammat) * sm_lambdatpre[node] * (1 - sm_lambdatpre[node]) / mu
                * torch.sum(tmpvec * params_candidates_pre[node]).item()
                for node in range(K)
            ]
        sm_lambdat = softmax_vec(lambdat)
        params_vector_global = sum(sm_lambdat[node] * params_candidates[node] for node in range(K))
        params_vector_global = params_vector_global * np.exp(gammat)
    return params_vector_global.detach().clone(), gammat, lambdat, sm_lambdat
