"""Hospital client: local training.

`local_train` is a port of the original `_local_train`
(external/FedHAW/functions/training.py), with the same operation order:
DataLoader(shuffle=True) -> load global params -> fresh optimizer -> E epochs
of mini-batch updates -> return flattened params. The order matters for exact
RNG equivalence. With mu_prox > 0 it adds the FedProx proximal term
mu_prox/2 * ||w - w_global||^2, written as in the original `train_fedprox`.
"""

from __future__ import annotations

import time

import torch
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def make_optimizer(model, opt: str, lr: float):
    """Same choices as the original `_make_optimizer`."""
    if opt == "SGD":
        return optim.SGD(model.parameters(), lr=lr), None
    if opt == "SGDmomentum":
        o = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        return o, optim.lr_scheduler.LambdaLR(o, lr_lambda=lambda e: 0.99 ** e)
    if opt == "SGDdecay":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4), None
    if opt == "Adam":
        return optim.Adam(model.parameters(), lr=lr), None
    raise ValueError(f"unknown optimizer {opt}")


class Client:
    """One simulated hospital. Holds its own training data, which never leaves it."""

    def __init__(self, name: str, dataset: torch.utils.data.TensorDataset):
        self.name = name
        self.dataset = dataset

    @property
    def n(self) -> int:
        return len(self.dataset)

    def local_train(self, params_vector_pre, model, criterion, lr, E, batch_size, opt="SGD",
                    device="cpu", mu_prox: float = 0.0):
        """Returns (updated flat params, summed batch loss, number of batches, seconds)."""
        t0 = time.perf_counter()
        if self.n == 0:   # empty hospital: nothing to train on, returns the global model unchanged
            return params_vector_pre.detach().clone(), 0.0, 0, time.perf_counter() - t0
        return (*local_train(self.dataset, params_vector_pre, batch_size, E, lr, device, model,
                             criterion, opt, mu_prox), time.perf_counter() - t0)


def local_train(dataset, params_vector_pre, batch_size, E, mu, device, model, criterion, opt,
                mu_prox: float = 0.0):
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    local_model = model.to(device)
    local_model.train()
    vector_to_parameters(params_vector_pre.detach().clone(), local_model.parameters())
    optimizer, scheduler = make_optimizer(local_model, opt, mu)
    anchor = params_vector_pre.detach().clone().to(device)

    running_loss, n_batches = 0.0, 0
    for _ in range(E):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(local_model(inputs), targets)
            if mu_prox > 0:
                loss = loss + 0.5 * mu_prox * torch.norm(
                    parameters_to_vector(local_model.parameters()) - anchor, p=2) ** 2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
    return parameters_to_vector(local_model.parameters()).detach().clone(), running_loss, n_batches
