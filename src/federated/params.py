"""Model <-> flat parameter-vector utilities.

The original code flattens and restores parameters with
torch.nn.utils.parameters_to_vector / vector_to_parameters, which follow the
fixed order of model.parameters(). All hospitals build the same MLP class, so
every vector has the same layout and length.
"""

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def get_flat_params(model: torch.nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_flat_params(model: torch.nn.Module, vec: torch.Tensor) -> None:
    vector_to_parameters(vec.detach().clone(), model.parameters())


def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
