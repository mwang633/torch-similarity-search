"""Distance computation utilities."""

import torch
from torch import Tensor


def l2_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute L2 (Euclidean) squared distances between x and y.

    Args:
        x: Query vectors of shape (n, dim)
        y: Database vectors of shape (m, dim)

    Returns:
        Distances of shape (n, m)
    """
    return torch.cdist(x, y, p=2.0).pow(2)


def inner_product(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute inner product similarities between x and y.

    For similarity search, we return negative inner product so that
    smaller values indicate more similar vectors (consistent with L2).

    Args:
        x: Query vectors of shape (n, dim)
        y: Database vectors of shape (m, dim)

    Returns:
        Negative inner products of shape (n, m)
    """
    return -torch.mm(x, y.t())
