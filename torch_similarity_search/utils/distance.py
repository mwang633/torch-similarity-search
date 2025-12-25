"""Distance computation utilities."""

from typing import Literal

import torch
from torch import Tensor, nn


class DistanceModule(nn.Module):
    """
    TorchScript-compatible distance metric module.

    All metrics return values where smaller = more similar,
    enabling consistent min-heap style top-k selection.
    """

    def __init__(self, metric: Literal["l2", "ip"] = "l2"):
        super().__init__()
        self._metric = metric

    @property
    def name(self) -> str:
        return self._metric

    def pairwise(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute pairwise distances.

        Args:
            x: Query vectors of shape (n, dim)
            y: Database vectors of shape (m, dim)

        Returns:
            Distances of shape (n, m)
        """
        if self._metric == "l2":
            return torch.cdist(x, y, p=2.0).pow(2)
        else:
            return -torch.mm(x, y.t())

    def batched(self, queries: Tensor, candidates: Tensor) -> Tensor:
        """
        Compute batched distances.

        Args:
            queries: Shape (batch_size, dim)
            candidates: Shape (batch_size, n_candidates, dim)

        Returns:
            Distances of shape (batch_size, n_candidates)
        """
        if self._metric == "l2":
            q_norm_sq = (queries**2).sum(dim=1, keepdim=True)
            v_norm_sq = (candidates**2).sum(dim=2)
            qv_dot = torch.bmm(
                queries.unsqueeze(1), candidates.transpose(1, 2)
            ).squeeze(1)
            return q_norm_sq + v_norm_sq - 2 * qv_dot
        else:
            return -torch.bmm(
                queries.unsqueeze(1), candidates.transpose(1, 2)
            ).squeeze(1)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Alias for pairwise distance (nn.Module interface)."""
        return self.pairwise(x, y)
