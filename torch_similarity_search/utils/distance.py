"""Distance computation utilities."""

from typing import Literal

import torch
from torch import Tensor, nn


def _normalize(x: Tensor, dim: int = -1) -> Tensor:
    """L2 normalize along specified dimension."""
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp(min=1e-12)


class DistanceModule(nn.Module):
    """
    TorchScript-compatible distance metric module.

    All metrics return values where smaller = more similar,
    enabling consistent min-heap style top-k selection.

    Metrics:
        - "l2": Squared L2 distance (Euclidean)
        - "ip": Negated inner product (for pre-normalized vectors)
        - "cosine": Cosine distance (1 - cosine_similarity), auto-normalizes
    """

    def __init__(self, metric: Literal["l2", "ip", "cosine"] = "l2"):
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
        elif self._metric == "cosine":
            # Normalize and compute cosine distance = 1 - cosine_similarity
            x_norm = _normalize(x, dim=1)
            y_norm = _normalize(y, dim=1)
            return 1.0 - torch.mm(x_norm, y_norm.t())
        else:  # ip
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
        elif self._metric == "cosine":
            # Normalize and compute cosine distance
            q_norm = _normalize(queries, dim=1)
            c_norm = _normalize(candidates, dim=2)
            cos_sim = torch.bmm(q_norm.unsqueeze(1), c_norm.transpose(1, 2)).squeeze(1)
            return 1.0 - cos_sim
        else:  # ip
            return -torch.bmm(queries.unsqueeze(1), candidates.transpose(1, 2)).squeeze(
                1
            )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Alias for pairwise distance (nn.Module interface)."""
        return self.pairwise(x, y)
