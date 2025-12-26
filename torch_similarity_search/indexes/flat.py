"""Flat index implementation."""

from typing import Literal, Tuple

import torch
from torch import Tensor

from torch_similarity_search.indexes.base import BaseIndex


class FlatIndex(BaseIndex):
    """
    Flat index for exact nearest neighbor search (brute-force).

    This index compares queries against all vectors - no approximation.
    Best for small datasets or when exact results are required.

    Args:
        dim: Dimensionality of vectors
        metric: Distance metric ('l2', 'ip' for inner product, or 'cosine')
        k: Default number of neighbors for forward() (default: 10)
    """

    def __init__(
        self,
        dim: int,
        metric: Literal["l2", "ip", "cosine"] = "l2",
        k: int = 10,
    ):
        super().__init__(metric=metric)
        self._dim = dim
        self._k = k

        # All vectors stored contiguously: (ntotal, dim)
        self.register_buffer("vectors", torch.zeros(0, dim))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        if value < 1:
            raise ValueError("k must be at least 1")
        self._k = value

    def add(self, vectors: Tensor) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Vectors of shape (n, dim) or (dim,) for single vector
        """
        vectors, _ = self._normalize_vectors(vectors)

        if self.ntotal == 0:
            self.vectors = vectors
        else:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)

    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Exact k-nearest neighbor search (brute-force).

        Args:
            queries: Query vectors of shape (batch_size, dim) or (dim,) for single query
            k: Number of nearest neighbors to return

        Returns:
            distances: Shape (batch_size, k) - distances to nearest neighbors
            indices: Shape (batch_size, k) - indices of nearest neighbors
        """
        queries, squeeze_output = self._normalize_vectors(queries, "Query vectors")
        batch_size = queries.shape[0]
        device = queries.device

        if self.ntotal == 0:
            # Empty index
            distances = torch.full((batch_size, k), float("inf"), device=device)
            indices = torch.full((batch_size, k), -1, dtype=torch.int, device=device)
            if squeeze_output:
                return distances.squeeze(0), indices.squeeze(0)
            return distances, indices

        # Compute distances to all vectors
        distances = self.distance.pairwise(queries, self.vectors)

        # Select top-k
        actual_k = min(k, self.ntotal)
        topk_dists, topk_indices = distances.topk(actual_k, dim=1, largest=False)

        # Convert to int32 for consistency
        topk_indices = topk_indices.int()

        # Pad if needed
        if actual_k < k:
            pad_dists = torch.full(
                (batch_size, k - actual_k), float("inf"), device=device
            )
            pad_indices = torch.full(
                (batch_size, k - actual_k), -1, dtype=torch.int, device=device
            )
            topk_dists = torch.cat([topk_dists, pad_dists], dim=1)
            topk_indices = torch.cat([topk_indices, pad_indices], dim=1)

        if squeeze_output:
            topk_dists = topk_dists.squeeze(0)
            topk_indices = topk_indices.squeeze(0)

        return topk_dists, topk_indices
