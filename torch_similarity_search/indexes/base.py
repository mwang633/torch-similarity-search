"""Abstract base class for similarity search indexes."""

from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor, nn


class BaseIndex(nn.Module):
    """Abstract base for all similarity search indexes."""

    @abstractmethod
    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Batched k-nearest neighbor search.

        Args:
            queries: Query vectors of shape (batch_size, dim) or (dim,) for single query
            k: Number of nearest neighbors to return

        Returns:
            distances: Shape (batch_size, k) - distances to nearest neighbors
            indices: Shape (batch_size, k) - indices of nearest neighbors
        """
        pass

    @abstractmethod
    def add(self, vectors: Tensor) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Vectors of shape (n, dim) or (dim,) for single vector
        """
        pass

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of indexed vectors."""
        pass

    def forward(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """Alias for search() to support nn.Module interface."""
        return self.search(queries, k)
