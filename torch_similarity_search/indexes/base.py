"""Abstract base class for similarity search indexes."""

from abc import abstractmethod
from typing import Literal, Tuple

from torch import Tensor, nn

from torch_similarity_search.utils.distance import DistanceModule


class BaseIndex(nn.Module):
    """Abstract base for all similarity search indexes."""

    def __init__(self, metric: Literal["l2", "ip", "cosine"] = "l2"):
        super().__init__()
        self.distance = DistanceModule(metric)

    def _normalize_vectors(
        self, vectors: Tensor, name: str = "Vectors"
    ) -> Tuple[Tensor, bool]:
        """
        Normalize input to 2D batch format and validate dimensions.

        Args:
            vectors: Input tensor of shape (n, dim) or (dim,)
            name: Name for error messages

        Returns:
            Tuple of (normalized 2D tensor, should_squeeze flag)
        """
        if vectors.dim() == 1:
            vec_dim = vectors.shape[0]
            vectors = vectors.unsqueeze(0)
            squeeze = True
        elif vectors.dim() == 2:
            vec_dim = vectors.shape[1]
            squeeze = False
        else:
            raise ValueError(f"{name} must be 1D or 2D, got {vectors.dim()}D")

        if vec_dim != self.dim:
            raise ValueError(
                f"{name} dimension {vec_dim} does not match index dimension {self.dim}"
            )

        return vectors, squeeze

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

    @property
    @abstractmethod
    def k(self) -> int:
        """Default k for forward() - number of neighbors to return."""
        pass

    def forward(self, queries: Tensor) -> Tuple[Tensor, Tensor]:
        """
        nn.Module interface for inference (uses configured k).

        For Triton/TorchScript deployment, k is fixed at export time.
        Use search(queries, k) for runtime-configurable k.
        """
        return self.search(queries, self.k)
