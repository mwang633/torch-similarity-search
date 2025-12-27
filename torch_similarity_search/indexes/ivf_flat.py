"""IVFFlat index implementation."""

from typing import Literal, Tuple

import torch
from torch import Tensor

from torch_similarity_search.indexes.base import BaseIndex


class IVFFlatIndex(BaseIndex):
    """
    Inverted File Flat index for approximate nearest neighbor search.

    This index partitions vectors into clusters (Voronoi cells) using centroids.
    During search, only vectors in the nearest clusters are compared.

    Args:
        dim: Dimensionality of vectors
        nlist: Number of clusters/centroids
        metric: Distance metric ('l2', 'ip' for inner product, or 'cosine')
        nprobe: Number of clusters to search (default: 20)
        k: Default number of neighbors for forward() (default: 10)
    """

    def __init__(
        self,
        dim: int,
        nlist: int,
        metric: Literal["l2", "ip", "cosine"] = "l2",
        nprobe: int = 20,
        k: int = 10,
    ):
        super().__init__(metric=metric)
        self._dim = dim
        self._nlist = nlist
        self._nprobe = nprobe
        self._k = k

        # Cluster centroids: (nlist, dim)
        self.register_buffer("centroids", torch.zeros(nlist, dim))

        # All vectors stored contiguously: (ntotal, dim)
        self.register_buffer("vectors", torch.zeros(0, dim))

        # Cluster assignment for each vector: (ntotal,)
        # Using int32 for better GPU performance (sufficient for up to 2B vectors)
        self.register_buffer("assignments", torch.zeros(0, dtype=torch.int))

        # Precomputed list boundaries for efficient indexing
        # list_offsets[i] = start index of cluster i in sorted order
        # list_sizes[i] = number of vectors in cluster i
        self.register_buffer("list_offsets", torch.zeros(nlist, dtype=torch.int))
        self.register_buffer("list_sizes", torch.zeros(nlist, dtype=torch.int))

        # Original indices (for returning correct IDs after sorting by cluster)
        # Using int32 for GPU efficiency (supports up to 2B vectors)
        self.register_buffer("indices", torch.zeros(0, dtype=torch.int))

        self._is_trained = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return self.vectors.shape[0]

    @property
    def nlist(self) -> int:
        return self._nlist

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value < 1 or value > self._nlist:
            raise ValueError(f"nprobe must be between 1 and {self._nlist}")
        self._nprobe = value

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        if value < 1:
            raise ValueError("k must be at least 1")
        self._k = value

    def train(self, vectors: Tensor) -> None:
        """
        Train the index by computing centroids via k-means.

        Args:
            vectors: Training vectors of shape (n, dim)
        """
        if vectors.dim() != 2:
            raise ValueError(
                f"Training vectors must be 2D (n, dim), got {vectors.dim()}D"
            )

        n, dim = vectors.shape
        if dim != self._dim:
            raise ValueError(
                f"Training vectors dimension {dim} does not match index dimension {self._dim}"
            )
        if n < self._nlist:
            raise ValueError(f"Need at least {self._nlist} vectors to train, got {n}")

        # Simple k-means initialization: random selection
        perm = torch.randperm(n, device=vectors.device)[: self._nlist]
        centroids = vectors[perm].clone()

        # K-means iterations (fully vectorized)
        max_iters = 20
        for _ in range(max_iters):
            # Assign vectors to nearest centroid
            dists = self.distance.pairwise(vectors, centroids)
            assignments = dists.argmin(dim=1)

            # Update centroids using scatter_add (vectorized)
            # Sum vectors per cluster
            new_centroids = torch.zeros_like(centroids)
            new_centroids.scatter_add_(
                0, assignments.unsqueeze(1).expand(-1, dim), vectors
            )

            # Count vectors per cluster
            counts = torch.zeros(self._nlist, device=vectors.device)
            counts.scatter_add_(0, assignments, torch.ones(n, device=vectors.device))

            # Compute mean (avoid division by zero for empty clusters)
            non_empty = counts > 0
            new_centroids[non_empty] /= counts[non_empty].unsqueeze(1)

            # Reinitialize empty clusters with random vectors
            empty_mask = ~non_empty
            if empty_mask.any():
                n_empty = int(empty_mask.sum().item())
                random_idx = torch.randint(n, (n_empty,), device=vectors.device)
                new_centroids[empty_mask] = vectors[random_idx]

            # Check convergence
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self.centroids = centroids
        self._is_trained = True

    def add(self, vectors: Tensor) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Vectors of shape (n, dim) or (dim,) for single vector
        """
        if not self._is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        vectors, _ = self._normalize_vectors(vectors)
        n = vectors.shape[0]
        device = vectors.device

        # Compute cluster assignments (int32 for GPU efficiency)
        dists = self.distance.pairwise(vectors, self.centroids)
        new_assignments = dists.argmin(dim=1).int()

        # Append to existing data
        old_ntotal = self.ntotal
        new_indices = torch.arange(
            old_ntotal, old_ntotal + n, dtype=torch.int, device=device
        )

        if old_ntotal == 0:
            self.vectors = vectors
            self.assignments = new_assignments
            self.indices = new_indices
        else:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
            self.assignments = torch.cat([self.assignments, new_assignments], dim=0)
            self.indices = torch.cat([self.indices, new_indices], dim=0)

        # Rebuild sorted structure for efficient search
        self._rebuild_lists()

    def _rebuild_lists(self) -> None:
        """Rebuild inverted list structure after adding vectors."""
        if self.ntotal == 0:
            return

        device = self.vectors.device

        # Sort by cluster assignment
        sorted_order = torch.argsort(self.assignments)
        self.vectors = self.vectors[sorted_order]
        self.indices = self.indices[sorted_order]
        self.assignments = self.assignments[sorted_order]

        # Compute list sizes and offsets (int32 for GPU efficiency)
        # Use bincount for O(n) instead of O(nlist * n) loop
        list_sizes = torch.bincount(self.assignments, minlength=self._nlist).to(
            dtype=torch.int, device=device
        )

        list_offsets = torch.zeros(self._nlist, dtype=torch.int, device=device)
        list_offsets[1:] = torch.cumsum(list_sizes[:-1], dim=0)

        self.list_sizes = list_sizes
        self.list_offsets = list_offsets

    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Batched k-nearest neighbor search (fully vectorized for GPU efficiency).

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

        # Find nearest centroids for each query: (batch_size, nprobe)
        centroid_dists = self.distance.pairwise(queries, self.centroids)
        _, probe_indices = centroid_dists.topk(self._nprobe, dim=1, largest=False)

        # Get list sizes and offsets for probed clusters
        # Shape: (batch_size, nprobe)
        probed_sizes = self.list_sizes[probe_indices]  # (batch_size, nprobe)
        probed_offsets = self.list_offsets[probe_indices]  # (batch_size, nprobe)

        # Find max list size across all probes (for padding)
        max_list_size = int(probed_sizes.max().item())

        if max_list_size == 0:
            # No vectors in any probed cluster
            all_distances = torch.full((batch_size, k), float("inf"), device=device)
            all_indices = torch.full(
                (batch_size, k), -1, dtype=torch.int, device=device
            )
            if squeeze_output:
                return all_distances.squeeze(0), all_indices.squeeze(0)
            return all_distances, all_indices

        # Build a 3D tensor of candidate indices: (batch_size, nprobe, max_list_size)
        # Each entry [b, p, i] = global vector index for query b, probe p, local index i
        # Use broadcasting: offset[b,p] + arange(max_list_size)
        local_indices = torch.arange(max_list_size, device=device)  # (max_list_size,)
        # (batch_size, nprobe, max_list_size)
        global_indices = probed_offsets.unsqueeze(2) + local_indices.view(1, 1, -1)

        # Mask for valid entries: local_idx < list_size[b, p]
        valid_mask_3d = local_indices.view(1, 1, -1) < probed_sizes.unsqueeze(
            2
        )  # (batch_size, nprobe, max_list_size)

        # Clamp indices for safe gathering (invalid will be masked out)
        safe_global_indices = global_indices.clamp(min=0, max=max(self.ntotal - 1, 0))

        # Gather vectors: (batch_size, nprobe, max_list_size, dim)
        candidate_vectors = self.vectors[safe_global_indices]

        # Gather original indices: (batch_size, nprobe, max_list_size)
        candidate_orig_idx = self.indices[safe_global_indices]

        # Reshape for batch distance computation
        # (batch_size, nprobe * max_list_size, dim)
        n_candidates = self._nprobe * max_list_size
        candidate_vectors_flat = candidate_vectors.view(batch_size, n_candidates, -1)
        candidate_orig_idx_flat = candidate_orig_idx.view(batch_size, n_candidates)
        valid_mask_flat = valid_mask_3d.view(batch_size, n_candidates)

        # Compute distances: (batch_size, n_candidates)
        distances = self.distance.batched(queries, candidate_vectors_flat)

        # Mask invalid candidates with infinity
        distances = distances.masked_fill(~valid_mask_flat, float("inf"))

        # Select top-k
        actual_k = min(k, n_candidates)
        topk_dists, topk_local = distances.topk(actual_k, dim=1, largest=False)
        topk_indices = candidate_orig_idx_flat.gather(1, topk_local)

        # Pad to k if needed
        if actual_k < k:
            pad_dists = torch.full(
                (batch_size, k - actual_k), float("inf"), device=device
            )
            pad_indices = torch.full(
                (batch_size, k - actual_k), -1, dtype=torch.int, device=device
            )
            topk_dists = torch.cat([topk_dists, pad_dists], dim=1)
            topk_indices = torch.cat([topk_indices, pad_indices], dim=1)

        # Mark invalid results (from padding) with -1
        topk_indices = topk_indices.masked_fill(topk_dists == float("inf"), -1)

        if squeeze_output:
            topk_dists = topk_dists.squeeze(0)
            topk_indices = topk_indices.squeeze(0)

        return topk_dists, topk_indices
