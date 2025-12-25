"""IVFFlat index implementation."""

from typing import Literal, Tuple

import torch
from torch import Tensor, nn

from torch_similarity_search.indexes.base import BaseIndex
from torch_similarity_search.utils.distance import l2_distance, inner_product


class IVFFlatIndex(BaseIndex):
    """
    Inverted File Flat index for approximate nearest neighbor search.

    This index partitions vectors into clusters (Voronoi cells) using centroids.
    During search, only vectors in the nearest clusters are compared.

    Args:
        dim: Dimensionality of vectors
        nlist: Number of clusters/centroids
        metric: Distance metric ('l2' or 'ip' for inner product)
        nprobe: Number of clusters to search (default: 1)
    """

    def __init__(
        self,
        dim: int,
        nlist: int,
        metric: Literal["l2", "ip"] = "l2",
        nprobe: int = 1,
    ):
        super().__init__()
        self._dim = dim
        self._nlist = nlist
        self._metric = metric
        self._nprobe = nprobe

        # Cluster centroids: (nlist, dim)
        self.register_buffer("centroids", torch.zeros(nlist, dim))

        # All vectors stored contiguously: (ntotal, dim)
        self.register_buffer("vectors", torch.zeros(0, dim))

        # Cluster assignment for each vector: (ntotal,)
        self.register_buffer("assignments", torch.zeros(0, dtype=torch.long))

        # Precomputed list boundaries for efficient indexing
        # list_offsets[i] = start index of cluster i in sorted order
        # list_sizes[i] = number of vectors in cluster i
        self.register_buffer("list_offsets", torch.zeros(nlist, dtype=torch.long))
        self.register_buffer("list_sizes", torch.zeros(nlist, dtype=torch.long))

        # Original indices (for returning correct IDs after sorting by cluster)
        self.register_buffer("indices", torch.zeros(0, dtype=torch.long))

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

    def _compute_distances(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute distances based on metric."""
        if self._metric == "l2":
            return l2_distance(x, y)
        else:
            return inner_product(x, y)

    def train(self, vectors: Tensor) -> None:
        """
        Train the index by computing centroids via k-means.

        Args:
            vectors: Training vectors of shape (n, dim)
        """
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)

        n = vectors.shape[0]
        if n < self._nlist:
            raise ValueError(f"Need at least {self._nlist} vectors to train, got {n}")

        # Simple k-means initialization: random selection
        perm = torch.randperm(n, device=vectors.device)[: self._nlist]
        centroids = vectors[perm].clone()

        # K-means iterations
        max_iters = 20
        for _ in range(max_iters):
            # Assign vectors to nearest centroid
            dists = self._compute_distances(vectors, centroids)
            assignments = dists.argmin(dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self._nlist, device=vectors.device)

            for i in range(self._nlist):
                mask = assignments == i
                if mask.any():
                    new_centroids[i] = vectors[mask].mean(dim=0)
                    counts[i] = mask.sum()
                else:
                    # Empty cluster: reinitialize with random vector
                    new_centroids[i] = vectors[torch.randint(n, (1,))]

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

        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)

        n = vectors.shape[0]
        device = vectors.device

        # Compute cluster assignments
        dists = self._compute_distances(vectors, self.centroids)
        new_assignments = dists.argmin(dim=1)

        # Append to existing data
        old_ntotal = self.ntotal
        new_indices = torch.arange(old_ntotal, old_ntotal + n, device=device)

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

        # Compute list sizes and offsets
        list_sizes = torch.zeros(self._nlist, dtype=torch.long, device=device)
        for i in range(self._nlist):
            list_sizes[i] = (self.assignments == i).sum()

        list_offsets = torch.zeros(self._nlist, dtype=torch.long, device=device)
        list_offsets[1:] = torch.cumsum(list_sizes[:-1], dim=0)

        self.list_sizes = list_sizes
        self.list_offsets = list_offsets

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
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = queries.shape[0]
        device = queries.device

        # Find nearest centroids for each query
        centroid_dists = self._compute_distances(queries, self.centroids)
        _, probe_indices = centroid_dists.topk(self._nprobe, dim=1, largest=False)

        # Initialize results with infinity distances
        all_distances = torch.full((batch_size, k), float("inf"), device=device)
        all_indices = torch.full((batch_size, k), -1, dtype=torch.long, device=device)

        # Search each query - using tensor indexing for TorchScript compatibility
        for q_idx in range(batch_size):
            query = queries[q_idx : q_idx + 1]  # (1, dim)
            candidate_distances: list[Tensor] = []
            candidate_indices: list[Tensor] = []

            # Probe each selected cluster
            for probe_idx in range(self._nprobe):
                cluster_id = int(probe_indices[q_idx, probe_idx])
                list_size = int(self.list_sizes[cluster_id])

                if list_size == 0:
                    continue

                offset = int(self.list_offsets[cluster_id])
                cluster_vectors = self.vectors[offset : offset + list_size]
                cluster_indices = self.indices[offset : offset + list_size]

                # Compute distances to vectors in this cluster
                dists = self._compute_distances(query, cluster_vectors).squeeze(0)
                candidate_distances.append(dists)
                candidate_indices.append(cluster_indices)

            if len(candidate_distances) > 0:
                # Concatenate all candidates
                all_cand_dists = torch.cat(candidate_distances)
                all_cand_indices = torch.cat(candidate_indices)

                # Select top-k
                num_candidates = all_cand_dists.shape[0]
                actual_k = min(k, num_candidates)

                topk_dists, topk_local = all_cand_dists.topk(actual_k, largest=False)
                topk_indices = all_cand_indices[topk_local]

                all_distances[q_idx, :actual_k] = topk_dists
                all_indices[q_idx, :actual_k] = topk_indices

        if squeeze_output:
            all_distances = all_distances.squeeze(0)
            all_indices = all_indices.squeeze(0)

        return all_distances, all_indices

    def forward(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """Alias for search() to support nn.Module interface."""
        return self.search(queries, k)
