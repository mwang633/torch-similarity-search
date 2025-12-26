"""IVFPQ index implementation."""

from typing import Literal, Tuple

import torch
from torch import Tensor

from torch_similarity_search.indexes.base import BaseIndex


class IVFPQIndex(BaseIndex):
    """
    Inverted File with Product Quantization index for approximate nearest neighbor search.

    This index combines IVF (coarse quantization) with PQ (fine quantization):
    1. Vectors are assigned to clusters using centroids (like IVFFlat)
    2. For L2: Residuals (vector - centroid) are compressed using Product Quantization
       For IP: Vectors are quantized directly (no residuals)
    3. During search, approximate distances are computed using precomputed tables

    Product Quantization splits each vector into M subvectors and quantizes each
    independently using ksub centroids (codebook). This reduces storage from
    dim * 4 bytes to M bytes per vector (when ksub=256).

    Args:
        dim: Dimensionality of vectors (must be divisible by M)
        nlist: Number of IVF clusters/centroids
        M: Number of PQ subquantizers (subvectors)
        nbits: Bits per subquantizer code (default: 8, meaning ksub=256)
        metric: Distance metric ('l2' or 'ip')
        nprobe: Number of clusters to search (default: 10)
        k: Default number of neighbors for forward() (default: 10)
    """

    def __init__(
        self,
        dim: int,
        nlist: int,
        M: int,
        nbits: int = 8,
        metric: Literal["l2", "ip"] = "l2",
        nprobe: int = 10,
        k: int = 10,
    ):
        if metric not in ("l2", "ip"):
            raise ValueError(f"Unsupported metric: {metric}. IVFPQ supports 'l2' or 'ip'.")

        if nbits > 8:
            raise ValueError(f"nbits must be <= 8 (got {nbits}). Codes are stored as uint8.")

        super().__init__(metric=metric)

        if dim % M != 0:
            raise ValueError(f"dim ({dim}) must be divisible by M ({M})")

        self._dim = dim
        self._nlist = nlist
        self._M = M
        self._nbits = nbits
        self._ksub = 2**nbits  # Number of centroids per subquantizer
        self._dsub = dim // M  # Dimension of each subvector
        self._nprobe = nprobe
        self._k = k

        # IVF centroids: (nlist, dim)
        self.register_buffer("centroids", torch.zeros(nlist, dim))

        # PQ codebooks: (M, ksub, dsub) - M codebooks, each with ksub centroids of dim dsub
        self.register_buffer("pq_centroids", torch.zeros(M, self._ksub, self._dsub))

        # Compressed codes: (ntotal, M) stored as uint8 (when nbits=8)
        # Each code[i, m] is an index into pq_centroids[m]
        self.register_buffer("codes", torch.zeros(0, M, dtype=torch.uint8))

        # Cluster assignments: (ntotal,)
        self.register_buffer("assignments", torch.zeros(0, dtype=torch.int))

        # Precomputed list boundaries
        self.register_buffer("list_offsets", torch.zeros(nlist, dtype=torch.int))
        self.register_buffer("list_sizes", torch.zeros(nlist, dtype=torch.int))

        # Original indices
        self.register_buffer("indices", torch.zeros(0, dtype=torch.int))

        self._is_trained = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return self.codes.shape[0]

    @property
    def nlist(self) -> int:
        return self._nlist

    @property
    def M(self) -> int:
        return self._M

    @property
    def nbits(self) -> int:
        return self._nbits

    @property
    def ksub(self) -> int:
        return self._ksub

    @property
    def dsub(self) -> int:
        return self._dsub

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
        Train the index (IVF centroids + PQ codebooks).

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

        min_vectors = max(self._nlist, self._ksub)
        if n < min_vectors:
            raise ValueError(f"Need at least {min_vectors} vectors to train, got {n}")

        device = vectors.device

        # Step 1: Train IVF centroids via k-means
        self._train_ivf(vectors)

        # Step 2: Prepare vectors for PQ training
        # For L2: use residuals (vector - centroid)
        # For IP: use vectors directly (no residuals)
        dists = self.distance.pairwise(vectors, self.centroids)
        assignments = dists.argmin(dim=1)

        if self.distance.name == "l2":
            # Compute residuals: vector - assigned_centroid
            pq_training_vectors = vectors - self.centroids[assignments]
        else:
            # For IP, quantize vectors directly
            pq_training_vectors = vectors

        # Step 3: Train PQ codebooks
        self._train_pq(pq_training_vectors, device)

        self._is_trained = True

    def _train_ivf(self, vectors: Tensor) -> None:
        """Train IVF centroids using k-means."""
        n = vectors.shape[0]
        dim = self._dim
        device = vectors.device

        # Random initialization
        perm = torch.randperm(n, device=device)[: self._nlist]
        centroids = vectors[perm].clone()

        # K-means iterations
        max_iters = 20
        for _ in range(max_iters):
            dists = self.distance.pairwise(vectors, centroids)
            assignments = dists.argmin(dim=1)

            new_centroids = torch.zeros_like(centroids)
            new_centroids.scatter_add_(
                0, assignments.unsqueeze(1).expand(-1, dim), vectors
            )

            counts = torch.zeros(self._nlist, device=device)
            counts.scatter_add_(0, assignments, torch.ones(n, device=device))

            non_empty = counts > 0
            new_centroids[non_empty] /= counts[non_empty].unsqueeze(1)

            empty_mask = ~non_empty
            if empty_mask.any():
                n_empty = int(empty_mask.sum().item())
                random_idx = torch.randint(n, (n_empty,), device=device)
                new_centroids[empty_mask] = vectors[random_idx]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self.centroids = centroids

    def _train_pq(self, residuals: Tensor, device: torch.device) -> None:
        """Train PQ codebooks using k-means on each subvector space."""
        n = residuals.shape[0]

        pq_centroids = torch.zeros(
            self._M, self._ksub, self._dsub, device=device, dtype=residuals.dtype
        )

        for m in range(self._M):
            # Extract subvectors for this subquantizer
            start = m * self._dsub
            end = start + self._dsub
            subvectors = residuals[:, start:end]  # (n, dsub)

            # K-means for this subspace
            perm = torch.randperm(n, device=device)[: self._ksub]
            centroids_m = subvectors[perm].clone()

            max_iters = 20
            for _ in range(max_iters):
                # Compute distances
                dists = torch.cdist(subvectors, centroids_m, p=2.0).pow(2)
                assignments = dists.argmin(dim=1)

                # Update centroids
                new_centroids = torch.zeros_like(centroids_m)
                new_centroids.scatter_add_(
                    0, assignments.unsqueeze(1).expand(-1, self._dsub), subvectors
                )

                counts = torch.zeros(self._ksub, device=device)
                counts.scatter_add_(0, assignments, torch.ones(n, device=device))

                non_empty = counts > 0
                new_centroids[non_empty] /= counts[non_empty].unsqueeze(1)

                empty_mask = ~non_empty
                if empty_mask.any():
                    n_empty = int(empty_mask.sum().item())
                    random_idx = torch.randint(n, (n_empty,), device=device)
                    new_centroids[empty_mask] = subvectors[random_idx]

                if torch.allclose(centroids_m, new_centroids, atol=1e-6):
                    break
                centroids_m = new_centroids

            pq_centroids[m] = centroids_m

        self.pq_centroids = pq_centroids

    def _encode(self, vectors: Tensor) -> Tensor:
        """
        Encode vectors using PQ.

        Args:
            vectors: Vectors (or residuals for L2) of shape (n, dim)

        Returns:
            codes: Shape (n, M) with values in [0, ksub)
        """
        n = vectors.shape[0]
        device = vectors.device
        codes = torch.zeros(n, self._M, dtype=torch.uint8, device=device)

        for m in range(self._M):
            start = m * self._dsub
            end = start + self._dsub
            subvectors = vectors[:, start:end]  # (n, dsub)

            if self.distance.name == "ip":
                # For IP: find centroid with highest inner product
                # (negated so argmin gives highest IP)
                scores = -torch.mm(subvectors, self.pq_centroids[m].t())
                codes[:, m] = scores.argmin(dim=1).to(torch.uint8)
            else:
                # For L2: find nearest centroid
                dists = torch.cdist(subvectors, self.pq_centroids[m], p=2.0)
                codes[:, m] = dists.argmin(dim=1).to(torch.uint8)

        return codes

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

        # Compute cluster assignments
        dists = self.distance.pairwise(vectors, self.centroids)
        new_assignments = dists.argmin(dim=1).int()

        # Encode vectors
        # For L2: use residuals (vector - centroid)
        # For IP: use vectors directly
        if self.distance.name == "l2":
            pq_vectors = vectors - self.centroids[new_assignments]
        else:
            pq_vectors = vectors
        new_codes = self._encode(pq_vectors)

        # Append to existing data
        old_ntotal = self.ntotal
        new_indices = torch.arange(
            old_ntotal, old_ntotal + n, dtype=torch.int, device=device
        )

        if old_ntotal == 0:
            self.codes = new_codes
            self.assignments = new_assignments
            self.indices = new_indices
        else:
            self.codes = torch.cat([self.codes, new_codes], dim=0)
            self.assignments = torch.cat([self.assignments, new_assignments], dim=0)
            self.indices = torch.cat([self.indices, new_indices], dim=0)

        # Rebuild sorted structure
        self._rebuild_lists()

    def _rebuild_lists(self) -> None:
        """Rebuild inverted list structure after adding vectors."""
        if self.ntotal == 0:
            return

        device = self.codes.device

        # Sort by cluster assignment
        sorted_order = torch.argsort(self.assignments)
        self.codes = self.codes[sorted_order]
        self.indices = self.indices[sorted_order]
        self.assignments = self.assignments[sorted_order]

        # Compute list sizes and offsets
        # Use bincount for O(n) instead of O(nlist * n) loop
        list_sizes = torch.bincount(
            self.assignments, minlength=self._nlist
        ).to(dtype=torch.int, device=device)

        list_offsets = torch.zeros(self._nlist, dtype=torch.int, device=device)
        list_offsets[1:] = torch.cumsum(list_sizes[:-1], dim=0)

        self.list_sizes = list_sizes
        self.list_offsets = list_offsets

    def _compute_distance_tables_ip(self, queries: Tensor) -> Tensor:
        """
        Compute distance tables for IP metric (TorchScript-compatible).

        Args:
            queries: Query vectors of shape (batch_size, dim)

        Returns:
            tables: Shape (batch_size, M, ksub) - negated inner products
        """
        batch_size = queries.shape[0]
        device = queries.device
        tables = torch.zeros(
            batch_size, self._M, self._ksub, device=device, dtype=queries.dtype
        )
        for m in range(self._M):
            start = m * self._dsub
            end = start + self._dsub
            query_sub = queries[:, start:end]  # (batch_size, dsub)
            tables[:, m, :] = -torch.mm(query_sub, self.pq_centroids[m].t())
        return tables

    def _compute_distance_tables_l2(
        self, queries: Tensor, centroids: Tensor
    ) -> Tensor:
        """
        Compute distance tables for L2 metric using residuals (TorchScript-compatible).

        For L2 with residual-based PQ:
            - residual = query - centroid (for each probed centroid)
            - Computes squared distances from residual subvectors to PQ centroids

        Args:
            queries: Query vectors of shape (batch_size, dim)
            centroids: Probed centroids of shape (batch_size, nprobe, dim)

        Returns:
            tables: Shape (batch_size, nprobe, M, ksub)
        """
        batch_size = queries.shape[0]
        nprobe = centroids.shape[1]
        device = queries.device

        # Compute residuals: (batch_size, nprobe, dim)
        residuals = queries.unsqueeze(1) - centroids

        # Precompute squared norms of PQ centroids: (M, ksub)
        # This is constant across queries so could be cached, but keeping it here
        # for TorchScript compatibility
        c_sq_all = (self.pq_centroids**2).sum(dim=-1)  # (M, ksub)

        tables = torch.zeros(
            batch_size,
            nprobe,
            self._M,
            self._ksub,
            device=device,
            dtype=queries.dtype,
        )

        for m in range(self._M):
            start = m * self._dsub
            end = start + self._dsub
            # residual_sub: (batch_size, nprobe, dsub)
            residual_sub = residuals[:, :, start:end]
            # Compute squared distances: ||r - c||^2 = ||r||^2 + ||c||^2 - 2*r.c
            r_sq = (residual_sub**2).sum(dim=-1, keepdim=True)  # (B, nprobe, 1)
            # (batch_size, nprobe, dsub) @ (dsub, ksub) -> (batch_size, nprobe, ksub)
            rc = torch.einsum("bnd,kd->bnk", residual_sub, self.pq_centroids[m])
            tables[:, :, m, :] = r_sq.squeeze(-1).unsqueeze(-1) + c_sq_all[m] - 2 * rc

        return tables

    def search(self, queries: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        """
        Batched k-nearest neighbor search using ADC (Asymmetric Distance Computation).

        Args:
            queries: Query vectors of shape (batch_size, dim) or (dim,) for single query
            k: Number of nearest neighbors to return

        Returns:
            distances: Shape (batch_size, k) - approximate distances to nearest neighbors
            indices: Shape (batch_size, k) - indices of nearest neighbors
        """
        queries, squeeze_output = self._normalize_vectors(queries, "Query vectors")
        batch_size = queries.shape[0]
        device = queries.device

        # Find nearest centroids for each query
        centroid_dists = self.distance.pairwise(queries, self.centroids)
        _, probe_indices = centroid_dists.topk(self._nprobe, dim=1, largest=False)

        # Get list sizes and offsets for probed clusters
        probed_sizes = self.list_sizes[probe_indices]
        probed_offsets = self.list_offsets[probe_indices]

        max_list_size = int(probed_sizes.max().item())

        if max_list_size == 0:
            all_distances = torch.full((batch_size, k), float("inf"), device=device)
            all_indices = torch.full(
                (batch_size, k), -1, dtype=torch.int, device=device
            )
            if squeeze_output:
                return all_distances.squeeze(0), all_indices.squeeze(0)
            return all_distances, all_indices

        # Build candidate indices
        local_indices = torch.arange(max_list_size, device=device)
        global_indices = probed_offsets.unsqueeze(2) + local_indices.view(1, 1, -1)
        valid_mask_3d = local_indices.view(1, 1, -1) < probed_sizes.unsqueeze(2)

        safe_global_indices = global_indices.clamp(min=0, max=max(self.ntotal - 1, 0))

        # Get codes and original indices for candidates
        # Shape: (batch_size, nprobe, max_list_size, M)
        candidate_codes = self.codes[safe_global_indices]
        candidate_orig_idx = self.indices[safe_global_indices]

        n_candidates = self._nprobe * max_list_size

        if self.distance.name == "ip":
            # For IP: compute distance tables from query subvectors
            # tables: (batch_size, M, ksub)
            distance_tables = self._compute_distance_tables_ip(queries)

            # Flatten codes: (batch_size, n_candidates, M)
            candidate_codes_flat = candidate_codes.view(
                batch_size, n_candidates, self._M
            )

            # Expand tables: (batch_size, n_candidates, M, ksub)
            tables_expanded = distance_tables.unsqueeze(1).expand(
                batch_size, n_candidates, self._M, self._ksub
            )

            # Gather and sum
            codes_expanded = candidate_codes_flat.unsqueeze(-1).long()
            gathered = tables_expanded.gather(-1, codes_expanded).squeeze(-1)
            distances = gathered.sum(dim=-1)  # (batch_size, n_candidates)
        else:
            # For L2: compute distance tables from query residuals
            # Get centroids for probed clusters: (batch_size, nprobe, dim)
            probed_centroids = self.centroids[probe_indices]

            # tables: (batch_size, nprobe, M, ksub)
            distance_tables = self._compute_distance_tables_l2(queries, probed_centroids)

            # For each candidate in each cluster, look up from that cluster's table
            # candidate_codes: (batch_size, nprobe, max_list_size, M)
            # We need to gather from the corresponding nprobe dimension

            # Expand tables: (batch_size, nprobe, max_list_size, M, ksub)
            tables_expanded = distance_tables.unsqueeze(2).expand(
                batch_size, self._nprobe, max_list_size, self._M, self._ksub
            )

            # Expand codes: (batch_size, nprobe, max_list_size, M, 1)
            codes_expanded = candidate_codes.unsqueeze(-1).long()

            # Gather: -> (batch_size, nprobe, max_list_size, M)
            gathered = tables_expanded.gather(-1, codes_expanded).squeeze(-1)

            # Sum over M subquantizers: (batch_size, nprobe, max_list_size)
            distances_3d = gathered.sum(dim=-1)

            # Flatten: (batch_size, n_candidates)
            distances = distances_3d.view(batch_size, n_candidates)

        # Flatten other arrays
        candidate_orig_idx_flat = candidate_orig_idx.view(batch_size, n_candidates)
        valid_mask_flat = valid_mask_3d.view(batch_size, n_candidates)

        # Mask invalid candidates
        distances = distances.masked_fill(~valid_mask_flat, float("inf"))

        # Select top-k
        actual_k = min(k, n_candidates)
        topk_dists, topk_local = distances.topk(actual_k, dim=1, largest=False)
        topk_indices = candidate_orig_idx_flat.gather(1, topk_local)

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

        topk_indices = topk_indices.masked_fill(topk_dists == float("inf"), -1)

        if squeeze_output:
            topk_dists = topk_dists.squeeze(0)
            topk_indices = topk_indices.squeeze(0)

        return topk_dists, topk_indices
