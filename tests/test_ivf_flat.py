"""Unit tests for IVFFlatIndex."""

import pytest
import torch

from torch_similarity_search.indexes.ivf_flat import IVFFlatIndex


class TestIVFFlatIndex:
    """Tests for IVFFlatIndex."""

    def test_init(self):
        """Test index initialization."""
        index = IVFFlatIndex(dim=128, nlist=10)
        assert index.dim == 128
        assert index.nlist == 10
        assert index.ntotal == 0
        assert index.nprobe == 20
        assert not index.is_trained

    def test_train(self):
        """Test index training."""
        index = IVFFlatIndex(dim=32, nlist=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)

        assert index.is_trained
        assert index.centroids.shape == (4, 32)

    def test_train_requires_enough_vectors(self):
        """Test that training requires at least nlist vectors."""
        index = IVFFlatIndex(dim=32, nlist=10)
        vectors = torch.randn(5, 32)

        with pytest.raises(ValueError, match="Need at least 10 vectors"):
            index.train(vectors)

    def test_add_requires_training(self):
        """Test that adding vectors requires training first."""
        index = IVFFlatIndex(dim=32, nlist=4)
        vectors = torch.randn(10, 32)

        with pytest.raises(RuntimeError, match="must be trained"):
            index.add(vectors)

    def test_add_vectors(self):
        """Test adding vectors to the index."""
        index = IVFFlatIndex(dim=32, nlist=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        assert index.ntotal == 100

    def test_add_single_vector(self):
        """Test adding a single vector."""
        index = IVFFlatIndex(dim=32, nlist=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors[0])  # Single vector (dim,)

        assert index.ntotal == 1

    def test_add_multiple_batches(self):
        """Test adding vectors in multiple batches."""
        index = IVFFlatIndex(dim=32, nlist=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors[:50])
        index.add(vectors[50:])

        assert index.ntotal == 100

    def test_search_basic(self):
        """Test basic search functionality."""
        index = IVFFlatIndex(dim=32, nlist=4, nprobe=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]  # Query with first vector
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        # First result should be the query itself (distance ~0)
        assert indices[0, 0] == 0
        assert distances[0, 0] < 1e-5

    def test_search_single_query(self):
        """Test search with a single query vector (dim,)."""
        index = IVFFlatIndex(dim=32, nlist=4, nprobe=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0]  # Single query (dim,)
        distances, indices = index.search(query, k=5)

        assert distances.shape == (5,)
        assert indices.shape == (5,)
        assert indices[0] == 0

    def test_search_batched(self):
        """Test batched search."""
        index = IVFFlatIndex(dim=32, nlist=4, nprobe=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        queries = vectors[:10]  # 10 queries
        distances, indices = index.search(queries, k=5)

        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
        # Each query should find itself as the nearest neighbor
        for i in range(10):
            assert indices[i, 0] == i

    def test_search_k_larger_than_ntotal(self):
        """Test search when k > ntotal."""
        index = IVFFlatIndex(dim=32, nlist=4, nprobe=4)
        vectors = torch.randn(10, 32)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=20)

        assert distances.shape == (1, 20)
        assert indices.shape == (1, 20)
        # Only first 10 should be valid
        assert (indices[0, :10] >= 0).all()
        assert (indices[0, 10:] == -1).all()
        assert (distances[0, 10:] == float("inf")).all()

    def test_nprobe_setter(self):
        """Test nprobe setter validation."""
        index = IVFFlatIndex(dim=32, nlist=10)

        index.nprobe = 5
        assert index.nprobe == 5

        with pytest.raises(ValueError):
            index.nprobe = 0

        with pytest.raises(ValueError):
            index.nprobe = 11

    def test_inner_product_metric(self):
        """Test inner product metric."""
        index = IVFFlatIndex(dim=32, nlist=4, metric="ip", nprobe=4)
        vectors = torch.randn(100, 32)
        # Normalize for meaningful inner product
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices[0, 0] == 0
        # For normalized vectors, IP distance should be close to -1 (negated IP)
        assert distances[0, 0] < -0.99

    def test_forward_alias(self):
        """Test that forward() uses configured k for Triton compatibility."""
        index = IVFFlatIndex(dim=32, nlist=4, nprobe=4, k=5)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        d1, i1 = index.search(query, k=5)
        d2, i2 = index.forward(query)  # Uses configured k=5

        assert torch.equal(d1, d2)
        assert torch.equal(i1, i2)

        # Test k setter
        index.k = 3
        d3, i3 = index.forward(query)
        assert d3.shape == (1, 3)
        assert i3.shape == (1, 3)

    def test_recall_with_higher_nprobe(self):
        """Test that higher nprobe gives better recall."""
        torch.manual_seed(42)
        index = IVFFlatIndex(dim=32, nlist=10, nprobe=1)
        vectors = torch.randn(1000, 32)

        index.train(vectors)
        index.add(vectors)

        queries = torch.randn(50, 32)

        # Compute ground truth with nprobe=10 (all clusters)
        index.nprobe = 10
        _, gt_indices = index.search(queries, k=10)

        # Compute with nprobe=1
        index.nprobe = 1
        _, indices_1 = index.search(queries, k=10)

        # Compute with nprobe=5
        index.nprobe = 5
        _, indices_5 = index.search(queries, k=10)

        def recall_at_k(pred, gt):
            matches = 0
            for i in range(pred.shape[0]):
                matches += len(set(pred[i].tolist()) & set(gt[i].tolist()))
            return matches / (pred.shape[0] * pred.shape[1])

        recall_1 = recall_at_k(indices_1, gt_indices)
        recall_5 = recall_at_k(indices_5, gt_indices)

        # Higher nprobe should give better recall
        assert recall_5 >= recall_1

    def test_cosine_metric(self):
        """Test cosine distance metric."""
        index = IVFFlatIndex(dim=32, nlist=4, metric="cosine", nprobe=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices[0, 0] == 0
        # For the query itself, cosine distance should be ~0 (1 - 1 = 0)
        assert distances[0, 0] < 1e-5

    def test_cosine_vs_manual(self):
        """Test cosine distance matches manual computation."""
        index = IVFFlatIndex(dim=32, nlist=4, metric="cosine", nprobe=4)

        torch.manual_seed(42)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        queries = torch.randn(10, 32)
        distances, indices = index.search(queries, k=5)

        # Verify results by recomputing cosine distances
        for i in range(10):
            for j in range(5):
                idx = indices[i, j].item()
                if idx >= 0:
                    q_norm = queries[i] / queries[i].norm()
                    v_norm = vectors[idx] / vectors[idx].norm()
                    expected_dist = 1.0 - (q_norm * v_norm).sum()
                    assert abs(distances[i, j] - expected_dist) < 1e-4

    def test_cuda_if_available(self):
        """Test that index works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        index = IVFFlatIndex(dim=32, nlist=4, nprobe=4)
        vectors = torch.randn(100, 32)

        index.train(vectors)
        index.add(vectors)

        # Move to CUDA
        index = index.cuda()
        queries = vectors[:10].cuda()

        distances, indices = index.search(queries, k=5)

        assert distances.device.type == "cuda"
        assert indices.device.type == "cuda"
