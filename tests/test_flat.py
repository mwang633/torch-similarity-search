"""Tests for FlatIndex."""

import torch
import pytest

from torch_similarity_search import FlatIndex


class TestFlatIndex:
    """Tests for FlatIndex."""

    def test_init(self):
        """Test index initialization."""
        index = FlatIndex(dim=128)
        assert index.dim == 128
        assert index.ntotal == 0
        assert index.k == 10

    def test_add_vectors(self):
        """Test adding vectors."""
        index = FlatIndex(dim=32)
        vectors = torch.randn(100, 32)

        index.add(vectors)

        assert index.ntotal == 100

    def test_add_single_vector(self):
        """Test adding a single vector (1D input)."""
        index = FlatIndex(dim=32)
        vector = torch.randn(32)

        index.add(vector)

        assert index.ntotal == 1

    def test_add_multiple_batches(self):
        """Test adding vectors in multiple batches."""
        index = FlatIndex(dim=32)

        index.add(torch.randn(50, 32))
        assert index.ntotal == 50

        index.add(torch.randn(30, 32))
        assert index.ntotal == 80

        index.add(torch.randn(32))  # Single vector
        assert index.ntotal == 81

    def test_search_basic(self):
        """Test basic search functionality."""
        index = FlatIndex(dim=32)
        vectors = torch.randn(100, 32)

        index.add(vectors)

        # Search for the first vector - should find itself
        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        # First result should be the query itself (distance ~0)
        assert indices[0, 0] == 0
        assert distances[0, 0] < 1e-5

    def test_search_single_query(self):
        """Test search with single query (1D input)."""
        index = FlatIndex(dim=32)
        vectors = torch.randn(100, 32)

        index.add(vectors)

        # Single query (1D tensor)
        query = vectors[0]
        distances, indices = index.search(query, k=5)

        # Should return 1D results for 1D input
        assert distances.shape == (5,)
        assert indices.shape == (5,)
        assert indices[0] == 0

    def test_search_batched(self):
        """Test batched search."""
        index = FlatIndex(dim=32)
        vectors = torch.randn(100, 32)

        index.add(vectors)

        # Batch of 10 queries
        queries = vectors[:10]
        distances, indices = index.search(queries, k=5)

        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
        # Each query should find itself first
        for i in range(10):
            assert indices[i, 0] == i

    def test_search_k_larger_than_ntotal(self):
        """Test search when k > ntotal."""
        index = FlatIndex(dim=32)
        vectors = torch.randn(5, 32)

        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=10)

        assert distances.shape == (1, 10)
        assert indices.shape == (1, 10)
        # First 5 should be valid, rest should be -1
        assert (indices[0, :5] >= 0).all()
        assert (indices[0, 5:] == -1).all()
        assert (distances[0, 5:] == float("inf")).all()

    def test_search_empty_index(self):
        """Test search on empty index."""
        index = FlatIndex(dim=32)

        query = torch.randn(1, 32)
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        assert (indices == -1).all()
        assert (distances == float("inf")).all()

    def test_inner_product_metric(self):
        """Test inner product distance metric."""
        index = FlatIndex(dim=32, metric="ip")

        # Create normalized vectors
        vectors = torch.randn(100, 32)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices[0, 0] == 0
        # For normalized vectors, IP distance should be close to -1 (negated IP)
        assert distances[0, 0] < -0.99

    def test_forward_alias(self):
        """Test that forward() uses configured k."""
        index = FlatIndex(dim=32, k=5)
        vectors = torch.randn(100, 32)

        index.add(vectors)

        query = vectors[0:1]
        d1, i1 = index.search(query, k=5)
        d2, i2 = index.forward(query)

        assert torch.equal(d1, d2)
        assert torch.equal(i1, i2)

        # Test k setter
        index.k = 3
        d3, i3 = index.forward(query)
        assert d3.shape == (1, 3)
        assert i3.shape == (1, 3)

    def test_exact_results(self):
        """Test that FlatIndex returns exact results."""
        torch.manual_seed(42)
        index = FlatIndex(dim=32)
        vectors = torch.randn(1000, 32)

        index.add(vectors)

        # Search for random queries
        queries = torch.randn(10, 32)
        distances, indices = index.search(queries, k=10)

        # Verify results by recomputing distances
        for i in range(10):
            for j in range(10):
                idx = indices[i, j].item()
                if idx >= 0:
                    expected_dist = ((queries[i] - vectors[idx]) ** 2).sum()
                    assert abs(distances[i, j] - expected_dist) < 1e-4

    def test_torchscript_export(self):
        """Test TorchScript export."""
        index = FlatIndex(dim=32, k=5)
        vectors = torch.randn(100, 32)
        index.add(vectors)

        # Export to TorchScript
        scripted = torch.jit.script(index)

        # Compare results
        queries = torch.randn(10, 32)
        d1, i1 = index.search(queries, k=5)
        d2, i2 = scripted.search(queries, k=5)

        torch.testing.assert_close(d1, d2)
        torch.testing.assert_close(i1, i2)

    def test_dimension_validation(self):
        """Test that dimension mismatch raises error."""
        index = FlatIndex(dim=32)
        index.add(torch.randn(10, 32))

        with pytest.raises(ValueError, match="dimension"):
            index.add(torch.randn(10, 64))

        with pytest.raises(ValueError, match="dimension"):
            index.search(torch.randn(10, 64), k=5)

    def test_cosine_metric(self):
        """Test cosine distance metric."""
        index = FlatIndex(dim=32, metric="cosine")

        # Create random vectors (cosine will auto-normalize)
        vectors = torch.randn(100, 32)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices[0, 0] == 0
        # For the query itself, cosine distance should be ~0 (1 - 1 = 0)
        assert distances[0, 0] < 1e-5

    def test_cosine_vs_manual(self):
        """Test cosine distance matches manual computation."""
        index = FlatIndex(dim=32, metric="cosine")

        torch.manual_seed(42)
        vectors = torch.randn(100, 32)
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
                    assert abs(distances[i, j] - expected_dist) < 1e-5

    def test_cuda_if_available(self):
        """Test CUDA support if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        index = FlatIndex(dim=32)
        vectors = torch.randn(100, 32)

        index.add(vectors)
        index = index.cuda()

        queries = torch.randn(10, 32, device="cuda")
        distances, indices = index.search(queries, k=5)

        assert distances.device.type == "cuda"
        assert indices.device.type == "cuda"
