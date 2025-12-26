"""Unit tests for IVFPQIndex."""

import pytest
import torch

from torch_similarity_search.indexes.ivf_pq import IVFPQIndex


class TestIVFPQIndex:
    """Tests for IVFPQIndex."""

    def test_init(self):
        """Test index initialization."""
        index = IVFPQIndex(dim=128, nlist=10, M=8)
        assert index.dim == 128
        assert index.nlist == 10
        assert index.M == 8
        assert index.nbits == 8
        assert index.ksub == 256
        assert index.dsub == 16  # 128 / 8
        assert index.ntotal == 0
        assert index.nprobe == 10
        assert not index.is_trained

    def test_init_dimension_validation(self):
        """Test that dim must be divisible by M."""
        with pytest.raises(ValueError, match="divisible"):
            IVFPQIndex(dim=100, nlist=10, M=8)  # 100 not divisible by 8

    def test_init_metric_validation(self):
        """Test that unsupported metrics raise an error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            IVFPQIndex(dim=128, nlist=10, M=8, metric="cosine")

    def test_init_nbits_validation(self):
        """Test that nbits must be <= 8."""
        with pytest.raises(ValueError, match="nbits must be <= 8"):
            IVFPQIndex(dim=128, nlist=10, M=8, nbits=16)

    def test_train(self):
        """Test index training."""
        index = IVFPQIndex(dim=64, nlist=4, M=8)
        vectors = torch.randn(500, 64)

        index.train(vectors)

        assert index.is_trained
        assert index.centroids.shape == (4, 64)
        assert index.pq_centroids.shape == (8, 256, 8)  # (M, ksub, dsub)

    def test_train_requires_enough_vectors(self):
        """Test that training requires sufficient vectors."""
        index = IVFPQIndex(dim=64, nlist=10, M=8)
        vectors = torch.randn(50, 64)  # Need at least max(nlist, ksub) = 256

        with pytest.raises(ValueError, match="Need at least"):
            index.train(vectors)

    def test_add_requires_training(self):
        """Test that adding vectors requires training first."""
        index = IVFPQIndex(dim=64, nlist=4, M=8)
        vectors = torch.randn(10, 64)

        with pytest.raises(RuntimeError, match="must be trained"):
            index.add(vectors)

    def test_add_vectors(self):
        """Test adding vectors to the index."""
        index = IVFPQIndex(dim=64, nlist=4, M=8)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors)

        assert index.ntotal == 500
        assert index.codes.shape == (500, 8)
        assert index.codes.dtype == torch.uint8

    def test_add_single_vector(self):
        """Test adding a single vector."""
        index = IVFPQIndex(dim=64, nlist=4, M=8)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors[0])  # Single vector (dim,)

        assert index.ntotal == 1

    def test_add_multiple_batches(self):
        """Test adding vectors in multiple batches."""
        index = IVFPQIndex(dim=64, nlist=4, M=8)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors[:250])
        index.add(vectors[250:])

        assert index.ntotal == 500

    def test_search_basic(self):
        """Test basic search functionality."""
        index = IVFPQIndex(dim=64, nlist=4, M=8, nprobe=4)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        # First result should be the query itself (or very close)
        # Note: PQ is approximate, so exact match not guaranteed
        assert indices[0, 0] == 0 or 0 in indices[0].tolist()

    def test_search_single_query(self):
        """Test search with a single query vector (dim,)."""
        index = IVFPQIndex(dim=64, nlist=4, M=8, nprobe=4)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0]  # Single query (dim,)
        distances, indices = index.search(query, k=5)

        assert distances.shape == (5,)
        assert indices.shape == (5,)

    def test_search_batched(self):
        """Test batched search."""
        index = IVFPQIndex(dim=64, nlist=4, M=8, nprobe=4)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors)

        queries = vectors[:10]  # 10 queries
        distances, indices = index.search(queries, k=5)

        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)

    def test_search_k_larger_than_candidates(self):
        """Test search when k > available candidates."""
        index = IVFPQIndex(dim=64, nlist=4, M=8, nprobe=1)
        vectors = torch.randn(20, 64)

        index.train(torch.randn(500, 64))  # Train on more data
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=50)

        assert distances.shape == (1, 50)
        assert indices.shape == (1, 50)
        # Some results should be padded with -1
        assert (indices[0] == -1).any() or index.ntotal >= 50

    def test_nprobe_setter(self):
        """Test nprobe setter validation."""
        index = IVFPQIndex(dim=64, nlist=10, M=8)

        index.nprobe = 5
        assert index.nprobe == 5

        with pytest.raises(ValueError):
            index.nprobe = 0

        with pytest.raises(ValueError):
            index.nprobe = 11

    def test_forward_alias(self):
        """Test that forward() uses configured k."""
        index = IVFPQIndex(dim=64, nlist=4, M=8, nprobe=4, k=5)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        d1, i1 = index.search(query, k=5)
        d2, i2 = index.forward(query)

        torch.testing.assert_close(d1, d2)
        torch.testing.assert_close(i1, i2)

        # Test k setter
        index.k = 3
        d3, i3 = index.forward(query)
        assert d3.shape == (1, 3)
        assert i3.shape == (1, 3)

    def test_recall_with_higher_nprobe(self):
        """Test that higher nprobe gives better recall."""
        torch.manual_seed(42)
        index = IVFPQIndex(dim=64, nlist=10, M=8, nprobe=1)
        vectors = torch.randn(1000, 64)

        index.train(vectors)
        index.add(vectors)

        queries = torch.randn(50, 64)

        # Compute with nprobe=10 (all clusters) as "ground truth"
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

    def test_compression_ratio(self):
        """Test that PQ actually compresses vectors."""
        dim = 128
        M = 16
        index = IVFPQIndex(dim=dim, nlist=4, M=M)
        vectors = torch.randn(1000, dim)

        index.train(vectors)
        index.add(vectors)

        # Original size: 1000 vectors * 128 dims * 4 bytes = 512KB
        # Compressed size: 1000 vectors * 16 bytes = 16KB (32x compression)
        assert index.codes.shape == (1000, M)
        assert index.codes.dtype == torch.uint8  # 1 byte per subquantizer

    def test_inner_product_metric(self):
        """Test inner product metric."""
        index = IVFPQIndex(dim=64, nlist=4, M=8, metric="ip", nprobe=4)

        # Use normalized vectors for meaningful IP
        vectors = torch.randn(500, 64)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        index.train(vectors)
        index.add(vectors)

        query = vectors[0:1]
        distances, indices = index.search(query, k=5)

        assert distances.shape == (1, 5)
        # First result should be the query itself (or very close)
        assert indices[0, 0] == 0 or 0 in indices[0].tolist()
        # For normalized vectors, IP distance should be negative (negated IP)
        # The best match should have the most negative distance (highest IP)
        assert distances[0, 0] < 0

    def test_inner_product_search(self):
        """Test that IP search finds similar vectors."""
        torch.manual_seed(42)
        index = IVFPQIndex(dim=64, nlist=4, M=8, metric="ip", nprobe=4)

        # Create normalized vectors
        vectors = torch.randn(500, 64)
        vectors = vectors / vectors.norm(dim=1, keepdim=True)

        index.train(vectors)
        index.add(vectors)

        # Search for each of the first 10 vectors
        queries = vectors[:10]
        distances, indices = index.search(queries, k=5)

        # Each query should find itself among top results (PQ is approximate)
        found_self = 0
        for i in range(10):
            if i in indices[i].tolist():
                found_self += 1

        # Should find itself in most cases
        assert found_self >= 5, f"Only found {found_self}/10 queries in their own results"

    def test_cuda_if_available(self):
        """Test that index works on CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        index = IVFPQIndex(dim=64, nlist=4, M=8, nprobe=4)
        vectors = torch.randn(500, 64)

        index.train(vectors)
        index.add(vectors)

        # Move to CUDA
        index = index.cuda()
        queries = vectors[:10].cuda()

        distances, indices = index.search(queries, k=5)

        assert distances.device.type == "cuda"
        assert indices.device.type == "cuda"
