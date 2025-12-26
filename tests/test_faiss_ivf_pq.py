"""FAISS IndexIVFPQ conversion tests."""

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from torch_similarity_search import FlatIndex, IVFPQIndex, from_faiss  # noqa: E402


class TestIVFPQConverter:
    """Tests for IndexIVFPQ conversion."""

    def test_convert_ivfpq(self):
        """Test converting FAISS IndexIVFPQ to PyTorch."""
        np.random.seed(42)
        dim = 64
        nlist = 10
        M = 8  # Number of subquantizers
        n_vectors = 1000
        n_queries = 50
        k = 10

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        # Build FAISS IVFPQ index
        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, 8)  # 8 bits per code
        faiss_index.train(vectors)
        faiss_index.add(vectors)
        faiss_index.nprobe = nlist  # Search all clusters

        # Convert to PyTorch
        torch_index = from_faiss(faiss_index)
        assert isinstance(torch_index, IVFPQIndex)
        assert torch_index.ntotal == n_vectors
        assert torch_index.M == M
        assert torch_index.nlist == nlist
        torch_index.nprobe = nlist

        # Search with PyTorch
        queries_tensor = torch.from_numpy(queries)
        torch_distances, torch_indices = torch_index.search(queries_tensor, k)

        # Verify shapes
        assert torch_distances.shape == (n_queries, k)
        assert torch_indices.shape == (n_queries, k)

        # All indices should be valid (no -1 padding when searching all clusters)
        assert (torch_indices >= 0).all()

    def test_convert_ivfpq_ip(self):
        """Test converting FAISS IndexIVFPQ with inner product metric."""
        np.random.seed(42)
        dim = 64
        nlist = 10
        M = 8
        n_vectors = 1000
        n_queries = 50
        k = 10

        # Use normalized vectors for meaningful IP
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        # Build FAISS IVFPQ index with inner product
        quantizer = faiss.IndexFlatIP(dim)
        faiss_index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, M, 8, faiss.METRIC_INNER_PRODUCT
        )
        faiss_index.train(vectors)
        faiss_index.add(vectors)
        faiss_index.nprobe = nlist

        # Convert to PyTorch
        torch_index = from_faiss(faiss_index)
        assert isinstance(torch_index, IVFPQIndex)
        assert torch_index.ntotal == n_vectors
        assert torch_index.distance.name == "ip"
        torch_index.nprobe = nlist

        # Search with PyTorch
        queries_tensor = torch.from_numpy(queries)
        torch_distances, torch_indices = torch_index.search(queries_tensor, k)

        # Verify shapes
        assert torch_distances.shape == (n_queries, k)
        assert torch_indices.shape == (n_queries, k)

        # Distances should be negative (negated IP)
        assert (torch_distances < 0).any()

        # All indices should be valid
        assert (torch_indices >= 0).all()

    def test_convert_empty_ivfpq(self):
        """Test converting an empty FAISS IndexIVFPQ."""
        dim = 64
        nlist = 10
        M = 8

        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, 8)

        # Train but don't add vectors
        vectors = np.random.randn(1000, dim).astype(np.float32)
        faiss_index.train(vectors)

        torch_index = from_faiss(faiss_index)

        assert torch_index.ntotal == 0
        assert torch_index.is_trained
        assert torch_index.M == M

    def test_ivfpq_recall(self):
        """Test IVFPQ recall: both FAISS and PyTorch should have similar recall vs exact."""
        np.random.seed(42)
        dim = 64
        nlist = 20
        M = 8
        n_vectors = 10000
        n_queries = 50
        k = 10

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        # Ground truth using FlatIndex (exact search)
        gt_index = FlatIndex(dim=dim, metric="l2")
        gt_index.add(torch.from_numpy(vectors))
        queries_tensor = torch.from_numpy(queries)
        _, gt_indices = gt_index.search(queries_tensor, k)
        gt_indices = gt_indices.numpy()

        # Build IVFPQ via FAISS
        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, 8)
        faiss_index.train(vectors)
        faiss_index.add(vectors)
        faiss_index.nprobe = nlist  # Search all clusters

        # Get FAISS results
        _, faiss_indices = faiss_index.search(queries, k)

        # Convert to PyTorch and search
        torch_index = from_faiss(faiss_index)
        torch_index.nprobe = nlist
        _, torch_indices = torch_index.search(queries_tensor, k)
        torch_indices = torch_indices.numpy()

        # Compute recall vs ground truth
        def compute_recall(pred, gt):
            recall = 0
            for i in range(pred.shape[0]):
                recall += len(set(pred[i].tolist()) & set(gt[i].tolist()))
            return recall / (pred.shape[0] * k)

        faiss_recall = compute_recall(faiss_indices, gt_indices)
        torch_recall = compute_recall(torch_indices, gt_indices)
        recall_diff = torch_recall - faiss_recall

        print(f"FAISS recall: {faiss_recall:.3f}")
        print(f"PyTorch recall: {torch_recall:.3f}")
        print(f"Difference (PyTorch - FAISS): {recall_diff:.3f}")

        # PyTorch recall should be at least as good as FAISS (within 5% tolerance).
        # Negative diff means our implementation is worse than FAISS.
        assert recall_diff >= -0.05, (
            f"PyTorch recall too low vs FAISS: FAISS={faiss_recall:.3f}, "
            f"PyTorch={torch_recall:.3f}, diff={recall_diff:.3f}"
        )

    def test_ivfpq_torchscript_export(self):
        """Test that converted IVFPQ can be exported to TorchScript."""
        np.random.seed(42)
        dim = 32
        nlist = 4
        M = 4
        n_vectors = 500

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFPQ(quantizer, dim, nlist, M, 8)
        faiss_index.train(vectors)
        faiss_index.add(vectors)

        torch_index = from_faiss(faiss_index)
        torch_index.nprobe = nlist

        # Export to TorchScript
        scripted = torch.jit.script(torch_index)

        # Test that it works
        queries = torch.randn(10, dim)
        d1, i1 = torch_index.search(queries, k=5)
        d2, i2 = scripted.search(queries, k=5)

        torch.testing.assert_close(d1, d2)
        torch.testing.assert_close(i1, i2)
