"""FAISS IndexIVFFlat conversion tests."""

import os
import tempfile

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from torch_similarity_search import from_faiss, FlatIndex  # noqa: E402


class TestIVFFlatConverter:
    """Tests for IndexIVFFlat conversion."""

    def test_convert_ivf_flat_l2(self):
        """Test converting FAISS IndexIVFFlat (L2) to PyTorch."""
        np.random.seed(42)
        dim = 64
        nlist = 10
        n_vectors = 1000
        n_queries = 50
        k = 10

        # Create random data
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        # Build FAISS index
        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        faiss_index.train(vectors)
        faiss_index.add(vectors)
        faiss_index.nprobe = nlist  # Search all clusters for exact comparison

        # Search with FAISS
        faiss_distances, faiss_indices = faiss_index.search(queries, k)

        # Convert to PyTorch
        torch_index = from_faiss(faiss_index)
        torch_index.nprobe = nlist

        # Search with PyTorch
        queries_tensor = torch.from_numpy(queries)
        torch_distances, torch_indices = torch_index.search(queries_tensor, k)

        # Compare results
        torch_distances = torch_distances.numpy()
        torch_indices = torch_indices.numpy()

        # Indices should match exactly (or be equivalent in case of ties)
        # For exact comparison, check that the sets of returned indices match
        for i in range(n_queries):
            faiss_set = set(faiss_indices[i].tolist())
            torch_set = set(torch_indices[i].tolist())
            # Allow for -1 padding
            faiss_set.discard(-1)
            torch_set.discard(-1)
            assert faiss_set == torch_set, (
                f"Query {i}: FAISS {faiss_set} != PyTorch {torch_set}"
            )

        # Distances should be very close
        np.testing.assert_allclose(
            torch_distances, faiss_distances, rtol=1e-4, atol=1e-4
        )

    def test_convert_ivf_flat_ip(self):
        """Test converting FAISS IndexIVFFlat (Inner Product) to PyTorch."""
        np.random.seed(42)
        dim = 64
        nlist = 10
        n_vectors = 1000
        n_queries = 50
        k = 10

        # Create random normalized data for inner product
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Build FAISS index
        quantizer = faiss.IndexFlatIP(dim)
        faiss_index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
        )
        faiss_index.train(vectors)
        faiss_index.add(vectors)
        faiss_index.nprobe = nlist

        # Search with FAISS
        faiss_distances, faiss_indices = faiss_index.search(queries, k)

        # Convert to PyTorch
        torch_index = from_faiss(faiss_index)
        torch_index.nprobe = nlist

        # Search with PyTorch
        queries_tensor = torch.from_numpy(queries)
        torch_distances, torch_indices = torch_index.search(queries_tensor, k)

        torch_distances = torch_distances.numpy()
        torch_indices = torch_indices.numpy()

        # Indices should match
        for i in range(n_queries):
            faiss_set = set(faiss_indices[i].tolist())
            torch_set = set(torch_indices[i].tolist())
            faiss_set.discard(-1)
            torch_set.discard(-1)
            assert faiss_set == torch_set

        # Distances: PyTorch returns negated IP, FAISS returns IP
        # So torch_distances should be close to -faiss_distances
        np.testing.assert_allclose(
            torch_distances, -faiss_distances, rtol=1e-4, atol=1e-4
        )

    def test_recall_at_k(self):
        """Test recall@k using FlatIndex as ground truth baseline."""
        np.random.seed(42)
        dim = 128
        nlist = 50
        n_vectors = 5000
        n_queries = 100
        k = 10

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        # Ground truth using our FlatIndex (exact search)
        gt_index = FlatIndex(dim=dim, metric="l2")
        gt_index.add(torch.from_numpy(vectors))
        queries_tensor = torch.from_numpy(queries)
        _, gt_indices = gt_index.search(queries_tensor, k)
        gt_indices = gt_indices.numpy()

        # Build IVF index via FAISS and convert
        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        faiss_index.train(vectors)
        faiss_index.add(vectors)
        torch_index = from_faiss(faiss_index)

        # Compute recall helper
        def compute_recall(pred, gt):
            recall = 0
            for i in range(pred.shape[0]):
                recall += len(set(pred[i].tolist()) & set(gt[i].tolist()))
            return recall / (pred.shape[0] * k)

        # Test different nprobe values - recall should increase
        prev_recall = 0
        for nprobe in [1, 5, 10, 20]:
            torch_index.nprobe = nprobe

            torch_distances, torch_indices = torch_index.search(queries_tensor, k)
            torch_indices = torch_indices.numpy()

            recall = compute_recall(torch_indices, gt_indices)

            # Recall should increase with nprobe
            assert recall >= prev_recall, (
                f"Recall decreased: nprobe={nprobe}, recall={recall:.3f}, "
                f"prev={prev_recall:.3f}"
            )
            prev_recall = recall

        # With nprobe=20, recall should be decent (>0.5)
        assert prev_recall > 0.5, f"Final recall too low: {prev_recall:.3f}"

    def test_empty_index_conversion(self):
        """Test converting an empty FAISS index."""
        dim = 64
        nlist = 10

        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        # Train but don't add vectors
        vectors = np.random.randn(100, dim).astype(np.float32)
        faiss_index.train(vectors)

        torch_index = from_faiss(faiss_index)

        assert torch_index.ntotal == 0
        assert torch_index.is_trained

    def test_torchscript_export(self):
        """Test that the converted index can be exported to TorchScript."""
        np.random.seed(42)
        dim = 32
        nlist = 4
        n_vectors = 100

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
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

    def test_save_load(self):
        """Test saving and loading the PyTorch index via TorchScript."""
        np.random.seed(42)
        dim = 32
        nlist = 4
        n_vectors = 100

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        quantizer = faiss.IndexFlatL2(dim)
        faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        faiss_index.train(vectors)
        faiss_index.add(vectors)

        torch_index = from_faiss(faiss_index)
        torch_index.nprobe = nlist

        # Export to TorchScript and save
        scripted = torch.jit.script(torch_index)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "index.pt")
            scripted.save(path)

            # Load without needing the class definition
            loaded_index = torch.jit.load(path)

            # Compare
            queries = torch.randn(10, dim)
            d1, i1 = torch_index.search(queries, k=5)
            d2, i2 = loaded_index.search(queries, k=5)

            torch.testing.assert_close(d1, d2)
            torch.testing.assert_close(i1, i2)
