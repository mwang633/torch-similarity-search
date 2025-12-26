"""FAISS IndexFlat conversion tests."""

import numpy as np
import pytest
import torch

faiss = pytest.importorskip("faiss")

from torch_similarity_search import from_faiss, FlatIndex  # noqa: E402


class TestFlatConverter:
    """Tests for IndexFlat conversion."""

    def test_convert_flat_l2(self):
        """Test converting FAISS IndexFlatL2 to PyTorch."""
        np.random.seed(42)
        dim = 64
        n_vectors = 1000
        n_queries = 50
        k = 10

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)

        # Build FAISS index
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(vectors)

        # Search with FAISS
        faiss_distances, faiss_indices = faiss_index.search(queries, k)

        # Convert to PyTorch
        torch_index = from_faiss(faiss_index)
        assert isinstance(torch_index, FlatIndex)
        assert torch_index.ntotal == n_vectors

        # Search with PyTorch
        queries_tensor = torch.from_numpy(queries)
        torch_distances, torch_indices = torch_index.search(queries_tensor, k)

        # Compare results - should be exact match
        torch_distances = torch_distances.numpy()
        torch_indices = torch_indices.numpy()

        np.testing.assert_array_equal(torch_indices, faiss_indices)
        np.testing.assert_allclose(
            torch_distances, faiss_distances, rtol=1e-5, atol=1e-5
        )

    def test_convert_flat_ip(self):
        """Test converting FAISS IndexFlatIP to PyTorch."""
        np.random.seed(42)
        dim = 64
        n_vectors = 1000
        n_queries = 50
        k = 10

        # Normalized vectors for inner product
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Build FAISS index
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)

        # Search with FAISS
        faiss_distances, faiss_indices = faiss_index.search(queries, k)

        # Convert to PyTorch
        torch_index = from_faiss(faiss_index)

        # Search with PyTorch
        queries_tensor = torch.from_numpy(queries)
        torch_distances, torch_indices = torch_index.search(queries_tensor, k)

        torch_distances = torch_distances.numpy()
        torch_indices = torch_indices.numpy()

        # Indices should match exactly
        np.testing.assert_array_equal(torch_indices, faiss_indices)

        # Distances: PyTorch returns negated IP, FAISS returns IP
        np.testing.assert_allclose(
            torch_distances, -faiss_distances, rtol=1e-5, atol=1e-5
        )

    def test_convert_empty_flat(self):
        """Test converting an empty FAISS IndexFlat."""
        dim = 64
        faiss_index = faiss.IndexFlatL2(dim)

        torch_index = from_faiss(faiss_index)

        assert torch_index.ntotal == 0
        assert torch_index.dim == dim

    def test_flat_torchscript_export(self):
        """Test that converted FlatIndex can be exported to TorchScript."""
        np.random.seed(42)
        dim = 32
        n_vectors = 100

        vectors = np.random.randn(n_vectors, dim).astype(np.float32)

        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(vectors)

        torch_index = from_faiss(faiss_index)

        # Export to TorchScript
        scripted = torch.jit.script(torch_index)

        # Test that it works
        queries = torch.randn(10, dim)
        d1, i1 = torch_index.search(queries, k=5)
        d2, i2 = scripted.search(queries, k=5)

        torch.testing.assert_close(d1, d2)
        torch.testing.assert_close(i1, i2)
