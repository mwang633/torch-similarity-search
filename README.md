# torch-similarity-search

[![PyPI version](https://badge.fury.io/py/torch-similarity-search.svg)](https://pypi.org/project/torch-similarity-search/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/mwang633/torch-similarity-search)

PyTorch-native similarity search. Convert trained FAISS indexes to pure `nn.Module` models for GPU inference.

**Train with FAISS, deploy with PyTorch.**

## Why?

- **No numpy overhead** - FAISS requires numpy conversion; this library keeps tensors on GPU
- **TorchScript export** - Deploy without FAISS dependency, load with just `torch.jit.load()`
- **GPU memory sharing** - Index vectors stay in GPU memory alongside your embedding model
- **Triton Inference Server ready** - Export once, serve anywhere

## Installation

```bash
pip install torch-similarity-search
```

For FAISS conversion support:

```bash
pip install torch-similarity-search faiss-cpu  # or faiss-gpu
```

## Quick Start

### Convert from FAISS

```python
import faiss
import torch
import torch_similarity_search as tss

# Train with FAISS (your existing workflow)
quantizer = faiss.IndexFlatL2(128)
index = faiss.IndexIVFFlat(quantizer, 128, 100)
index.train(vectors)
index.add(vectors)

# Convert to PyTorch
model = tss.from_faiss(index)
model = model.cuda()
model.nprobe = 10

# Search with PyTorch tensors (no numpy!)
queries = torch.randn(32, 128, device="cuda")
distances, indices = model.search(queries, k=10)
```

### Build from Scratch

```python
import torch
from torch_similarity_search import IVFFlatIndex

# Create and train
index = IVFFlatIndex(dim=128, nlist=100, metric="l2")
training_vectors = torch.randn(10000, 128)
index.train(training_vectors)
index.add(training_vectors)

# Move to GPU
index = index.cuda()

# Search
queries = torch.randn(32, 128, device="cuda")
distances, indices = index.search(queries, k=10)
```

### Export for Production

```python
# Export to TorchScript (no torch_similarity_search needed to load!)
scripted = torch.jit.script(model)
scripted.save("index.pt")

# Load anywhere - just needs PyTorch
model = torch.jit.load("index.pt")
model = model.cuda()
distances, indices = model.search(queries, k=10)
```

### Use with Embedding Models

```python
# End-to-end GPU inference
class SearchModel(torch.nn.Module):
    def __init__(self, encoder, index):
        super().__init__()
        self.encoder = encoder
        self.index = index

    def forward(self, text_embeddings):
        # Everything stays on GPU
        return self.index.search(text_embeddings, k=10)

# Export the complete pipeline
model = SearchModel(encoder, index)
torch.jit.script(model).save("search_pipeline.pt")
```

## Supported Index Types

| FAISS Index | PyTorch Module | Status |
|-------------|----------------|--------|
| `IndexFlat` | `FlatIndex` | ✅ Supported |
| `IndexIVFFlat` | `IVFFlatIndex` | ✅ Supported |
| `IndexIVFPQ` | `IVFPQIndex` | ✅ Supported |

## API Reference

### `FlatIndex`

Brute-force exact search - compares against all vectors. Best for small datasets or exact results.

```python
from torch_similarity_search import FlatIndex

index = FlatIndex(
    dim=128,          # Vector dimensionality
    metric="l2",      # Distance metric: "l2", "ip" (inner product), or "cosine"
    k=10,             # Default k for forward() method
)

index.add(vectors)    # No training required
distances, indices = index.search(queries, k=10)
```

### `IVFFlatIndex`

Inverted File Flat index - partitions vectors into clusters for fast approximate search.

```python
from torch_similarity_search import IVFFlatIndex

index = IVFFlatIndex(
    dim=128,          # Vector dimensionality
    nlist=100,        # Number of clusters (higher = faster but less accurate)
    metric="l2",      # Distance metric: "l2", "ip" (inner product), or "cosine"
    nprobe=10,        # Clusters to search at query time
    k=10,             # Default k for forward() method
)

index.train(vectors)  # Train centroids first
index.add(vectors)
distances, indices = index.search(queries, k=10)
```

**Common Methods (both index types):**

| Method | Description |
|--------|-------------|
| `add(vectors)` | Add vectors to index. Accepts `(n, dim)` or `(dim,)` tensors. |
| `search(queries, k)` | Find k nearest neighbors. Returns `(distances, indices)` tensors. |
| `forward(queries)` | Same as `search()` but uses configured `k`. For TorchScript export. |

**IVFFlatIndex-specific:**

| Method/Property | Description |
|-----------------|-------------|
| `train(vectors)` | Train cluster centroids via k-means. Requires `n >= nlist`. |
| `nprobe` | Clusters to probe during search (settable, higher = more accurate) |
| `is_trained` | Whether index has been trained |

### `IVFPQIndex`

Inverted File with Product Quantization - combines clustering with vector compression for memory-efficient approximate search. Best for large datasets where memory is a concern.

```python
from torch_similarity_search import IVFPQIndex

index = IVFPQIndex(
    dim=128,          # Vector dimensionality (must be divisible by M)
    nlist=100,        # Number of IVF clusters
    M=8,              # Number of PQ subquantizers (compression factor)
    nbits=8,          # Bits per code (default: 8, meaning 256 centroids per subquantizer)
    metric="l2",      # Distance metric: "l2" or "ip" (inner product)
    nprobe=10,        # Clusters to search at query time
    k=10,             # Default k for forward() method
)

index.train(vectors)  # Train IVF centroids and PQ codebooks
index.add(vectors)
distances, indices = index.search(queries, k=10)
```

**Compression:** With `M=8` and `nbits=8`, each 128-dim vector (512 bytes) is compressed to just 8 bytes - a 64x reduction in memory usage.

### `from_faiss(index)`

Convert a FAISS index to PyTorch.

```python
from torch_similarity_search import from_faiss

torch_index = from_faiss(faiss_index)  # Returns FlatIndex, IVFFlatIndex, or IVFPQIndex
```

**Supported:**
- `faiss.IndexFlatL2`, `faiss.IndexFlatIP` → `FlatIndex`
- `faiss.IndexIVFFlat` → `IVFFlatIndex`
- `faiss.IndexIVFPQ` → `IVFPQIndex`

## Requirements

- Python 3.11+
- PyTorch 2.0+
- NumPy (for FAISS conversion only)
- FAISS (optional, for conversion only)

## License

MIT
