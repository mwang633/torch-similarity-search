# torch-similarity-search

PyTorch-native similarity search library. Convert trained FAISS indexes to pure `nn.Module` models for GPU inference.

**Train with FAISS, serve with PyTorch.**

## Why?

- **No numpy overhead** - FAISS requires numpy conversion; this keeps tensors on GPU
- **TorchScript/ONNX export** - Unified inference stack without FAISS dependency
- **GPU memory sharing** - Vectors stay in GPU memory alongside embedding models
- **Production ready** - No FAISS required in deployment containers

## Installation

```bash
pip install torch-similarity-search
```

## Quick Start

### From FAISS

```python
import faiss
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

# Search with PyTorch tensors
queries = torch.randn(32, 128).cuda()
distances, indices = model.search(queries, k=10)
```

### From Scratch

```python
import torch
import torch_similarity_search as tss

# Create and train index
index = tss.IVFFlatIndex(dim=128, nlist=100, metric="l2")
index.train(vectors)
index.add(vectors)

# Search
distances, indices = index.search(queries, k=10)
```

### Export to TorchScript

```python
# Export for deployment (no torch_similarity_search needed to load)
scripted = torch.jit.script(model)
scripted.save("index.pt")

# Load anywhere
model = torch.jit.load("index.pt")
distances, indices = model.search(queries, k=10)
```

## Supported Index Types

| FAISS Index | PyTorch Module | Status |
|-------------|----------------|--------|
| `IndexIVFFlat` | `IVFFlatIndex` | Supported |
| `IndexIVFPQ` | `IVFPQIndex` | Planned |

## API

### `IVFFlatIndex`

```python
IVFFlatIndex(
    dim: int,           # Vector dimensionality
    nlist: int,         # Number of clusters
    metric: str = "l2", # "l2" or "ip" (inner product)
    nprobe: int = 1,    # Clusters to search (accuracy vs speed)
)
```

**Methods:**
- `train(vectors)` - Train centroids via k-means
- `add(vectors)` - Add vectors to the index
- `search(queries, k)` - Find k nearest neighbors

**Properties:**
- `ntotal` - Number of indexed vectors
- `nprobe` - Number of clusters to probe (settable)

### `from_faiss(index)`

Convert a trained FAISS index to PyTorch. Currently supports `IndexIVFFlat`.

## Performance

- Batched queries: `(batch_size, dim)` -> `(batch_size, k)`
- GPU optimized with `torch.cdist`
- TorchScript compatible for kernel fusion

## License

MIT
