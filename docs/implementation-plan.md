# torch-similarity-search: Implementation Plan

## Project Vision
A PyTorch-native similarity search library that converts trained FAISS indexes (IVFFlat, IVFPQ) into pure `nn.Module` models for GPU inference. Train with FAISS, serve with PyTorch.

**Why this matters:**
- FAISS requires numpy conversion overhead; this eliminates it
- Enables TorchScript/ONNX export for unified inference stacks
- Keeps vectors in GPU memory alongside embedding models
- No FAISS dependency in production containers

**Differentiation from TorchPQ:**
- TorchPQ focuses on differentiable training from scratch
- This project focuses on FAISS import → PyTorch export workflow

---

## Phase 1: Core Infrastructure

### 1.1 Project Setup
- [ ] Fix typo: rename project to `torch-similarity-search` (currently "simliarity")
- [ ] Update `pyproject.toml` with dependencies: `torch>=2.0`, `faiss-cpu` (dev), `numpy`
- [ ] Create package structure:
  ```
  torch_similarity_search/
  ├── __init__.py
  ├── indexes/
  │   ├── __init__.py
  │   ├── base.py          # Abstract base class
  │   ├── ivf_flat.py      # IVFFlat implementation
  │   └── ivf_pq.py        # IVFPQ implementation
  ├── converters/
  │   ├── __init__.py
  │   └── faiss_converter.py  # FAISS → PyTorch conversion
  └── utils/
      ├── __init__.py
      └── distance.py      # Distance computation utilities
  ```

### 1.2 Base Index Module
**File:** `torch_similarity_search/indexes/base.py`
```python
class BaseIndex(nn.Module):
    """Abstract base for all similarity search indexes."""

    @abstractmethod
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
        pass

    @abstractmethod
    def add(self, vectors: Tensor) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Vectors of shape (n, dim) or (dim,) for single vector
        """
        pass
```

---

## Phase 2: IVFFlat Implementation

### 2.1 Core IVFFlat Module
**File:** `torch_similarity_search/indexes/ivf_flat.py`

Key components:
- `centroids`: Buffer of shape `(nlist, dim)` - cluster centers
- `inverted_lists`: Dict mapping cluster_id → vector indices
- `vectors`: All indexed vectors stored as buffer

**Search algorithm:**
1. Compute distances from query to all centroids
2. Select top `nprobe` nearest centroids
3. For each selected centroid, compute distances to all vectors in its list
4. Return top-k across all probed lists

### 2.2 FAISS IVFFlat Converter
**File:** `torch_similarity_search/converters/faiss_converter.py`

Extract from FAISS:
- `index.quantizer.reconstruct_n(0, nlist)` → centroids
- `index.invlists.list_size(i)` → list sizes
- `index.invlists.get_ids(i)` → vector IDs per list
- Reconstruct vectors from inverted lists

---

## Phase 3: IVFPQ Implementation

### 3.1 Product Quantization Module
**File:** `torch_similarity_search/indexes/ivf_pq.py`

Key components:
- `centroids`: Buffer `(nlist, dim)` - coarse quantizer
- `pq_centroids`: Buffer `(M, ksub, dsub)` - PQ codebooks
  - M = number of subquantizers
  - ksub = 256 (typically)
  - dsub = dim / M
- `codes`: Buffer `(n_vectors, M)` - PQ codes (uint8)
- `coarse_assignments`: Buffer `(n_vectors,)` - cluster assignments

**Search algorithm (ADC - Asymmetric Distance Computation):**
1. Find `nprobe` nearest centroids for query
2. Compute residual = query - centroid
3. Build distance table: for each subvector, precompute distances to all 256 PQ centroids
4. For each code in probed lists, sum up distances from table
5. Return top-k

### 3.2 FAISS IVFPQ Converter
Extract from FAISS:
- Coarse centroids (same as IVFFlat)
- `index.pq.centroids` → PQ codebooks
- `index.pq.M`, `index.pq.dsub` → PQ parameters
- Codes from inverted lists

---

## Phase 4: GPU Optimization

### 4.1 Efficient Distance Computation
- Use `torch.cdist` for L2 distances (optimized for GPU)
- Batch centroid comparisons
- Fused operations where possible

### 4.2 Memory Layout
- Contiguous tensors for coalesced memory access
- Consider `torch.compile()` for kernel fusion
- Profile with `torch.profiler`

### 4.3 Batched Search
- Support batch queries: `(batch_size, dim)` → `(batch_size, k)`
- Parallelize across queries on GPU

---

## Phase 5: API & Integration

### 5.1 High-Level API
```python
import torch_similarity_search as tss

# From FAISS
model = tss.from_faiss(faiss_index)
model = model.cuda()

# Search
distances, indices = model.search(queries, k=10)

# Export
scripted = torch.jit.script(model)
torch.onnx.export(model, queries, "index.onnx")
```

### 5.2 Serialization
- `model.save("index.pt")` - PyTorch native format
- `model.load("index.pt")` - Load without FAISS dependency
- Compatible with `torch.save/load`

---

## Phase 6: Testing & Benchmarks

### 6.1 Correctness Tests
- Compare search results with FAISS (recall@k)
- Test edge cases: empty lists, single vector, etc.
- Test with various index configurations

### 6.2 Performance Benchmarks
- Latency: queries/second at various batch sizes
- Memory: GPU memory usage vs FAISS-GPU
- Compare with: FAISS-GPU, TorchPQ

---

## Implementation Order

1. **Week 1:** Project setup + IVFFlat implementation
2. **Week 2:** IVFPQ implementation + PQ distance computation
3. **Week 3:** FAISS converters + API polish
4. **Week 4:** GPU optimization + benchmarks
5. **Week 5:** Documentation + examples + release

---

## Key Files to Create

| File | Purpose |
|------|---------|
| `torch_similarity_search/__init__.py` | Public API exports |
| `torch_similarity_search/indexes/base.py` | Abstract base class |
| `torch_similarity_search/indexes/ivf_flat.py` | IVFFlat implementation |
| `torch_similarity_search/indexes/ivf_pq.py` | IVFPQ implementation |
| `torch_similarity_search/converters/faiss_converter.py` | FAISS → PyTorch |
| `tests/test_ivf_flat.py` | IVFFlat tests |
| `tests/test_ivf_pq.py` | IVFPQ tests |
| `benchmarks/bench_search.py` | Performance benchmarks |

---

## Design Decisions

1. **Distance metrics:** L2 (Euclidean) + Inner Product
2. **Dynamic updates:** Static only - rebuild index for updates (simpler, faster)
3. **Scale:** In-memory first, architecture supports future extension to larger scales
4. **Focus:** Production inference from FAISS-trained indexes (not differentiable training)
