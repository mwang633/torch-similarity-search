# Changelog

## [0.0.3] - 2025-12-26

### FlatIndex + IndexFlat Converter

- **`FlatIndex`** - Brute-force exact nearest neighbor search. No training required, compares against all vectors. Best for small datasets or when exact results are needed.
- **`from_faiss()` now supports `IndexFlatL2` and `IndexFlatIP`** - Converts FAISS flat indexes to `FlatIndex`
- **FlatIndex as baseline** - Recall tests now use `FlatIndex` as ground truth for IVFFlat accuracy testing

### Cosine Similarity

- **Cosine distance metric** - FlatIndex and IVFFlatIndex now support `metric="cosine"` for cosine similarity search
- Auto-normalizes vectors during distance computation (1 - cosine_similarity)
- TorchScript compatible

### IVFPQIndex + IndexIVFPQ Converter

- **`IVFPQIndex`** - Inverted File with Product Quantization for memory-efficient approximate search
  - Combines IVF clustering with PQ compression
  - Splits vectors into M subvectors, each quantized with 2^nbits centroids
  - 64x compression (128-dim float32 → 8 bytes with M=8, nbits=8)
  - ADC (Asymmetric Distance Computation) for fast approximate search
  - Supports both L2 and inner product (IP) distance metrics
- **`from_faiss()` now supports `IndexIVFPQ`** - Converts FAISS IVFPQ indexes including PQ codebooks
- Full TorchScript support for production deployment

### Breaking Changes

- **Python 3.11+ required** - Minimum Python version increased from 3.10 to 3.11

### Improvements

- **Optimized list rebuilding** - `_rebuild_lists()` now uses `torch.bincount` for O(n) instead of O(nlist × n) complexity
- **Memory-efficient cosine distance** - Computes cosine similarity without allocating normalized tensor copies
- **FAISS converter validation** - Added verification that recovered vector count matches expected ntotal

## [0.0.2] - 2025-12-25

### Optimized IVFFlat Implementation

This release significantly improves the `IVFFlatIndex` implementation with GPU-optimized operations and cleaner architecture.

#### Performance Improvements

- **Vectorized k-means training** - Replaced Python loop with `scatter_add_` for GPU-efficient centroid updates
- **int32 tensors throughout** - All index tensors use int32 instead of int64 for better GPU memory bandwidth (supports up to 2B vectors)
- **Optimized distance computation** - `DistanceModule` uses `torch.cdist` for pairwise (2x faster than manual) and `torch.bmm` for batched operations

#### New Features

- **TorchScript-compatible `DistanceModule`** - Reusable `nn.Module` for L2 and inner product distance metrics
- **Unified input validation** - `_normalize_vectors()` in base class handles 1D/2D input normalization and dimension validation
- **Configurable default k** - `forward()` method uses configurable `k` property for Triton Inference Server compatibility

#### API Improvements

- **Input validation on all methods** - `train()`, `add()`, and `search()` now validate tensor dimensions with clear error messages
- **Consistent error messages** - All validation errors include expected vs actual dimensions
- **int32 overflow protection** - FAISS converter validates indices fit in int32 range before conversion

#### Architecture

- **`BaseIndex` abstract class** - Common interface for all index types with shared distance module and validation
- **`DistanceModule`** - Extracted distance computation to separate `nn.Module` for reusability and TorchScript export

#### Documentation

- Updated README with PyPI badges, installation instructions, and comprehensive API reference
- Added examples for FAISS conversion, from-scratch usage, TorchScript export, and embedding model integration

## [0.0.1] - 2025-12-24

Initial release with basic IVFFlat index and FAISS converter.
