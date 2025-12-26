"""Convert FAISS indexes to PyTorch modules."""

from typing import Union

import torch
import numpy as np
import faiss

from torch_similarity_search.indexes.flat import FlatIndex
from torch_similarity_search.indexes.ivf_flat import IVFFlatIndex
from torch_similarity_search.indexes.ivf_pq import IVFPQIndex


def from_faiss(index) -> Union[FlatIndex, IVFFlatIndex, IVFPQIndex]:
    """
    Convert a FAISS index to a PyTorch module.

    Supported index types:
    - faiss.IndexFlatL2, faiss.IndexFlatIP -> FlatIndex (no training required)
    - faiss.IndexIVFFlat -> IVFFlatIndex (must be trained)
    - faiss.IndexIVFPQ -> IVFPQIndex (must be trained)

    Args:
        index: A FAISS index (trained, for IVF-based indexes)

    Returns:
        A PyTorch index module with the same data

    Example:
        >>> import faiss
        >>> import torch_similarity_search as tss
        >>>
        >>> # Create FAISS index
        >>> index = faiss.IndexFlatL2(128)
        >>> index.add(vectors)
        >>>
        >>> # Convert to PyTorch
        >>> model = tss.from_faiss(index)
        >>> model = model.cuda()
        >>> distances, indices = model.search(queries, k=10)
    """
    index_type = type(index).__name__

    if index_type in ("IndexFlatL2", "IndexFlatIP"):
        return _convert_flat(index)
    elif index_type == "IndexIVFFlat":
        return _convert_ivf_flat(index)
    elif index_type == "IndexIVFPQ":
        return _convert_ivf_pq(index)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")


def _convert_flat(faiss_index) -> FlatIndex:
    """Convert a FAISS IndexFlat to PyTorch."""
    dim = faiss_index.d
    ntotal = faiss_index.ntotal

    # Determine metric type
    # FAISS: METRIC_L2 = 1, METRIC_INNER_PRODUCT = 0
    metric = "l2" if faiss_index.metric_type == 1 else "ip"

    # Create PyTorch index
    torch_index = FlatIndex(dim=dim, metric=metric)

    if ntotal == 0:
        return torch_index

    # Extract all vectors
    vectors = faiss_index.reconstruct_n(0, ntotal)
    torch_index.vectors = torch.from_numpy(vectors.copy()).float()

    return torch_index


def _convert_ivf_flat(faiss_index) -> IVFFlatIndex:
    """Convert a FAISS IndexIVFFlat to PyTorch."""
    # Extract parameters
    dim = faiss_index.d
    nlist = faiss_index.nlist
    ntotal = faiss_index.ntotal

    # Determine metric type
    # FAISS: METRIC_L2 = 1, METRIC_INNER_PRODUCT = 0
    metric = "l2" if faiss_index.metric_type == 1 else "ip"

    # Create PyTorch index
    torch_index = IVFFlatIndex(dim=dim, nlist=nlist, metric=metric)

    # Extract centroids from the quantizer
    centroids = faiss_index.quantizer.reconstruct_n(0, nlist)
    torch_index.centroids = torch.from_numpy(centroids.copy()).float()
    torch_index._is_trained = True

    if ntotal == 0:
        return torch_index

    # Extract vectors and assignments from inverted lists
    all_vectors = []
    all_indices = []
    all_assignments = []

    invlists = faiss_index.invlists

    for list_id in range(nlist):
        list_size = invlists.list_size(list_id)
        if list_size == 0:
            continue

        # Get vector IDs in this list using rev_swig_ptr
        ids_ptr = invlists.get_ids(list_id)
        ids = faiss.rev_swig_ptr(ids_ptr, list_size).copy()

        # Get codes (raw vectors) from inverted list
        # Codes are stored as uint8 bytes, need to reinterpret as float32
        code_size = faiss_index.code_size  # bytes per vector
        codes_ptr = invlists.get_codes(list_id)
        codes = faiss.rev_swig_ptr(codes_ptr, list_size * code_size)
        vectors = (
            np.frombuffer(codes.tobytes(), dtype=np.float32)
            .reshape(list_size, dim)
            .copy()
        )

        all_vectors.append(vectors)
        all_indices.append(ids)
        all_assignments.append(np.full(list_size, list_id, dtype=np.int32))

    # Concatenate all data
    if not all_vectors:
        # Inconsistent state: ntotal > 0 but all inverted lists are empty
        raise ValueError(
            "FAISS index is inconsistent: ntotal > 0 but all inverted lists are empty."
        )
    vectors = np.concatenate(all_vectors, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    assignments = np.concatenate(all_assignments, axis=0)

    # Check indices fit in int32 (max ~2.1B)
    if indices.max() > np.iinfo(np.int32).max:
        raise ValueError(
            f"FAISS index contains IDs exceeding int32 range "
            f"(max ID: {indices.max()}). int32 supports up to {np.iinfo(np.int32).max}."
        )

    # Verify recovered vector count matches expected ntotal
    if vectors.shape[0] != ntotal:
        raise ValueError(
            f"FAISS index is inconsistent: expected {ntotal} vectors, "
            f"but recovered {vectors.shape[0]} from inverted lists."
        )

    # Sort by original index to maintain order
    sort_order = np.argsort(indices)
    vectors = vectors[sort_order]
    indices = indices[sort_order]
    assignments = assignments[sort_order]

    # Set buffers (all int32 for GPU efficiency)
    torch_index.vectors = torch.from_numpy(vectors).float()
    torch_index.indices = torch.from_numpy(indices.astype(np.int32)).int()
    torch_index.assignments = torch.from_numpy(assignments).int()

    # Rebuild list structure
    torch_index._rebuild_lists()

    return torch_index


def _convert_ivf_pq(faiss_index) -> IVFPQIndex:
    """Convert a FAISS IndexIVFPQ to PyTorch."""
    # Extract parameters
    dim = faiss_index.d
    nlist = faiss_index.nlist
    ntotal = faiss_index.ntotal

    # Extract PQ parameters
    pq = faiss_index.pq
    M = pq.M  # Number of subquantizers
    nbits = pq.nbits  # Bits per code (typically 8)
    ksub = pq.ksub  # 2^nbits centroids per subquantizer
    dsub = pq.dsub  # Dimension per subquantizer

    # Determine metric type
    # FAISS: METRIC_L2 = 1, METRIC_INNER_PRODUCT = 0
    metric = "l2" if faiss_index.metric_type == 1 else "ip"

    # Create PyTorch index
    torch_index = IVFPQIndex(dim=dim, nlist=nlist, M=M, nbits=nbits, metric=metric)

    # Extract IVF centroids from the quantizer
    centroids = faiss_index.quantizer.reconstruct_n(0, nlist)
    torch_index.centroids = torch.from_numpy(centroids.copy()).float()

    # Extract PQ codebook centroids
    # FAISS stores them as (M * ksub, dsub) flattened, we need (M, ksub, dsub)
    pq_centroids = faiss.vector_to_array(pq.centroids).reshape(M, ksub, dsub).copy()
    torch_index.pq_centroids = torch.from_numpy(pq_centroids).float()

    torch_index._is_trained = True

    if ntotal == 0:
        return torch_index

    # Extract codes and assignments from inverted lists
    all_codes = []
    all_indices = []
    all_assignments = []

    invlists = faiss_index.invlists
    code_size = faiss_index.code_size  # M bytes per vector (when nbits=8)

    for list_id in range(nlist):
        list_size = invlists.list_size(list_id)
        if list_size == 0:
            continue

        # Get vector IDs in this list
        ids_ptr = invlists.get_ids(list_id)
        ids = faiss.rev_swig_ptr(ids_ptr, list_size).copy()

        # Get PQ codes from inverted list
        # Codes are stored as uint8 bytes: (list_size, M)
        codes_ptr = invlists.get_codes(list_id)
        codes_raw = faiss.rev_swig_ptr(codes_ptr, list_size * code_size)
        codes = codes_raw.reshape(list_size, M).copy()

        all_codes.append(codes)
        all_indices.append(ids)
        all_assignments.append(np.full(list_size, list_id, dtype=np.int32))

    # Concatenate all data
    if not all_codes:
        # Inconsistent state: ntotal > 0 but all inverted lists are empty
        raise ValueError(
            "FAISS index is inconsistent: ntotal > 0 but all inverted lists are empty."
        )
    codes = np.concatenate(all_codes, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    assignments = np.concatenate(all_assignments, axis=0)

    # Check indices fit in int32
    if indices.max() > np.iinfo(np.int32).max:
        raise ValueError(
            f"FAISS index contains IDs exceeding int32 range "
            f"(max ID: {indices.max()}). int32 supports up to {np.iinfo(np.int32).max}."
        )

    # Verify recovered vector count matches expected ntotal
    if codes.shape[0] != ntotal:
        raise ValueError(
            f"FAISS index is inconsistent: expected {ntotal} vectors, "
            f"but recovered {codes.shape[0]} from inverted lists."
        )

    # Sort by original index to maintain order
    sort_order = np.argsort(indices)
    codes = codes[sort_order]
    indices = indices[sort_order]
    assignments = assignments[sort_order]

    # Set buffers
    torch_index.codes = torch.from_numpy(codes).to(torch.uint8)
    torch_index.indices = torch.from_numpy(indices.astype(np.int32)).int()
    torch_index.assignments = torch.from_numpy(assignments).int()

    # Rebuild list structure
    torch_index._rebuild_lists()

    return torch_index
