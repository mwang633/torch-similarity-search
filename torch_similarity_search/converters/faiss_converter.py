"""Convert FAISS indexes to PyTorch modules."""

from typing import Union

import torch
import numpy as np
import faiss

from torch_similarity_search.indexes.flat import FlatIndex
from torch_similarity_search.indexes.ivf_flat import IVFFlatIndex


def from_faiss(index) -> Union[FlatIndex, IVFFlatIndex]:
    """
    Convert a trained FAISS index to a PyTorch module.

    Supported index types:
    - faiss.IndexFlatL2, faiss.IndexFlatIP -> FlatIndex
    - faiss.IndexIVFFlat -> IVFFlatIndex

    Args:
        index: A FAISS index

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

    if index_type in ("IndexFlatL2", "IndexFlatIP", "IndexFlat"):
        return _convert_flat(index)
    elif index_type == "IndexIVFFlat":
        return _convert_ivf_flat(index)
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
    vectors = np.concatenate(all_vectors, axis=0)
    indices = np.concatenate(all_indices, axis=0)
    assignments = np.concatenate(all_assignments, axis=0)

    # Check indices fit in int32 (max ~2.1B)
    if indices.max() > np.iinfo(np.int32).max:
        raise ValueError(
            f"FAISS index contains IDs exceeding int32 range "
            f"(max ID: {indices.max()}). int32 supports up to {np.iinfo(np.int32).max}."
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
