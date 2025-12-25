"""PyTorch-native similarity search library."""

from torch_similarity_search.indexes.ivf_flat import IVFFlatIndex
from torch_similarity_search.converters.faiss_converter import from_faiss

__all__ = ["IVFFlatIndex", "from_faiss"]
__version__ = "0.1.0"
