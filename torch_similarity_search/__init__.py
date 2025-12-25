"""PyTorch-native similarity search library."""

from torch_similarity_search.converters.faiss_converter import from_faiss
from torch_similarity_search.indexes.flat import FlatIndex
from torch_similarity_search.indexes.ivf_flat import IVFFlatIndex

__all__ = ["FlatIndex", "IVFFlatIndex", "from_faiss"]
__version__ = "0.0.2"
