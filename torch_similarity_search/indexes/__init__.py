"""Index implementations."""

from torch_similarity_search.indexes.base import BaseIndex
from torch_similarity_search.indexes.ivf_flat import IVFFlatIndex

__all__ = ["BaseIndex", "IVFFlatIndex"]
