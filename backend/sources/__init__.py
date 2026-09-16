"""ModelSource abstractions — HuggingFace + ModelScope implementations."""
from __future__ import annotations

from .base import ModelSource, RemoteFile, ModelMetadata, get_source
from .huggingface import HuggingFaceSource
from .modelscope import ModelScopeSource

__all__ = [
    "ModelSource",
    "RemoteFile",
    "ModelMetadata",
    "HuggingFaceSource",
    "ModelScopeSource",
    "get_source",
]
