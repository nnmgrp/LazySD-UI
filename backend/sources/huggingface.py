"""
HuggingFaceSource — implementation using huggingface_hub SDK.

Native resume via .incomplete files.
SHA-256 available for LFS-tracked files only (lfs_sha256 field).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Disable Rust-based hf-xet and hf_transfer to prevent out-of-memory / allocation crashes on Windows
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
try:
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_DISABLE_XET = True
    _hfc.HF_HUB_ENABLE_HF_TRANSFER = False
except Exception:
    pass

from .base import ModelSource, RemoteFile, ModelMetadata, ProgressCallback


class HuggingFaceSource(ModelSource):
    source_type = "huggingface"

    def __init__(self, token: Optional[str] = None):
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise ImportError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            ) from e

        self.token = token
        self.api = HfApi(token=token)

    def list_files(self, repo_id: str, revision: str = "main") -> list[RemoteFile]:
        info = self.api.repo_info(repo_id, revision=revision, files_metadata=True)
        return [
            RemoteFile(
                path=sibling.rfilename,
                size=sibling.size or 0,
                sha256=getattr(sibling, "lfs_sha256", None),
                is_lfs=getattr(sibling, "lfs", False),
                revision=revision,
            )
            for sibling in info.siblings
        ]

    def file_exists(self, repo_id: str, file_path: str, revision: str = "main") -> bool:
        try:
            from huggingface_hub import hf_hub_url, get_hf_file_metadata
            url = hf_hub_url(repo_id=repo_id, filename=file_path, revision=revision)
            get_hf_file_metadata(url, token=self.token)
            return True
        except Exception:
            return False

    def get_metadata(self, repo_id: str, revision: str = "main") -> ModelMetadata:
        info = self.api.repo_info(repo_id, revision=revision, files_metadata=True)
        return ModelMetadata(
            repo_id=repo_id,
            description=info.description if hasattr(info, "description") else "",
            license=getattr(info, "license", "") or "",
            files_count=len(info.siblings),
            size_total=sum((s.size or 0) for s in info.siblings),
            last_modified=str(info.last_modified) if hasattr(info, "last_modified") else None,
            tags=list(getattr(info, "tags", []) or []),
        )

    def download_file(
        self,
        repo_id: str,
        file_path: str,
        local_dir: Path,
        revision: str = "main",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        from huggingface_hub import hf_hub_download

        # Ensure target directory exists
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # hf_hub_download has built-in tqdm progress; for our callback we'd need
        # to wrap it. For Phase 2 — rely on default tqdm; callback integration is Phase 4.
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=self.token,
        )
        return Path(local_path)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: Path,
        allow_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        revision: str = "main",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        from huggingface_hub import snapshot_download

        # Ensure target directory exists
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        local_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=self.token,
        )

        # Write completion marker
        self.mark_complete(Path(local_dir), repo_id)

        return Path(local_path)
