"""
ModelScopeSource — implementation using modelscope SDK.

Native resume via incomplete files.
SHA-256 available via file listing (Sha256 field).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import ModelSource, RemoteFile, ModelMetadata, ProgressCallback


class ModelScopeSource(ModelSource):
    source_type = "modelscope"

    def __init__(self, token: Optional[str] = None):
        try:
            from modelscope.hub.api import HubApi
        except ImportError as e:
            raise ImportError(
                "modelscope not installed. Run: pip install modelscope"
            ) from e

        self.token = token
        # Set token env if provided (modelscope uses env var)
        if token:
            import os
            os.environ["MODELSCOPE_API_TOKEN"] = token

        self.api = HubApi()

    def list_files(self, repo_id: str, revision: str = "master") -> list[RemoteFile]:
        """ModelScope uses 'master' as default revision (not 'main')."""
        files = self.api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True
        )
        return [
            RemoteFile(
                path=f["Path"],
                size=f.get("Size", 0) or 0,
                sha256=f.get("Sha256"),
                is_lfs=f.get("IsLFS", False),
                revision=revision,
            )
            for f in files
        ]

    def file_exists(self, repo_id: str, file_path: str, revision: str = "master") -> bool:
        files = self.list_files(repo_id, revision=revision)
        return any(f.path == file_path for f in files)

    def get_metadata(self, repo_id: str, revision: str = "master") -> ModelMetadata:
        # ModelScope API doesn't have a single repo_info call like HF.
        # We approximate by listing files and summing sizes.
        files = self.list_files(repo_id, revision=revision)
        return ModelMetadata(
            repo_id=repo_id,
            description="",
            license="",
            files_count=len(files),
            size_total=sum(f.size for f in files),
            last_modified=None,
            tags=[],
        )

    def download_file(
        self,
        repo_id: str,
        file_path: str,
        local_dir: Path,
        revision: str = "master",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        from modelscope.hub.file_download import model_file_download

        local_path = model_file_download(
            model_id=repo_id,
            file_path=file_path,
            revision=revision,
            local_dir=str(local_dir),
        )
        return Path(local_path)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: Path,
        allow_patterns: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        revision: str = "master",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        from modelscope import snapshot_download

        local_path = snapshot_download(
            model_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=revision,
            local_dir=str(local_dir),
        )

        # Write completion marker
        self.mark_complete(Path(local_dir), repo_id)

        return Path(local_path)
