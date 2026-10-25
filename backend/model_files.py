"""Safe, manifest-driven deletion of downloaded model files.

The cache directory is shared by many models and by repo-local encoders/VAEs.
A model must therefore never be deleted by removing a repository directory:
only exact files referenced by its manifest may be removed, and files that are
still referenced by another registry model must be kept.

This module is deliberately free of FastAPI/registry imports so it can be used
by both the HTTP API and the custom-model registry code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


def clean_rel(value: Any) -> str:
    """Normalize a manifest-relative path to POSIX form without a leading slash."""
    return str(value or "").replace("\\", "/").lstrip("/")


def model_repo_id(model) -> str:
    sources = getattr(model, "sources", None)
    primary = getattr(sources, "primary", None) if sources else None
    return str(getattr(primary, "repo_id", "") or "")


def model_file_refs(model) -> set[tuple[str, str]]:
    """Every ``(repo_id, relative_path)`` referenced by one model manifest."""
    primary_repo = model_repo_id(model)
    refs: set[tuple[str, str]] = set()

    for variant in getattr(model, "transformers", []) or []:
        repo = getattr(variant, "repo_id", None) or primary_repo
        path = clean_rel(getattr(variant, "path", ""))
        if repo and path:
            refs.add((str(repo), path))

    for encoder in getattr(model, "text_encoders", []) or []:
        source = encoder.get("source") or {}
        repo = source.get("repo_id") or encoder.get("repo") or primary_repo
        path = clean_rel(encoder.get("path") or encoder.get("file"))
        if repo and path:
            refs.add((str(repo), path))

    engine_args = getattr(model, "engine_args", {}) or {}
    vae_repo = engine_args.get("vae_repo") or primary_repo
    vae_path = clean_rel(
        engine_args.get("vae_remote") or engine_args.get("vae_file") or engine_args.get("vae")
    )
    if vae_repo and vae_path:
        refs.add((str(vae_repo), vae_path))

    for aux in engine_args.get("aux_files") or []:
        if isinstance(aux, dict):
            repo = aux.get("repo_id") or primary_repo
            path = clean_rel(aux.get("path"))
        else:
            repo, path = primary_repo, clean_rel(aux)
        if repo and path:
            refs.add((str(repo), path))

    return refs


def collect_delete_targets(
    model,
    all_models: list[Any],
    cache,
    variant_id: Optional[str] = None,
) -> tuple[list[tuple[str, str]], set[tuple[str, str]], set[str]]:
    """Collect exact files to delete for a model or one variant.

    Returns ``(targets, protected, skipped_shared)``. ``protected`` contains
    every file referenced by another model; ``skipped_shared`` contains the
    subset that would otherwise have been deleted.
    """
    primary_repo = model_repo_id(model)
    if not primary_repo:
        return [], set(), set()

    protected: set[tuple[str, str]] = set()
    for other in all_models:
        if getattr(other, "id", None) != getattr(model, "id", None):
            protected.update(model_file_refs(other))

    targets: list[tuple[str, str]] = []
    skipped_shared: set[str] = set()

    def add(repo: str, rel_value: Any, protect: bool = True) -> None:
        rel = clean_rel(rel_value)
        if not repo or not rel:
            return
        key = (repo, rel)
        if key in targets:
            return
        if protect and key in protected:
            skipped_shared.add(f"{repo}/{rel}")
            return
        targets.append(key)

    selected_variant = None
    if variant_id:
        selected_variant = next(
            (variant for variant in getattr(model, "transformers", []) or []
             if getattr(variant, "id", None) == variant_id),
            None,
        )
        if selected_variant is None:
            raise KeyError(variant_id)

    if selected_variant is not None:
        add(getattr(selected_variant, "repo_id", None) or primary_repo, selected_variant.path)

        # Keep OrbitQuant/SVDQ sidecars while any other variant still lives in
        # the same folder. They are not listed in the manifest but are required
        # to reconstruct the transformer.
        variant_repo = getattr(selected_variant, "repo_id", None) or primary_repo
        variant_dir = cache.path_for(variant_repo, selected_variant.path).parent
        siblings = [
            variant for variant in getattr(model, "transformers", []) or []
            if getattr(variant, "id", None) != getattr(selected_variant, "id", None)
            and cache.path_for(
                getattr(variant, "repo_id", None) or primary_repo, variant.path
            ).parent == variant_dir
        ]
        try:
            inside_cache = variant_dir.resolve().is_relative_to(cache.cache_dir.resolve())
        except OSError:
            inside_cache = False
        if not siblings and inside_cache and variant_dir.is_dir():
            repo_dir = cache.repo_dir_for(variant_repo)
            for sidecar in variant_dir.iterdir():
                low = sidecar.name.lower()
                if not (sidecar.is_file() and (low.startswith("orbitquant_") or low.startswith("svdq_"))):
                    continue
                try:
                    rel_sidecar = sidecar.relative_to(repo_dir).as_posix()
                except ValueError:
                    continue
                add(variant_repo, rel_sidecar)
        return targets, protected, skipped_shared

    for variant in getattr(model, "transformers", []) or []:
        add(getattr(variant, "repo_id", None) or primary_repo, variant.path)

    # Repo-local encoders/VAEs belong to this model. Shared encoders in other
    # repositories are never touched because they are protected above.
    for encoder in getattr(model, "text_encoders", []) or []:
        source = encoder.get("source") or {}
        repo = source.get("repo_id") or encoder.get("repo") or primary_repo
        if repo == primary_repo:
            add(repo, encoder.get("path") or encoder.get("file"))

    engine_args = getattr(model, "engine_args", {}) or {}
    vae_repo = engine_args.get("vae_repo") or primary_repo
    if vae_repo == primary_repo:
        add(vae_repo, engine_args.get("vae_remote") or engine_args.get("vae_file") or engine_args.get("vae"))

    for aux in engine_args.get("aux_files") or []:
        if isinstance(aux, dict):
            repo = aux.get("repo_id") or primary_repo
            path = aux.get("path")
        else:
            repo, path = primary_repo, aux
        if repo == primary_repo:
            add(repo, path)

    return targets, protected, skipped_shared


def delete_targets(cache, targets: list[tuple[str, str]]) -> dict[str, Any]:
    """Delete exact cached files (and their shards/partials), then prune empty dirs."""
    cache_root = cache.cache_dir.resolve()
    deleted_files: list[str] = []
    removed_dirs: list[str] = []
    freed_bytes = 0
    touched_dirs: set[Path] = set()

    for repo, rel in targets:
        base_path = cache.path_for(repo, rel)
        try:
            if not base_path.resolve().is_relative_to(cache_root):
                continue
        except OSError:
            continue

        candidates = [base_path]
        match = re.match(
            r"^(?P<stem>.+)-00001-of-(?P<total>\d{5})(?P<ext>\.safetensors|\.gguf)$",
            base_path.name,
            re.I,
        )
        if match:
            pattern = f"{match.group('stem')}-*-of-{match.group('total')}{match.group('ext')}"
            candidates.extend(sorted(base_path.parent.glob(pattern)))

        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                if not candidate.resolve().is_relative_to(cache_root):
                    continue
            except OSError:
                continue

            freed_bytes += candidate.stat().st_size
            candidate.unlink(missing_ok=True)
            deleted_files.append(candidate.relative_to(cache_root).as_posix())
            touched_dirs.add(candidate.parent)

            partial = candidate.with_suffix(candidate.suffix + ".incomplete")
            if partial.is_file():
                freed_bytes += partial.stat().st_size
                partial.unlink(missing_ok=True)
                deleted_files.append(partial.relative_to(cache_root).as_posix())

    # Remove only empty directories, never a directory that still has files.
    for start in sorted(touched_dirs, key=lambda path: len(path.parts), reverse=True):
        current = start
        while current != cache_root and current.is_relative_to(cache_root) and current.is_dir():
            try:
                if any(current.iterdir()):
                    break
                current.rmdir()
                removed_dirs.append(current.relative_to(cache_root).as_posix())
            except OSError:
                break
            current = current.parent

    return {
        "deleted_files": deleted_files,
        "deleted_files_count": len(deleted_files),
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 1),
        "removed_dirs": removed_dirs,
    }
