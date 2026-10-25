"""Safe, manifest-driven deletion of downloaded model files.

It also resolves VAE source/local targets and detects already-downloaded VAEs
in both the primary model repository and a separate VAE repository.

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


VAE_SUFFIXES = frozenset({".safetensors", ".pt", ".pth", ".bin", ".ckpt"})


def clean_rel(value: Any) -> str:
    """Normalize a manifest-relative path to POSIX form without a leading slash."""
    return str(value or "").replace("\\", "/").lstrip("/")


def safe_repo_rel(value: Any) -> str:
    """Return a safe repository-relative path or an empty string."""
    raw = str(value or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/"):
        return ""
    rel = clean_rel(raw)
    parts = [part for part in rel.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts) or ":" in parts[0]:
        return ""
    return "/".join(parts)


def _vae_declared_paths(engine_args: dict[str, Any]) -> list[str]:
    """VAE paths as declared in a manifest, most specific first."""
    paths: list[str] = []
    for key in ("vae_remote", "vae_file", "vae"):
        rel = safe_repo_rel(engine_args.get(key))
        if rel and rel not in paths:
            paths.append(rel)
    return paths


def _vae_local_path(engine_args: dict[str, Any], source_path: str) -> str:
    """Local VAE path used by the runtimes (relative to the primary repo)."""
    for key in ("vae", "vae_file"):
        rel = safe_repo_rel(engine_args.get(key))
        if rel:
            return rel
    return Path(source_path).name if source_path else ""


def _vae_candidate_paths(declared: list[str]) -> list[str]:
    """Repository-relative places a declared VAE may already be cached."""
    candidates: list[str] = []
    for rel in declared:
        name = Path(rel).name
        for candidate in (rel, name, f"vae/{name}"):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _infer_repo_vae(repo_dir: Path) -> str:
    """Find an already-downloaded repo-local VAE when a card does not declare one."""
    try:
        if not repo_dir.is_dir():
            return ""
    except OSError:
        return ""

    def rank(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if name.startswith("diffusion_pytorch_model"):
            priority = 0
        elif "texture" not in name:
            priority = 1
        else:
            priority = 2
        return priority, 0 if "vae" in name else 1, name

    vae_dir = repo_dir / "vae"
    try:
        if vae_dir.is_dir():
            files = [
                path for path in vae_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VAE_SUFFIXES
            ]
            if files:
                return f"vae/{min(files, key=rank).name}"
    except OSError:
        pass

    try:
        root_files = [
            path for path in repo_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in VAE_SUFFIXES
            and ("vae" in path.name.lower() or path.name.lower() == "ae.safetensors")
        ]
        if root_files:
            return min(root_files, key=rank).name
    except OSError:
        pass
    return ""


def vae_download_refs(model, primary_repo: str) -> Optional[dict[str, str]]:
    """Source and local destination for a manifest-declared VAE download.

    Runtimes download ``vae_remote`` from ``vae_repo`` but store the file in the
    primary model repository under ``vae`` (or ``vae_file``).
    """
    engine_args = getattr(model, "engine_args", {}) or {}
    declared = _vae_declared_paths(engine_args)
    if not declared:
        return None
    source_repo = str(engine_args.get("vae_repo") or primary_repo or "")
    local_repo = primary_repo or source_repo
    local_path = _vae_local_path(engine_args, declared[0])
    if not source_repo or not local_repo or not local_path:
        return None
    return {
        "source_repo": source_repo,
        "source_path": declared[0],
        "local_repo": local_repo,
        "local_path": local_path,
    }


def vae_payload_from_args(engine_args: dict[str, Any], primary_repo: str, cache) -> Optional[dict[str, Any]]:
    """Describe the VAE for a model card: source, local target and cache state.

    A declared VAE is considered installed when any of its known local layouts
    exists, either in the primary repository or in the separate VAE repository.
    When a card has no VAE declaration, an already-downloaded repo-local VAE is
    inferred so older manifests still show the component.
    """
    args = engine_args or {}
    primary_repo = str(primary_repo or "")
    declared = _vae_declared_paths(args)

    for key in ("vae_remote", "vae_file", "vae"):
        raw = str(args.get(key) or "").strip()
        if not raw:
            continue
        try:
            absolute = Path(raw)
            if absolute.is_absolute() and absolute.is_file():
                return {
                    "repo_id": str(args.get("vae_repo") or primary_repo),
                    "path": absolute.as_posix(),
                    "installed": True,
                    "local_repo_id": primary_repo,
                    "local_path": absolute.as_posix(),
                    "inferred": False,
                }
        except OSError:
            continue

    if not declared:
        if not primary_repo:
            return None
        inferred = _infer_repo_vae(cache.repo_dir_for(primary_repo))
        if not inferred:
            return None
        return {
            "repo_id": primary_repo,
            "path": inferred,
            "installed": True,
            "local_repo_id": primary_repo,
            "local_path": inferred,
            "inferred": True,
        }

    source_repo = str(args.get("vae_repo") or primary_repo)
    local_path = _vae_local_path(args, declared[0])
    source_path = declared[0]
    candidate_paths = _vae_candidate_paths(declared)

    repos: list[str] = []
    for repo in (primary_repo, source_repo):
        if repo and repo not in repos:
            repos.append(repo)

    for repo in repos:
        repo_dir = cache.repo_dir_for(repo)
        for rel in candidate_paths:
            try:
                if (repo_dir / rel).is_file():
                    return {
                        "repo_id": source_repo or repo,
                        "path": source_path,
                        "installed": True,
                        "local_repo_id": primary_repo or repo,
                        "local_path": local_path,
                        "inferred": False,
                        "resolved_repo_id": repo,
                        "resolved_path": rel,
                    }
            except OSError:
                continue

    return {
        "repo_id": source_repo or primary_repo,
        "path": source_path,
        "installed": False,
        "local_repo_id": primary_repo or source_repo,
        "local_path": local_path,
        "inferred": False,
    }


def vae_payload(model, primary_repo: str, cache) -> Optional[dict[str, Any]]:
    """Model-object wrapper around :func:`vae_payload_from_args`."""
    return vae_payload_from_args(getattr(model, "engine_args", {}) or {}, primary_repo, cache)


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
    vae_path = safe_repo_rel(
        engine_args.get("vae_remote") or engine_args.get("vae_file") or engine_args.get("vae")
    )
    if vae_repo and vae_path:
        refs.add((str(vae_repo), vae_path))
    local_vae = _vae_local_path(engine_args, vae_path)
    if primary_repo and local_vae:
        refs.add((primary_repo, local_vae))

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
    vae_path = safe_repo_rel(
        engine_args.get("vae_remote") or engine_args.get("vae_file") or engine_args.get("vae")
    )
    local_vae = _vae_local_path(engine_args, vae_path)
    if local_vae:
        add(primary_repo, local_vae)
    if vae_repo == primary_repo:
        add(vae_repo, vae_path)

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
