"""Model-card VAE listing: declared VAE plus every other VAE on disk."""
from types import SimpleNamespace

from backend.model_files import vae_payload_list


class _Cache:
    def __init__(self, root):
        self.root = root

    def repo_dir_for(self, repo_id: str):
        return self.root / repo_id.replace("/", "__")


def _touch(path, payload=b"vae"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_vae_payload_list_includes_declared_and_extra_files(tmp_path):
    cache = _Cache(tmp_path)
    repo = "org/model"
    repo_dir = cache.repo_dir_for(repo)
    declared = _touch(repo_dir / "vae" / "diffusion_pytorch_model.safetensors")
    extra = _touch(repo_dir / "vae" / "texture_fix_vae.safetensors")
    _touch(repo_dir / "model.safetensors", b"weights")

    model = SimpleNamespace(engine_args={"vae": "vae/diffusion_pytorch_model.safetensors"})
    entries = vae_payload_list(model, repo, cache)

    assert [entry["path"] for entry in entries] == [
        "vae/diffusion_pytorch_model.safetensors",
        "vae/texture_fix_vae.safetensors",
    ]
    assert entries[0]["inferred"] is False
    assert entries[1]["inferred"] is True
    assert entries[0]["local_file"] == declared.as_posix()
    assert entries[1]["local_file"] == extra.as_posix()


def test_vae_payload_list_dedupes_separate_vae_repo_copy(tmp_path):
    cache = _Cache(tmp_path)
    repo = "org/model"
    vae_repo = "org/vae"
    _touch(cache.repo_dir_for(repo) / "vae" / "diffusion_pytorch_model.safetensors")
    _touch(cache.repo_dir_for(vae_repo) / "vae" / "flux2-vae.safetensors")

    model = SimpleNamespace(engine_args={
        "vae": "vae/diffusion_pytorch_model.safetensors",
        "vae_repo": vae_repo,
        "vae_remote": "vae/flux2-vae.safetensors",
    })
    entries = vae_payload_list(model, repo, cache)

    assert len(entries) == 1
    assert entries[0]["path"] == "vae/flux2-vae.safetensors"
