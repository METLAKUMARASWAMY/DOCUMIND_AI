from __future__ import annotations

import json
from pathlib import Path

_WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
}
_WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".pdparams"}
_SHARD_INDEX_NAMES = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}


def checkpoint_assets_complete(path: Path) -> bool:
    """Return true only when a checkpoint contains at least one usable weight set.

    Hugging Face creates snapshot directories and metadata before large weights finish.
    Treating that directory alone as a checkpoint makes interrupted downloads look active.
    This check follows snapshot symlinks and validates every file named by a shard index.
    """

    if path.is_file():
        return path.stat().st_size > 0
    if not path.is_dir():
        return False

    index_paths = [
        index_path for index_name in _SHARD_INDEX_NAMES for index_path in path.rglob(index_name)
    ]
    for index_path in index_paths:
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(payload.get("weight_map", {}).values())
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if shard_names and all(
            (index_path.parent / shard).is_file() and (index_path.parent / shard).stat().st_size > 0
            for shard in shard_names
        ):
            return True
    if index_paths:
        return False

    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            continue
        if candidate.name in _WEIGHT_NAMES or candidate.suffix.lower() in _WEIGHT_SUFFIXES:
            return True
    return False


def local_or_hf_cached_checkpoint(value: str) -> Path | None:
    local = Path(value).expanduser()
    if checkpoint_assets_complete(local):
        return local
    if "/" not in value:
        return None
    repository = (
        Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + value.replace("/", "--"))
    )
    try:
        revision = (repository / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snapshot = repository / "snapshots" / revision
    return snapshot if checkpoint_assets_complete(snapshot) else None
