from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CACHE_SCHEMA_VERSION = 1


class CacheError(ValueError):
    """Raised when a feature cache is incomplete or incompatible."""


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def cache_key(settings: Mapping[str, Any]) -> str:
    payload = {"schema_version": CACHE_SCHEMA_VERSION, "settings": settings}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_arrays(
    window_ids: Sequence[str], global_features: np.ndarray, temporal_features: np.ndarray
) -> None:
    if global_features.ndim != 2 or global_features.shape[1] != 1024:
        raise CacheError(f"global features must have shape [N,1024], got {global_features.shape}")
    if temporal_features.ndim != 3 or temporal_features.shape[1:] != (32, 1024):
        raise CacheError(f"temporal features must have shape [N,32,1024], got {temporal_features.shape}")
    if len(window_ids) != global_features.shape[0] or len(window_ids) != temporal_features.shape[0]:
        raise CacheError("window IDs and feature row counts differ")
    if len(set(window_ids)) != len(window_ids):
        raise CacheError("window IDs must be unique")


def write_feature_cache(
    root: Path | str,
    key: str,
    window_ids: Sequence[str],
    global_features: np.ndarray,
    temporal_features: np.ndarray,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write one video cache directory and return its final path."""

    global_features = np.asarray(global_features)
    temporal_features = np.asarray(temporal_features)
    window_ids = [str(value) for value in window_ids]
    _validate_arrays(window_ids, global_features, temporal_features)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / key
    if final_dir.exists():
        read_feature_cache(final_dir, expected_key=key)
        return final_dir
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
    try:
        np.save(temporary_dir / "global.npy", global_features.astype(np.float16, copy=False), allow_pickle=False)
        np.save(temporary_dir / "temporal.npy", temporal_features.astype(np.float16, copy=False), allow_pickle=False)
        payload: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "complete": True,
            "window_ids": window_ids,
            "global_shape": list(global_features.shape),
            "temporal_shape": list(temporal_features.shape),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        with (temporary_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_dir, final_dir)
    except Exception:
        for path in temporary_dir.iterdir():
            path.unlink(missing_ok=True)
        temporary_dir.rmdir()
        raise
    return final_dir


def read_feature_cache(path: Path | str, *, expected_key: str | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        with (path / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheError(f"invalid cache metadata: {path}") from exc
    if metadata.get("schema_version") != CACHE_SCHEMA_VERSION or metadata.get("complete") is not True:
        raise CacheError(f"cache is incomplete or incompatible: {path}")
    actual_key = str(metadata.get("cache_key", ""))
    if expected_key is not None and actual_key != expected_key:
        raise CacheError(f"cache key mismatch: expected {expected_key}, got {actual_key}")
    try:
        window_ids = [str(value) for value in metadata["window_ids"]]
        global_features = np.load(path / "global.npy", allow_pickle=False)
        temporal_features = np.load(path / "temporal.npy", allow_pickle=False)
    except (KeyError, OSError, ValueError) as exc:
        raise CacheError(f"cache arrays are missing or unreadable: {path}") from exc
    _validate_arrays(window_ids, global_features, temporal_features)
    if list(global_features.shape) != metadata.get("global_shape") or list(temporal_features.shape) != metadata.get(
        "temporal_shape"
    ):
        raise CacheError(f"cache metadata shape mismatch: {path}")
    return {
        "cache_key": actual_key,
        "window_ids": window_ids,
        "global_features": global_features,
        "temporal_features": temporal_features,
        "metadata": metadata.get("metadata", {}),
    }
