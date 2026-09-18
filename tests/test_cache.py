from __future__ import annotations

import json

import numpy as np
import pytest

from jepa_risk.cache import CacheError, cache_key, read_feature_cache, write_feature_cache


def test_cache_key_changes_when_settings_change() -> None:
    base = {"model": "vjepa2", "window_sec": 2.0, "letterbox": True}
    assert cache_key(base) != cache_key({**base, "window_sec": 1.0})
    assert cache_key(base) == cache_key(dict(reversed(list(base.items()))))


def test_feature_cache_roundtrip_and_shapes(tmp_path) -> None:
    key = cache_key({"model": "test"})
    global_features = np.zeros((2, 1024), dtype=np.float32)
    temporal_features = np.ones((2, 32, 1024), dtype=np.float32)
    path = write_feature_cache(
        tmp_path,
        key,
        ["video@1", "video@2"],
        global_features,
        temporal_features,
        metadata={"video_id": "video"},
    )
    loaded = read_feature_cache(path, expected_key=key)
    assert loaded["window_ids"] == ["video@1", "video@2"]
    assert loaded["global_features"].dtype == np.float16
    assert loaded["temporal_features"].shape == (2, 32, 1024)


def test_incomplete_cache_is_rejected(tmp_path) -> None:
    key = cache_key({"model": "test"})
    path = tmp_path / key
    path.mkdir()
    (path / "metadata.json").write_text(
        json.dumps({"schema_version": 1, "cache_key": key, "complete": False}), encoding="utf-8"
    )
    with pytest.raises(CacheError, match="incomplete"):
        read_feature_cache(path, expected_key=key)
