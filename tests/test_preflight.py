from pathlib import Path

import yaml

from jepa_risk.preflight import collect_preflight


def test_collect_preflight_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "phase1.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "dataset": "nexar",
                    "root": "data/nexar",
                    "annotations": "data/annotations.parquet",
                },
                "model": {"encoder_id": "test/model"},
                "execution": {"device": "cpu", "dtype": "float32"},
            }
        ),
        encoding="utf-8",
    )

    result = collect_preflight(config_path)

    assert result["config"]["path"] == str(config_path.resolve())
    assert result["data"]["root"]["path"] == str((tmp_path / "data/nexar").resolve())
    assert result["data"]["dataset"] == "nexar"
    assert result["model"]["encoder_id"] == "test/model"
    assert result["execution"]["selected_device"] == "cpu"
