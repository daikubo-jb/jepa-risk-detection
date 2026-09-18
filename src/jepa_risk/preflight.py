from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output or None


def _path_status(path_value: str | None, base_dir: Path) -> dict[str, Any]:
    if not path_value:
        return {"configured": False, "path": None, "exists": False}
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    return {
        "configured": True,
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "is_file": path.is_file() if path.exists() else False,
    }


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    )
    result: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "mps_available": mps_available,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        result["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return result


def collect_preflight(config_path: Path) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    data_config = config.get("data") or {}
    model_config = config.get("model") or {}
    execution_config = config.get("execution") or {}
    config_dir = config_path.parent

    uv_path = shutil.which("uv")
    uv_version = _command_version([uv_path, "--version"]) if uv_path else None
    ffprobe_path = shutil.which("ffprobe")
    try:
        import pyarrow  # noqa: F401
    except Exception:
        pyarrow_available = False
    else:
        pyarrow_available = True

    if execution_config.get("device", "auto") == "auto":
        torch_status = _torch_status()
        if torch_status.get("cuda_available"):
            selected_device = "cuda"
        elif torch_status.get("mps_available"):
            selected_device = "mps"
        else:
            selected_device = "cpu"
    else:
        torch_status = _torch_status()
        selected_device = execution_config["device"]

    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "path": str(config_path),
            "resolved": config,
        },
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uv_path": uv_path,
            "uv_version": uv_version,
            "cwd": os.getcwd(),
        },
        "torch": torch_status,
        "execution": {
            "requested_device": execution_config.get("device", "auto"),
            "selected_device": selected_device,
            "dtype": execution_config.get("dtype", "float32"),
        },
        "data": {
            "dataset": data_config.get("dataset"),
            "root": _path_status(data_config.get("root"), config_dir),
            "annotations": _path_status(data_config.get("annotations"), config_dir),
            "videos": _path_status(data_config.get("videos"), config_dir),
        },
        "tools": {
            "ffprobe_path": ffprobe_path,
            "ffprobe_available": ffprobe_path is not None,
            "pyarrow_available": pyarrow_available,
        },
        "model": {
            "encoder_id": model_config.get("encoder_id"),
            "predictor_source": model_config.get("predictor_source"),
            "predictor_commit": model_config.get("predictor_commit"),
            "checkpoint": _path_status(model_config.get("checkpoint"), config_dir),
        },
        "checks": {
            "uv_available": uv_path is not None,
            "python_supported": (3, 12) <= sys.version_info[:2] < (3, 14),
            "torch_available": bool(torch_status.get("available")),
            "data_root_available": bool((_path_status(data_config.get("root"), config_dir)).get("exists")),
            "ffprobe_available": ffprobe_path is not None,
            "pyarrow_available": pyarrow_available,
        },
    }


def write_preflight(config_path: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    result = collect_preflight(config_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preflight.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(output_path)
    return output_path, result
