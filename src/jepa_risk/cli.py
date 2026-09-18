from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .manifest import assign_splits, load_manifest, manifest_bundle, write_json_atomic
from .preflight import write_preflight
from .windows import generate_all_windows
from .cache import cache_key, read_feature_cache, write_feature_cache
from .features import extract_pooled_features, load_vjepa2
from .preprocessing import load_video_window
from .probe import (
    load_cached_split,
    load_probe_model,
    save_probe_model,
    score_window_rows,
    train_probe_model,
)
from .evaluation import EvaluationError, evaluate_score_payloads
from .predictor import PredictorError, load_official_bundle, masked_prediction_error
from .report import ReportError, build_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jepa-risk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="record runtime and data readiness")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser("prepare", help="build dataset manifest, splits, and windows")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    extract = subparsers.add_parser("extract", help="extract V-JEPA2 pooled features")
    extract.add_argument("--config", type=Path, required=True)
    extract.add_argument(
        "--prepared",
        type=Path,
        required=True,
        help="directory containing manifest.json and windows.json from prepare",
    )
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--split", choices=("train", "validation", "test"), default=None)
    extract.add_argument("--selection", choices=("all", "smoke20", "pilot"), default="all")
    extract.add_argument("--limit-videos", type=int, default=None)
    extract.add_argument("--max-windows", type=int, default=None)
    extract.add_argument("--batch-size", type=int, default=4)
    extract.add_argument("--local-files-only", action="store_true")

    train_probe = subparsers.add_parser("train-probe", help="fit the frozen-feature linear probe")
    train_probe.add_argument("--config", type=Path, required=True)
    train_probe.add_argument("--prepared", type=Path, required=True)
    train_probe.add_argument("--features", type=Path, required=True)
    train_probe.add_argument("--output", type=Path, required=True)
    train_probe.add_argument("--cache-key", default=None)
    train_probe.add_argument(
        "--train-limit-videos",
        type=int,
        default=None,
        help="maximum number of videos used for fitting (defaults to --limit-videos)",
    )
    train_probe.add_argument(
        "--validation-limit-videos",
        type=int,
        default=None,
        help="maximum number of videos used for C/threshold validation (defaults to --limit-videos)",
    )
    train_probe.add_argument("--limit-videos", type=int, default=None)
    train_probe.add_argument("--train-selection", choices=("all", "smoke20", "pilot"), default="all")
    train_probe.add_argument("--validation-selection", choices=("all", "smoke20", "pilot"), default="all")

    score_probe = subparsers.add_parser("score-probe", help="score cached windows with a trained probe")
    score_probe.add_argument("--config", type=Path, required=True)
    score_probe.add_argument("--prepared", type=Path, required=True)
    score_probe.add_argument("--features", type=Path, required=True)
    score_probe.add_argument("--model", type=Path, required=True)
    score_probe.add_argument("--output", type=Path, required=True)
    score_probe.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    score_probe.add_argument("--cache-key", default=None)
    score_probe.add_argument("--limit-videos", type=int, default=None)
    score_probe.add_argument("--selection", choices=("all", "smoke20", "pilot"), default="all")

    score_prediction = subparsers.add_parser(
        "score-prediction", help="score windows with the frozen Meta V-JEPA2 predictor"
    )
    score_prediction.add_argument("--config", type=Path, required=True)
    score_prediction.add_argument("--prepared", type=Path, required=True)
    score_prediction.add_argument("--output", type=Path, required=True)
    score_prediction.add_argument("--checkpoint", type=Path, default=None)
    score_prediction.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    score_prediction.add_argument("--selection", choices=("all", "smoke20", "pilot"), default="all")
    score_prediction.add_argument("--limit-videos", type=int, default=None)
    score_prediction.add_argument("--max-windows", type=int, default=None)
    score_prediction.add_argument("--batch-size", type=int, default=1)
    score_prediction.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-video checkpoints under the output directory",
    )
    score_prediction.add_argument("--local-files-only", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="evaluate saved A/B score artifacts")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--prepared", type=Path, required=True)
    evaluate.add_argument("--probe-scores", type=Path, required=True)
    evaluate.add_argument("--prediction-scores", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    evaluate.add_argument("--selection", choices=("all", "smoke20", "pilot"), default="all")
    evaluate.add_argument("--bootstrap", type=int, default=1000)
    evaluate.add_argument("--seed", type=int, default=42)

    report = subparsers.add_parser("report", help="combine evaluation artifacts and estimate full-run cost")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--prepared", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--feature-measurement", type=Path, default=None)
    report.add_argument("--prediction-measurement", type=Path, default=None)
    report.add_argument("--probe-metrics", type=Path, default=None)
    report.add_argument("--prediction-metrics", type=Path, default=None)
    report.add_argument("--probe-scores", type=Path, default=None)
    report.add_argument("--prediction-scores", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        output_path, result = write_preflight(args.config, args.output)
        print(json.dumps({"output": str(output_path), "checks": result["checks"]}, ensure_ascii=False))
        return 0 if result["checks"]["uv_available"] and result["checks"]["python_supported"] else 2
    if args.command == "prepare":
        config_path = args.config.expanduser().resolve()
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            records = load_manifest(config, config_path.parent)
            splits = assign_splits(records)
            bundle = manifest_bundle(records, splits)
            output_dir = args.output.expanduser().resolve()
            write_json_atomic(bundle, output_dir / "manifest.json")
            windows_config = config.get("windows") or {}
            windows = generate_all_windows(
                records,
                window_sec=float(windows_config.get("window_sec", 2.0)),
                stride_sec=float(windows_config.get("stride_sec", 0.5)),
                input_frames=int(windows_config.get("input_frames", 64)),
            )
            write_json_atomic(
                {
                    "schema_version": 1,
                    "manifest_sha256": bundle["manifest_sha256"],
                    "windows": windows,
                },
                output_dir / "windows.json",
            )
        except (OSError, TypeError, ValueError, KeyError) as exc:
            print(f"prepare failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"output": str(output_dir), "videos": len(records), "windows": len(windows)}, ensure_ascii=False))
        return 0
    if args.command == "extract":
        return _extract(args)
    if args.command == "train-probe":
        return _train_probe(args)
    if args.command == "score-probe":
        return _score_probe(args)
    if args.command == "score-prediction":
        return _score_prediction(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "report":
        return _report(args)
    return 2


def _read_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


PREDICTION_CHECKPOINT_SCHEMA_VERSION = 1


def _prediction_checkpoint_path(output_dir: Path, key: str, video_id: str) -> Path:
    return output_dir / "checkpoints" / key / f"{video_id}.json"


def _read_prediction_checkpoint(
    path: Path,
    *,
    key: str,
    split: str,
    selection: str,
    video_id: str,
    expected_windows: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = _read_object(path)
    if payload.get("schema_version") != PREDICTION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"prediction checkpoint schema is incompatible: {path}")
    if payload.get("cache_key") != key or payload.get("split") != split or payload.get("selection") != selection:
        raise ValueError(f"prediction checkpoint settings do not match: {path}")
    if payload.get("video_id") != video_id:
        raise ValueError(f"prediction checkpoint video does not match: {path}")
    summary = payload.get("summary")
    scores = payload.get("scores")
    if not isinstance(summary, dict) or summary.get("status") != "scored":
        raise ValueError(f"prediction checkpoint summary is incomplete: {path}")
    if not isinstance(scores, list):
        raise ValueError(f"prediction checkpoint scores are missing: {path}")
    expected_ids = [str(window["window_id"]) for window in expected_windows]
    actual_ids = [str(row.get("window_id")) for row in scores if isinstance(row, dict)]
    if actual_ids != expected_ids or len(scores) != len(expected_windows):
        raise ValueError(f"prediction checkpoint window order does not match: {path}")
    for row, window in zip(scores, expected_windows):
        if not isinstance(row, dict):
            raise ValueError(f"prediction checkpoint contains an invalid score row: {path}")
        if row.get("video_id") != video_id or row.get("window_id") != window["window_id"]:
            raise ValueError(f"prediction checkpoint window identity does not match: {path}")
        if not bool(row.get("valid", False)):
            raise ValueError(f"prediction checkpoint contains an invalid score: {path}")
        try:
            available_at_s = float(row["available_at_s"])
            score = float(row["score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"prediction checkpoint contains a non-numeric score: {path}") from exc
        if not np.isfinite(available_at_s) or not np.isfinite(score):
            raise ValueError(f"prediction checkpoint contains a non-finite score: {path}")
        if available_at_s != float(window["available_at_s"]):
            raise ValueError(f"prediction checkpoint timestamp does not match: {path}")
    if summary.get("windows") != len(expected_windows):
        raise ValueError(f"prediction checkpoint summary window count does not match: {path}")
    return summary, scores


def _write_prediction_checkpoint(
    output_dir: Path,
    *,
    key: str,
    split: str,
    selection: str,
    video_id: str,
    summary: dict[str, object],
    scores: list[dict[str, object]],
) -> None:
    write_json_atomic(
        {
            "schema_version": PREDICTION_CHECKPOINT_SCHEMA_VERSION,
            "cache_key": key,
            "split": split,
            "selection": selection,
            "video_id": video_id,
            "summary": summary,
            "scores": scores,
        },
        _prediction_checkpoint_path(output_dir, key, video_id),
    )


def _extract(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    prepared_dir = args.prepared.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        manifest = _read_object(prepared_dir / "manifest.json")
        windows_bundle = _read_object(prepared_dir / "windows.json")
        if manifest.get("manifest_sha256") != windows_bundle.get("manifest_sha256"):
            raise ValueError("manifest and windows have different manifest_sha256")
        records = {str(record["video_id"]): record for record in manifest.get("records", [])}
        all_windows = [window for window in windows_bundle.get("windows", []) if window.get("valid", False)]
        if args.split is not None:
            all_windows = [window for window in all_windows if window.get("split") == args.split]
        windows_by_video: dict[str, list[dict]] = {}
        for window in all_windows:
            windows_by_video.setdefault(str(window["video_id"]), []).append(window)
        video_ids = sorted(windows_by_video)
        if args.selection != "all":
            selection_key = f"{args.selection}_ids"
            selection_ids = manifest.get("splits", {}).get(selection_key)
            if not isinstance(selection_ids, list):
                raise ValueError(f"prepared manifest has no {selection_key}")
            selection_order = [str(video_id) for video_id in selection_ids]
            video_ids = [video_id for video_id in selection_order if video_id in windows_by_video]
        if args.limit_videos is not None:
            if args.limit_videos <= 0:
                raise ValueError("--limit-videos must be positive")
            video_ids = video_ids[: args.limit_videos]
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if not video_ids:
            raise ValueError("no valid windows selected")
        model_config = config.get("model") or {}
        execution_config = config.get("execution") or {}
        window_config = config.get("windows") or {}
        model_id = str(model_config.get("encoder_id", "facebook/vjepa2-vitl-fpc64-256"))
        dtype = str(execution_config.get("dtype", "float32"))
        windows_sha256 = hashlib.sha256(
            json.dumps(
                windows_bundle.get("windows", []), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        settings = {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "windows_sha256": windows_sha256,
            "model_id": model_id,
            "model_revision": model_config.get("revision"),
            "preprocessing": {"letterbox": True, "target_size": 256, "processor": "official"},
            "windows": window_config,
            "pooling": {"global": "mean(t,h,w)", "temporal": "mean(h,w)"},
            "compute_dtype": dtype,
            "storage_dtype": "float16",
            "extractor_version": 3,
        }
        key = cache_key(settings)
        model, processor, device = load_vjepa2(
            model_id,
            device=str(execution_config.get("device", "auto")),
            dtype=dtype,
            local_files_only=args.local_files_only,
            revision=model_config.get("revision"),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        summary: dict[str, object] = {
            "schema_version": 1,
            "cache_key": key,
            "model_id": model_id,
            "device": str(device),
            "selected_videos": video_ids,
            "videos": [],
        }
        failures = 0
        for video_id in video_ids:
            video_windows = windows_by_video[video_id]
            if args.max_windows is not None:
                if args.max_windows <= 0:
                    raise ValueError("--max-windows must be positive")
                video_windows = video_windows[: args.max_windows]
            record = records.get(video_id)
            if record is None:
                raise ValueError(f"manifest record is missing for {video_id}")
            cache_root = output_dir / video_id
            cache_path = cache_root / key
            if cache_path.exists():
                loaded = read_feature_cache(cache_path, expected_key=key)
                expected_ids = [str(window["window_id"]) for window in video_windows]
                if loaded["window_ids"] == expected_ids:
                    summary["videos"].append(
                        {"video_id": video_id, "status": "cached", "windows": len(expected_ids), "cache": str(cache_path)}
                    )
                    continue
                raise ValueError(f"existing cache has a different window order: {cache_path}")
            started = time.perf_counter()
            global_rows = []
            temporal_rows = []
            window_ids = []
            try:
                for start in range(0, len(video_windows), args.batch_size):
                    batch_windows = video_windows[start : start + args.batch_size]
                    tensors = [
                        load_video_window(record["data_path"], window["sampled_frame_ids"], processor)
                        for window in batch_windows
                    ]
                    pooled = extract_pooled_features(model, torch.cat(tensors, dim=0), device=device)
                    for row_index, window in enumerate(batch_windows):
                        global_rows.append(pooled.global_features[row_index].cpu().numpy())
                        temporal_rows.append(pooled.temporal_features[row_index].cpu().numpy())
                        window_ids.append(str(window["window_id"]))
                cache_path = write_feature_cache(
                    cache_root,
                    key,
                    window_ids,
                    np.stack(global_rows),
                    np.stack(temporal_rows),
                    metadata={"video_id": video_id, "model_id": model_id, "device": str(device)},
                )
            except Exception as exc:
                failures += 1
                summary["videos"].append(
                    {"video_id": video_id, "status": "failed", "error": str(exc), "windows": len(video_windows)}
                )
                continue
            summary["videos"].append(
                {
                    "video_id": video_id,
                    "status": "extracted",
                    "windows": len(window_ids),
                    "batch_size": args.batch_size,
                    "seconds": round(time.perf_counter() - started, 3),
                    "cache": str(cache_path),
                }
            )
        write_json_atomic(summary, output_dir / "summary.json")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"extract failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output_dir), "cache_key": key, "videos": len(video_ids), "failures": failures}, ensure_ascii=False))
    return 2 if failures else 0


def _train_probe(args: argparse.Namespace) -> int:
    try:
        config_path = args.config.expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        train_limit = args.train_limit_videos if args.train_limit_videos is not None else args.limit_videos
        validation_limit = (
            args.validation_limit_videos
            if args.validation_limit_videos is not None
            else args.limit_videos
        )
        train_features, train_windows, _, resolved_key = load_cached_split(
            args.prepared.expanduser().resolve(),
            args.features.expanduser().resolve(),
            split="train",
            cache_key=args.cache_key,
            limit_videos=train_limit,
            selection=args.train_selection,
        )
        validation_features, validation_windows, validation_records, validation_key = load_cached_split(
            args.prepared.expanduser().resolve(),
            args.features.expanduser().resolve(),
            split="validation",
            cache_key=resolved_key,
            limit_videos=validation_limit,
            selection=args.validation_selection,
        )
        if validation_key != resolved_key:
            raise ValueError("train and validation cache keys differ")
        probe_config = config.get("probe") or {}
        model = train_probe_model(
            train_features,
            [int(window["label"]) for window in train_windows],
            validation_features,
            validation_windows,
            validation_records,
            c_candidates=probe_config.get("c_candidates", (0.01, 0.1, 1.0, 10.0)),
            stride_sec=float((config.get("windows") or {}).get("stride_sec", 0.5)),
            cache_key=resolved_key,
        )
        output_dir = args.output.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_probe_model(output_dir / "probe.pkl", model)
        write_json_atomic(
            {
                "schema_version": 1,
                "model": "standard_scaler_logistic_regression",
                "cache_key": resolved_key,
                "selected_c": model.selected_c,
                "threshold": model.threshold,
                "threshold_f1": model.threshold_f1,
                "history": model.history,
                "train_windows": len(train_windows),
                "validation_windows": len(validation_windows),
                "train_selection": args.train_selection,
                "validation_selection": args.validation_selection,
                "train_limit_videos": train_limit,
                "validation_limit_videos": validation_limit,
                "artifact": str(model_path),
            },
            output_dir / "metadata.json",
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"train-probe failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(output_dir), "selected_c": model.selected_c, "threshold": model.threshold},
            ensure_ascii=False,
        )
    )
    return 0


def _score_probe(args: argparse.Namespace) -> int:
    try:
        _ = args.config.expanduser().resolve()
        model = load_probe_model(args.model.expanduser().resolve())
        features, windows, _, resolved_key = load_cached_split(
            args.prepared.expanduser().resolve(),
            args.features.expanduser().resolve(),
            split=args.split,
            cache_key=args.cache_key or model.cache_key,
            limit_videos=args.limit_videos,
            selection=args.selection,
        )
        if model.cache_key is not None and resolved_key != model.cache_key:
            raise ValueError("feature cache key does not match probe model")
        scores = score_window_rows(model, features, windows)
        output_dir = args.output.expanduser().resolve()
        write_json_atomic(
            {"schema_version": 1, "cache_key": resolved_key, "split": args.split, "scores": scores},
            output_dir / "scores.json",
        )
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"score-probe failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(output_dir / 'scores.json'), "split": args.split, "windows": len(scores)}, ensure_ascii=False))
    return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score_prediction(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    prepared_dir = args.prepared.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        manifest = _read_object(prepared_dir / "manifest.json")
        windows_bundle = _read_object(prepared_dir / "windows.json")
        if manifest.get("manifest_sha256") != windows_bundle.get("manifest_sha256"):
            raise ValueError("manifest and windows have different manifest_sha256")
        records = {str(record["video_id"]): record for record in manifest.get("records", [])}
        windows_by_video: dict[str, list[dict]] = {}
        for window in windows_bundle.get("windows", []):
            if window.get("split") == args.split and window.get("valid", False):
                windows_by_video.setdefault(str(window["video_id"]), []).append(window)
        video_ids = sorted(windows_by_video)
        if args.selection != "all":
            selection_ids = manifest.get("splits", {}).get(f"{args.selection}_ids")
            if not isinstance(selection_ids, list):
                raise ValueError(f"prepared manifest has no {args.selection}_ids")
            video_ids = [str(video_id) for video_id in selection_ids if str(video_id) in windows_by_video]
        if args.limit_videos is not None:
            if args.limit_videos <= 0:
                raise ValueError("--limit-videos must be positive")
            video_ids = video_ids[: args.limit_videos]
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if not video_ids:
            raise ValueError("no valid windows selected")

        model_config = config.get("model") or {}
        execution_config = config.get("execution") or {}
        window_config = config.get("windows") or {}
        checkpoint_value = args.checkpoint or model_config.get("checkpoint")
        if not checkpoint_value:
            raise ValueError("predictor checkpoint is required (--checkpoint or model.checkpoint)")
        checkpoint_path = Path(str(checkpoint_value)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (config_path.parent / checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"predictor checkpoint not found: {checkpoint_path}")
        predictor_commit = model_config.get("predictor_commit")
        if not predictor_commit:
            raise ValueError("model.predictor_commit must be fixed for score-prediction")
        windows_sha256 = hashlib.sha256(
            json.dumps(windows_bundle.get("windows", []), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        settings = {
            "manifest_sha256": manifest.get("manifest_sha256"),
            "windows_sha256": windows_sha256,
            "model": "vjepa2-official-vitl",
            "model_revision": predictor_commit,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "preprocessing": {"letterbox": True, "target_size": 256, "processor": "official"},
            "windows": window_config,
            "mask": {"context_temporal_tokens": 24, "target_temporal_tokens": 8, "mask_index": 1},
            "score": "mean_abs_l1_target_layernorm",
            "dtype": str(execution_config.get("dtype", "float32")),
            "extractor_version": 1,
        }
        key = cache_key(settings)
        selected_windows_by_video: dict[str, list[dict]] = {}
        for video_id in video_ids:
            record = records.get(video_id)
            if record is None:
                raise ValueError(f"manifest record is missing for {video_id}")
            video_windows = windows_by_video[video_id]
            if args.max_windows is not None:
                if args.max_windows <= 0:
                    raise ValueError("--max-windows must be positive")
                video_windows = video_windows[: args.max_windows]
            selected_windows_by_video[video_id] = video_windows

        score_rows_by_video: dict[str, list[dict[str, object]]] = {}
        summaries_by_video: dict[str, dict[str, object]] = {}
        pending_video_ids: list[str] = []
        for video_id in video_ids:
            video_windows = selected_windows_by_video[video_id]
            checkpoint_file = _prediction_checkpoint_path(output_dir, key, video_id)
            if args.resume and checkpoint_file.is_file():
                summary, cached_scores = _read_prediction_checkpoint(
                    checkpoint_file,
                    key=key,
                    split=args.split,
                    selection=args.selection,
                    video_id=video_id,
                    expected_windows=video_windows,
                )
                summaries_by_video[video_id] = summary
                score_rows_by_video[video_id] = cached_scores
            else:
                pending_video_ids.append(video_id)

        processor = None
        bundle = None
        if pending_video_ids:
            from transformers import VJEPA2VideoProcessor

            model_id = str(model_config.get("encoder_id", "facebook/vjepa2-vitl-fpc64-256"))
            processor = VJEPA2VideoProcessor.from_pretrained(
                model_id,
                revision=model_config.get("revision"),
                local_files_only=args.local_files_only,
            )
            device_name = str(execution_config.get("device", "auto"))
            if device_name == "auto":
                from .features import resolve_device

                device = resolve_device("auto")
            else:
                device = torch.device(device_name)
            from .features import resolve_dtype

            bundle = load_official_bundle(
                checkpoint_path,
                device=device,
                dtype=resolve_dtype(str(execution_config.get("dtype", "float32"))),
            )

        for video_id in pending_video_ids:
            record = records[video_id]
            video_windows = selected_windows_by_video[video_id]
            current_scores: list[dict[str, object]] = []
            started = time.perf_counter()
            try:
                for start in range(0, len(video_windows), args.batch_size):
                    batch_windows = video_windows[start : start + args.batch_size]
                    tensors = [
                        load_video_window(record["data_path"], window["sampled_frame_ids"], processor)
                        for window in batch_windows
                    ]
                    scores = masked_prediction_error(bundle, torch.cat(tensors, dim=0)).cpu().tolist()
                    for window, score in zip(batch_windows, scores):
                        current_scores.append(
                            {
                                "video_id": window["video_id"],
                                "window_id": window["window_id"],
                                "available_at_s": float(window["available_at_s"]),
                                "score": float(score),
                                "valid": True,
                                "failure_reason": None,
                            }
                        )
            except Exception as exc:
                summaries_by_video[video_id] = {"video_id": video_id, "status": "failed", "error": str(exc)}
                score_rows_by_video[video_id] = []
                continue
            summary = {
                "video_id": video_id,
                "status": "scored",
                "windows": len(video_windows),
                "batch_size": args.batch_size,
                "seconds": round(time.perf_counter() - started, 3),
            }
            _write_prediction_checkpoint(
                output_dir,
                key=key,
                split=args.split,
                selection=args.selection,
                video_id=video_id,
                summary=summary,
                scores=current_scores,
            )
            summaries_by_video[video_id] = summary
            score_rows_by_video[video_id] = current_scores

        summaries = [summaries_by_video[video_id] for video_id in video_ids]
        score_rows = [
            row
            for video_id in video_ids
            for row in score_rows_by_video.get(video_id, [])
        ]
        failures = sum(1 for video in summaries if video.get("status") == "failed")
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            {
                "schema_version": 1,
                "cache_key": key,
                "split": args.split,
                "selection": args.selection,
                "checkpoint": str(checkpoint_path),
                "predictor_commit": predictor_commit,
                "windows": len(score_rows),
                "total_seconds": sum(
                    float(video.get("seconds", 0.0))
                    for video in summaries
                    if video.get("status") == "scored"
                ),
                "videos": summaries,
                "scores": score_rows,
            },
            output_dir / "scores.json",
        )
    except (OSError, TypeError, ValueError, KeyError, PredictorError, RuntimeError) as exc:
        print(f"score-prediction failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {"output": str(output_dir / "scores.json"), "split": args.split, "windows": len(score_rows), "failures": failures},
            ensure_ascii=False,
        )
    )
    return 2 if failures else 0


def _evaluate(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    prepared_dir = args.prepared.expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        manifest = _read_object(prepared_dir / "manifest.json")
        windows_bundle = _read_object(prepared_dir / "windows.json")
        if manifest.get("manifest_sha256") != windows_bundle.get("manifest_sha256"):
            raise ValueError("manifest and windows have different manifest_sha256")
        records = [
            record
            for record in manifest.get("records", [])
            if isinstance(record, dict) and record.get("split") == args.split
        ]
        if not records:
            raise ValueError(f"prepared manifest has no records for split={args.split}")
        probe_payload = _read_object(args.probe_scores.expanduser().resolve())
        prediction_payload = _read_object(args.prediction_scores.expanduser().resolve())
        for name, payload in (("probe", probe_payload), ("prediction", prediction_payload)):
            payload_split = payload.get("split")
            if payload_split is not None and payload_split != args.split:
                raise ValueError(f"{name} score artifact split does not match --split")
        window_config = config.get("windows") or {}
        metrics = evaluate_score_payloads(
            records,
            probe_payload,
            prediction_payload,
            stride_sec=float(window_config.get("stride_sec", 0.5)),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            split=args.split,
            selection=args.selection,
        )
        output_dir = args.output.expanduser().resolve()
        write_json_atomic(metrics, output_dir / "metrics.json")
    except (OSError, TypeError, ValueError, KeyError, EvaluationError) as exc:
        print(f"evaluate failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output_dir / "metrics.json"),
                "split": args.split,
                "videos": metrics["video_count"],
                "common_frames": metrics["common"]["frame_count"],
            },
            ensure_ascii=False,
        )
    )
    return 0


def _report(args: argparse.Namespace) -> int:
    try:
        report = build_report(
            config_path=args.config,
            prepared_dir=args.prepared,
            feature_measurement_path=args.feature_measurement,
            prediction_measurement_path=args.prediction_measurement,
            probe_metrics_path=args.probe_metrics,
            prediction_metrics_path=args.prediction_metrics,
            probe_scores_path=args.probe_scores,
            prediction_scores_path=args.prediction_scores,
        )
        json_path, markdown_path = write_report(report, args.output)
    except (OSError, TypeError, ValueError, KeyError, ReportError) as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
