"""Traceable, case-level QuantiPhy evaluation.

This module intentionally reuses the released evaluation implementation for
prompt construction, video sampling, model loading, generation parameters and
MRA scoring.  It adds durable case/model-run records and browser-friendly frame
previews without changing the released ``evaluation.py`` output contract.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any, Iterable
import uuid

import numpy as np
from PIL import Image, ImageDraw
from jinja2 import Template
import torch
from vllm import SamplingParams

from . import evaluation as released


DATASET = "QuantiPhy"
DEFAULT_SPLIT = "val"
DEFAULT_OUTPUT_ROOT = released.REPOSITORY_ROOT / "data" / "evaluation"
MODEL_IDS = {
    "4b": "MirroS-Lab/Code-as-World-VL-4B",
    "9b": "MirroS-Lab/Code-as-World-VL-9B",
    "27b_base": "Qwen/Qwen3.5-27B",
}
PARSER_VERSION = "quantiphy_numeric_unit_v1"
METRIC_VERSION = "code_as_world.evaluation._mra_v1"
SELECTION_RULE = "first_success_else_last_attempt"
SHA256_CHUNK_SIZE = 8 * 1024 * 1024
UNIT_PATTERN = re.compile(
    r"^\s*([A-Za-z\u00b5\u03bc%\u00b0][A-Za-z0-9\u00b5\u03bc%\u00b0/\u00b7*^_.\-\u00b2\u00b3]*)"
)
QUESTION_UNIT_PATTERN = re.compile(
    r"\bin\s+([A-Za-z\u00b5\u03bc%\u00b0][A-Za-z0-9\u00b5\u03bc%\u00b0/\u00b7*^_.\-\s\u00b2\u00b3]*?)\s*[?.!]?$",
    flags=re.IGNORECASE,
)
UNIT_ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "second": "s",
    "seconds": "s",
    "degree": "deg",
    "degrees": "deg",
    "meters per second": "m/s",
    "metres per second": "m/s",
    "centimeters per second": "cm/s",
    "centimetres per second": "cm/s",
    "meters per second squared": "m/s^2",
    "metres per second squared": "m/s^2",
    "centimeters per second squared": "cm/s^2",
    "centimetres per second squared": "cm/s^2",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_run_id(model: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"quantiphy_val_{model}_{timestamp}"


def _slug(value: Any, fallback: str = "unknown") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return (text[:72] or fallback)


def _stable_id(prefix: str, label: Any, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{_slug(label)}_{digest}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(SHA256_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _file_descriptor(path: Path, uri: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    digest = _sha256(path)
    return {
        "uri": uri or str(path),
        "format": path.suffix.lstrip(".").lower() or "binary",
        "size_bytes": stat.st_size,
        "sha256": digest,
        "checksum": {"algorithm": "sha256", "value": digest},
    }


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(released.REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_uri()


def _dataset_uri(path: Path, root: Path | None, namespace: str) -> str:
    resolved = path.resolve()
    if root is not None:
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
            return f"dataset://{namespace}/{relative}"
        except ValueError:
            pass
    return _repo_relative(resolved)


def _git_info(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    dirty_output = run("status", "--porcelain") if commit else None
    return {
        "commit": commit,
        "dirty": bool(dirty_output) if dirty_output is not None else None,
        "remote": run("remote", "get-url", "origin") if commit else None,
    }


def _package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "transformers",
        "vllm",
        "qwen-vl-utils",
        "decord",
        "numpy",
        "jinja2",
        "Pillow",
    )
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _gpu_info() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": list(properties.major_minor)
                if hasattr(properties, "major_minor")
                else [properties.major, properties.minor],
            }
        )
    return devices


def _checkpoint_manifest(model_path: Path) -> dict[str, Any]:
    included_suffixes = {
        ".json",
        ".jinja",
        ".model",
        ".safetensors",
        ".txt",
        ".tiktoken",
    }
    files = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        if path.suffix.lower() not in included_suffixes:
            continue
        files.append(
            {
                "path": path.relative_to(model_path).as_posix(),
                "size_bytes": path.stat().st_size,
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "files": files,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "note": "Fingerprint covers checkpoint file names and sizes, not full weight contents.",
    }


def _infer_huggingface_revision(model_path: Path) -> str | None:
    cache = model_path / ".cache" / "huggingface" / "download"
    if not cache.is_dir():
        return None
    candidates: Counter[str] = Counter()
    hex_pattern = re.compile(r"\b[0-9a-f]{40}\b", flags=re.IGNORECASE)
    for path in cache.rglob("*.metadata"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        candidates.update(match.lower() for match in hex_pattern.findall(text))
    return candidates.most_common(1)[0][0] if candidates else None


def _extract_unit(text: Any) -> str | None:
    cleaned = released._strip_answer_tags(text)
    number = released.NUMBER_PATTERN.search(cleaned)
    if number is None:
        return None
    match = UNIT_PATTERN.match(cleaned[number.end() :])
    return match.group(1).rstrip(".") if match else None


def _normalise_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    return unit.casefold().replace(" ", "").replace("μ", "u").replace("µ", "u")


def _unit_from_question(question: Any) -> str | None:
    match = QUESTION_UNIT_PATTERN.search(str(question or "").strip())
    if match is None:
        return None
    raw_unit = re.sub(r"\s+", " ", match.group(1).strip()).casefold()
    return UNIT_ALIASES.get(raw_unit, raw_unit.replace(" ", ""))


def _parse_answer(raw_response: Any) -> dict[str, Any]:
    value = released._parse_prediction(raw_response)
    unit = _extract_unit(raw_response)
    if value is None:
        status = "failed"
        failure_reason = "no_numeric_value"
    else:
        status = "success"
        failure_reason = None
    return {
        "parser_version": PARSER_VERSION,
        "status": status,
        "failure_reason": failure_reason,
        "value": value,
        "unit": unit,
        "normalised_unit": _normalise_unit(unit),
    }


def _relative_error(prediction: float | None, gold: float | None) -> float | None:
    if prediction is None or gold is None or gold == 0:
        return None
    return abs(prediction - gold) / max(abs(gold), 1e-9)


def _error_tags(
    status: str,
    parsed: dict[str, Any] | None,
    gold_value: float | None,
    gold_unit: str | None,
) -> list[str]:
    if status == "preprocessing_failed":
        return ["preprocessing_error"]
    if status == "inference_failed":
        return ["inference_error"]
    if parsed is None or parsed["status"] != "success":
        return ["parse_error"]
    tags: list[str] = []
    predicted_unit = parsed.get("normalised_unit")
    normalised_gold_unit = _normalise_unit(gold_unit)
    if normalised_gold_unit and not predicted_unit:
        tags.append("unit_missing")
    elif normalised_gold_unit and predicted_unit != normalised_gold_unit:
        tags.append("unit_mismatch_unverified")
    relative_error = _relative_error(parsed.get("value"), gold_value)
    if relative_error is not None and relative_error >= 0.5:
        tags.append("large_numeric_error")
    elif relative_error is not None and relative_error > 0:
        tags.append("numeric_error")
    return tags


def _row_value(
    row: list[str], output_columns: list[tuple[str, int]], *names: str
) -> str:
    wanted = {name.casefold() for name in names}
    for name, index in output_columns:
        if name.casefold() in wanted and index < len(row):
            value = str(row[index] or "").strip()
            if value:
                return value
    return ""


def _decorate_records(
    records: list[dict[str, Any]],
    source_rows: list[list[str]],
    output_columns: list[tuple[str, int]],
) -> None:
    seen_qa_ids: Counter[str] = Counter()
    for index, (record, row) in enumerate(zip(records, source_rows)):
        video_id = str(record.get("video_id") or f"row_{index}")
        original_id = str(record.get("original_id") or index)
        record["case_id"] = _stable_id(
            "quantiphy_val", video_id, f"QuantiPhy|val|{video_id}"
        )
        qa_identity = f"QuantiPhy|val|{original_id}|{video_id}|{record.get('question', '')}"
        qa_id = _stable_id("quantiphy_val_qa", original_id, qa_identity)
        seen_qa_ids[qa_id] += 1
        if seen_qa_ids[qa_id] > 1:
            qa_id = f"{qa_id}_{seen_qa_ids[qa_id]}"
        record["qa_id"] = qa_id
        explicit_answer_unit = _row_value(
            row,
            output_columns,
            "answer_unit",
            "ground_truth_unit",
            "unit",
            "units",
        )
        record["answer_unit"] = (
            explicit_answer_unit
            or _extract_unit(record.get("answer"))
            or _unit_from_question(record.get("question"))
        )
        record["answer_unit_source"] = (
            "dataset_column"
            if explicit_answer_unit
            else "answer_text"
            if _extract_unit(record.get("answer"))
            else "question_text"
            if record["answer_unit"]
            else None
        )
        record["physical_quantity"] = _row_value(
            row,
            output_columns,
            "physical_quantity",
            "quantity",
            "question_type",
        ) or None


def _frame_to_image(frame: Any) -> Image.Image:
    array = _json_safe(frame)
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"expected a 3D video frame, got shape={array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(array[..., :3], 0, 255).astype(np.uint8), mode="RGB")


def _make_montage(images: list[tuple[Image.Image, str]], path: Path) -> None:
    if not images:
        return
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    thumb_width = 320
    label_height = 28
    prepared: list[tuple[Image.Image, str]] = []
    for image, label in images:
        thumb = image.copy()
        thumb.thumbnail((thumb_width, 240))
        prepared.append((thumb, label))
    cell_width = max(image.width for image, _ in prepared)
    cell_height = max(image.height for image, _ in prepared) + label_height
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(prepared):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + image.height + 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=88)


def _artifact_id(case_id: str, kind: str, uri: str) -> str:
    return _stable_id("artifact", kind, f"{case_id}|{kind}|{uri}")


def _save_sampled_frames(
    run_dir: Path,
    record: dict[str, Any],
    multimodal_data: dict[str, Any],
    evaluation_run_id: str,
    processing_run_id: str,
    source_video_descriptor: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = record["case_id"]
    case_root = run_dir / "media" / case_id
    manifest_path = case_root / "sampled_frames.json"
    montage_path = case_root / "sampled_frames.jpg"
    existing_created_at = None
    if manifest_path.is_file():
        try:
            existing_created_at = json.loads(
                manifest_path.read_text(encoding="utf-8")
            ).get("created_at")
        except (OSError, json.JSONDecodeError):
            existing_created_at = None
    manifest_videos: list[dict[str, Any]] = []
    montage_images: list[tuple[Image.Image, str]] = []

    for video_index, (video_frames, raw_metadata) in enumerate(multimodal_data.get("video", [])):
        metadata = dict(_json_safe(raw_metadata) or {})
        frame_indices = list(metadata.get("frames_indices") or range(len(video_frames)))
        fps = float(metadata.get("fps") or released.VIDEO_TIMESTAMP_FPS)
        video_root = case_root / f"video_{video_index:02d}"
        video_root.mkdir(parents=True, exist_ok=True)
        frame_records = []
        for sample_index, frame in enumerate(video_frames):
            source_index = int(frame_indices[sample_index]) if sample_index < len(frame_indices) else sample_index
            timestamp_seconds = source_index / fps if fps > 0 else None
            image = _frame_to_image(frame)
            frame_path = video_root / f"sample_{sample_index:02d}_frame_{source_index:06d}.png"
            if not frame_path.exists():
                image.save(frame_path, format="PNG")
            descriptor = _file_descriptor(frame_path, _repo_relative(frame_path))
            frame_records.append(
                {
                    "sample_index": sample_index,
                    "source_frame_index": source_index,
                    "timestamp_seconds": timestamp_seconds,
                    "width": image.width,
                    "height": image.height,
                    **descriptor,
                }
            )
            montage_images.append(
                (image, f"v{video_index} sample={sample_index} frame={source_index} t={timestamp_seconds:.3f}s")
            )
        manifest_videos.append(
            {
                "video_index": video_index,
                "metadata": metadata,
                "frames": frame_records,
            }
        )

    manifest = {
        "case_id": case_id,
        "evaluation_run_id": evaluation_run_id,
        "source_video": source_video_descriptor,
        "sampling_rule": {
            "requested_nframes": released.VIDEO_NFRAMES,
            "requested_video_fps": released.VIDEO_FPS,
            "timestamp_fps_fallback": released.VIDEO_TIMESTAMP_FPS,
            "min_pixels": released.MIN_PIXELS,
            "max_pixels": released.MAX_PIXELS,
            "do_sample_frames": False,
        },
        "videos": manifest_videos,
        "created_at": existing_created_at or _utc_now(),
    }
    _write_json(manifest_path, manifest)
    _make_montage(montage_images, montage_path)

    artifacts = []
    if source_video_descriptor is not None:
        uri = source_video_descriptor["uri"]
        artifacts.append(
            {
                "artifact_id": _artifact_id(case_id, "source_video", uri),
                "case_id": case_id,
                "run_id": processing_run_id,
                "evaluation_run_id": evaluation_run_id,
                "stage": "evaluation_input",
                "kind": "source_video",
                **source_video_descriptor,
                "status": "success",
                "created_at": _utc_now(),
                "producer": "source_dataset",
                "config_path": _repo_relative(run_dir / "config.json"),
                "parent_artifact_ids": [],
                "preview_uri": _repo_relative(montage_path),
            }
        )
    manifest_descriptor = _file_descriptor(manifest_path, _repo_relative(manifest_path))
    parent_ids = [artifacts[0]["artifact_id"]] if artifacts else []
    artifacts.append(
        {
            "artifact_id": _artifact_id(case_id, "sampled_frames_manifest", manifest_descriptor["uri"]),
            "case_id": case_id,
            "run_id": processing_run_id,
            "evaluation_run_id": evaluation_run_id,
            "stage": "evaluation_input",
            "kind": "sampled_frames_manifest",
            **manifest_descriptor,
            "status": "success",
            "created_at": _utc_now(),
            "producer": "code_as_world.case_evaluation",
            "config_path": _repo_relative(run_dir / "config.json"),
            "parent_artifact_ids": parent_ids,
            "preview_uri": _repo_relative(montage_path),
        }
    )
    return manifest, artifacts


def _source_video_descriptor(
    record: dict[str, Any], video_dir: Path | None, checksum_cache: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    values = released._video_values(record)
    if len(values) != 1:
        return None
    value = values[0]
    if value.startswith(("http://", "https://", "s3://", "file://")):
        return {
            "uri": value,
            "format": Path(value).suffix.lstrip("."),
            "size_bytes": None,
            "sha256": None,
            "checksum": None,
        }
    path = Path(value).resolve()
    key = str(path)
    if key not in checksum_cache:
        checksum_cache[key] = _file_descriptor(
            path,
            _dataset_uri(path, video_dir, "quantiphy-validation"),
        )
    return checksum_cache[key]


def _messages_and_prompt(
    record: dict[str, Any], processor: Any, format_prompt: Template
) -> tuple[list[dict[str, Any]], str]:
    messages = [
        {"role": "system", "content": released.SYSTEM_PROMPT},
        {"role": "user", "content": released._message_content(record, format_prompt)},
    ]
    rendered = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    return messages, rendered


def _case_record(
    record: dict[str, Any],
    video_dir: Path | None,
    source_descriptor: dict[str, Any] | None,
    frame_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_metadata = None
    if frame_manifest and frame_manifest.get("videos"):
        video_metadata = frame_manifest["videos"][0].get("metadata")
    resolution = None
    if video_metadata and video_metadata.get("width") and video_metadata.get("height"):
        resolution = {
            "width": video_metadata["width"],
            "height": video_metadata["height"],
        }
    return {
        "case_id": record["case_id"],
        "source_id": record.get("video_id"),
        "dataset": DATASET,
        "split": DEFAULT_SPLIT,
        "source_type": record.get("video_source") or "video",
        "media_uri": source_descriptor.get("uri") if source_descriptor else _dataset_uri(
            Path(released._video_values(record)[0]), video_dir, "quantiphy-validation"
        ),
        "media": source_descriptor,
        "duration_seconds": video_metadata.get("duration") if video_metadata else None,
        "fps": video_metadata.get("fps") if video_metadata else record.get("fps") or None,
        "resolution": resolution,
        "status": "success" if source_descriptor is not None else "pending",
        "failure_reason": None if source_descriptor is not None else "source_metadata_unavailable",
    }


def _qa_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "qa_id": record["qa_id"],
        "case_id": record["case_id"],
        "source_id": record.get("original_id"),
        "dataset": DATASET,
        "split": DEFAULT_SPLIT,
        "question": record.get("raw_question") or record.get("question"),
        "prior": record.get("ground_truth_prior") or None,
        "depth_info": record.get("depth_info") or None,
        "ground_truth": {
            "raw": str(record.get("answer") or ""),
            "value": released._parse_prediction(record.get("answer")),
            "unit": record.get("answer_unit"),
            "unit_source": record.get("answer_unit_source"),
            "source_category": "benchmark_annotation",
        },
        "task_type": record.get("inference_type") or None,
        "physical_quantity": record.get("physical_quantity"),
        "category": released._category(record),
        "video_type": record.get("video_type") or None,
        "status": "success",
        "failure_reason": None,
    }


def _base_prediction(
    record: dict[str, Any],
    args: argparse.Namespace,
    model_run_id: str,
    created_at: str,
) -> dict[str, Any]:
    return {
        "model_run_id": model_run_id,
        "run_id": args.processing_run_id,
        "evaluation_run_id": args.evaluation_run_id,
        "case_id": record["case_id"],
        "qa_id": record["qa_id"],
        "dataset": DATASET,
        "split": DEFAULT_SPLIT,
        "category": released._category(record),
        "model": {
            "model_key": args.model,
            "model_id": args.model_id,
            "checkpoint_uri": _repo_relative(args.model_path),
            "revision": args.resolved_model_revision,
        },
        "selection_rule": SELECTION_RULE,
        "code_commit": args.code_commit,
        "config_path": args.config_uri,
        "created_at": created_at,
    }


def _failure_prediction(
    record: dict[str, Any],
    args: argparse.Namespace,
    status: str,
    failure_reason: str,
    preprocessing_seconds: float | None = None,
    attempt_ids: list[str] | None = None,
) -> dict[str, Any]:
    created_at = _utc_now()
    model_run_id = _stable_id(
        "modelrun", record["qa_id"], f"{args.evaluation_run_id}|{record['qa_id']}|{args.model_id}"
    )
    gold_value = released._parse_prediction(record.get("answer"))
    return {
        **_base_prediction(record, args, model_run_id, created_at),
        "status": status,
        "failure_reason": failure_reason,
        "attempt_ids": attempt_ids or [],
        "selected_attempt_id": None,
        "prompt": None,
        "raw_response": None,
        "parsed": None,
        "ground_truth": {
            "raw": str(record.get("answer") or ""),
            "value": gold_value,
            "unit": record.get("answer_unit"),
            "unit_source": record.get("answer_unit_source"),
        },
        "score": {
            "metric": "MRA",
            "metric_version": METRIC_VERSION,
            "value": 0.0,
            "relative_error": None,
        },
        "error_tags": _error_tags(status, None, gold_value, record.get("answer_unit")),
        "timing": {
            "preprocessing_seconds": preprocessing_seconds,
            "batch_inference_seconds": None,
            "amortized_inference_seconds": None,
            "latency_measurement": None,
        },
        "usage": {"prompt_tokens": None, "generated_tokens": 0},
        "retry_count": max(0, len(attempt_ids or []) - 1),
    }


def _successful_prediction(
    item: dict[str, Any],
    output: Any,
    attempt_ids: list[str],
    selected_attempt_id: str,
    batch_seconds: float,
    batch_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record = item["record"]
    completion = output.outputs[0]
    response = completion.text
    parsed = _parse_answer(response)
    gold_value = released._parse_prediction(record.get("answer"))
    score = released._mra(response, str(record.get("answer") or ""))
    model_run_id = item["model_run_id"]
    return {
        **_base_prediction(record, args, model_run_id, item["created_at"]),
        "status": "success",
        "failure_reason": None,
        "attempt_ids": attempt_ids,
        "selected_attempt_id": selected_attempt_id,
        "input": {
            "source_video_uri": item["source_descriptor"].get("uri")
            if item["source_descriptor"]
            else None,
            "sampled_frames_manifest_uri": item["sampled_frames_manifest_uri"],
        },
        "prompt": {
            "system": released.SYSTEM_PROMPT,
            "user_content": item["messages"][1]["content"],
            "rendered": item["rendered_prompt"],
            "format_template_uri": _repo_relative(released.FORMAT_PROMPT),
            "chat_template_uri": _repo_relative(released.CHAT_TEMPLATE),
        },
        "raw_response": response,
        "parsed": parsed,
        "ground_truth": {
            "raw": str(record.get("answer") or ""),
            "value": gold_value,
            "unit": record.get("answer_unit"),
        },
        "score": {
            "metric": "MRA",
            "metric_version": METRIC_VERSION,
            "value": score,
            "relative_error": _relative_error(parsed.get("value"), gold_value),
        },
        "error_tags": _error_tags("success", parsed, gold_value, record.get("answer_unit")),
        "timing": {
            "preprocessing_seconds": item["preprocessing_seconds"],
            "batch_inference_seconds": batch_seconds,
            "amortized_inference_seconds": batch_seconds / max(batch_size, 1),
            "latency_measurement": "batch_wall_clock_amortized",
        },
        "usage": {
            "prompt_tokens": len(item["engine_input"]["prompt_token_ids"]),
            "generated_tokens": len(completion.token_ids or []),
        },
        "finish_reason": released._normalise_finish_reason(completion.finish_reason),
        "stop_reason": released._json_safe(completion.stop_reason),
        "retry_count": max(0, len(attempt_ids) - 1),
    }


def _run_config(args: argparse.Namespace, input_csv_descriptor: dict[str, Any]) -> dict[str, Any]:
    repository = _git_info(released.REPOSITORY_ROOT)
    benchmark_repository = _git_info(args.benchmark_repo_path) if args.benchmark_repo_path else {
        "commit": args.benchmark_commit,
        "dirty": None,
        "remote": args.benchmark_repository,
    }
    checkpoint = _checkpoint_manifest(args.model_path)
    settings = released.MODEL_SETTINGS[args.model]
    return {
        "evaluation_run_id": args.evaluation_run_id,
        "run_id": args.processing_run_id,
        "status": "running",
        "created_at": _utc_now(),
        "dataset": DATASET,
        "benchmark": {
            "name": DATASET,
            "version": args.benchmark_version,
            "split": DEFAULT_SPLIT,
            "repository": benchmark_repository,
            "data_uri": args.dataset_uri,
            "license": args.dataset_license,
            "input_csv": input_csv_descriptor,
            "source_case_count": args.source_case_count,
            "source_qa_count": args.source_qa_count,
            "selected_case_count": args.selected_case_count,
            "selected_qa_count": args.selected_qa_count,
            "local_mounts": {
                "input_csv": str(args.input_csv),
                "video_dir": str(args.video_dir) if args.video_dir else None,
            },
        },
        "model": {
            "model_key": args.model,
            "model_id": args.model_id,
            "checkpoint_uri": _repo_relative(args.model_path),
            "requested_revision": args.model_revision,
            "resolved_revision": args.resolved_model_revision,
            "checkpoint_manifest": checkpoint,
        },
        "code": {
            "repository": repository,
            "entrypoint": "python -m code_as_world.case_evaluation",
            "released_evaluation_module": "code_as_world.evaluation",
            "entrypoint_file": _file_descriptor(
                Path(__file__).resolve(), _repo_relative(Path(__file__).resolve())
            ),
            "released_evaluation_file": _file_descriptor(
                Path(released.__file__).resolve(),
                _repo_relative(Path(released.__file__).resolve()),
            ),
        },
        "prompts": {
            "system_prompt": released.SYSTEM_PROMPT,
            "format_template": _file_descriptor(
                released.FORMAT_PROMPT, _repo_relative(released.FORMAT_PROMPT)
            ),
            "chat_template": _file_descriptor(
                released.CHAT_TEMPLATE, _repo_relative(released.CHAT_TEMPLATE)
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": _gpu_info(),
        },
        "inference": {
            "backend": "vLLM",
            "dtype": "bfloat16",
            "quantization": None,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": settings["gpu_memory_utilization"],
            "max_model_len": released.MAX_MODEL_LEN,
            "max_prompt_length": released.MAX_PROMPT_LENGTH,
            "batch_size": args.batch_size,
            "max_retries": args.max_retries,
            "gdn_prefill_backend": args.gdn_prefill_backend,
            "seed": released.SEED,
            **released.SAMPLING_CONFIG,
        },
        "video_sampling": {
            "nframes": released.VIDEO_NFRAMES,
            "video_fps": released.VIDEO_FPS,
            "timestamp_fps_fallback": released.VIDEO_TIMESTAMP_FPS,
            "min_pixels": released.MIN_PIXELS,
            "max_pixels": released.MAX_PIXELS,
            "save_frames": not args.no_save_frames,
        },
        "parser": {"name": "numeric_value_and_unit", "version": PARSER_VERSION},
        "metric": {"name": "MRA", "version": METRIC_VERSION},
        "selection_rule": SELECTION_RULE,
        "comparability": {
            "released_settings_reused": [
                "prompt templates",
                "video sampling",
                "vLLM model settings",
                "sampling parameters",
                "answer parser numeric rule",
                "MRA implementation",
            ],
            "intentional_deviations": [
                f"generation is chunked with batch_size={args.batch_size}",
                "failed cases are retained instead of aborting the complete run",
                f"failed inference batches may retry up to {args.max_retries} times",
                "trace records and sampled-frame artifacts are emitted per case",
                *(
                    [
                        "vLLM GDN prefill backend is explicitly set to "
                        f"{args.gdn_prefill_backend}"
                    ]
                    if args.gdn_prefill_backend
                    else []
                ),
            ],
        },
        "sample_selection": {"limit": args.limit},
    }


def _validate_resume_config(existing: dict[str, Any], current: dict[str, Any]) -> None:
    checks = {
        "dataset": (existing.get("dataset"), current.get("dataset")),
        "model_id": (
            existing.get("model", {}).get("model_id"),
            current.get("model", {}).get("model_id"),
        ),
        "model_revision": (
            existing.get("model", {}).get("resolved_revision"),
            current.get("model", {}).get("resolved_revision"),
        ),
        "input_csv_sha256": (
            existing.get("benchmark", {}).get("input_csv", {}).get("sha256"),
            current.get("benchmark", {}).get("input_csv", {}).get("sha256"),
        ),
        "selected_qa_count": (
            existing.get("benchmark", {}).get("selected_qa_count"),
            current.get("benchmark", {}).get("selected_qa_count"),
        ),
        "checkpoint_manifest": (
            existing.get("model", {}).get("checkpoint_manifest", {}).get("manifest_sha256"),
            current.get("model", {}).get("checkpoint_manifest", {}).get("manifest_sha256"),
        ),
        "entrypoint_sha256": (
            existing.get("code", {}).get("entrypoint_file", {}).get("sha256"),
            current.get("code", {}).get("entrypoint_file", {}).get("sha256"),
        ),
        "released_evaluation_sha256": (
            existing.get("code", {}).get("released_evaluation_file", {}).get("sha256"),
            current.get("code", {}).get("released_evaluation_file", {}).get("sha256"),
        ),
        "format_template_sha256": (
            existing.get("prompts", {}).get("format_template", {}).get("sha256"),
            current.get("prompts", {}).get("format_template", {}).get("sha256"),
        ),
        "chat_template_sha256": (
            existing.get("prompts", {}).get("chat_template", {}).get("sha256"),
            current.get("prompts", {}).get("chat_template", {}).get("sha256"),
        ),
        "inference_settings": (existing.get("inference"), current.get("inference")),
        "video_sampling": (existing.get("video_sampling"), current.get("video_sampling")),
    }
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(
            "cannot resume because immutable run settings changed: " + ", ".join(mismatches)
        )


def _prediction_metrics(
    predictions: list[dict[str, Any]], total_expected: int, args: argparse.Namespace
) -> dict[str, Any]:
    status_counts = Counter(record.get("status", "unknown") for record in predictions)
    parse_counts = Counter(
        (record.get("parsed") or {}).get("status", "not_attempted") for record in predictions
    )
    error_counts = Counter(
        tag for record in predictions for tag in record.get("error_tags", [])
    )
    category_scores: dict[str, list[float]] = defaultdict(list)
    all_scores = []
    successful_scores = []
    for record in predictions:
        score = float((record.get("score") or {}).get("value") or 0.0)
        all_scores.append(score)
        if record.get("status") == "success":
            successful_scores.append(score)
        category_scores[str(record.get("category") or "unknown")].append(score)
    missing = max(0, total_expected - len(predictions))
    all_scores.extend([0.0] * missing)
    return {
        "evaluation_run_id": args.evaluation_run_id,
        "run_id": args.processing_run_id,
        "dataset": DATASET,
        "split": DEFAULT_SPLIT,
        "model_id": args.model_id,
        "code_commit": args.code_commit,
        "config_path": args.config_uri,
        "metric": "MRA",
        "metric_version": METRIC_VERSION,
        "status": "complete" if len(predictions) == total_expected else "partial",
        "counts": {
            "input": total_expected,
            "processed": len(predictions),
            "missing": missing,
            "success": status_counts.get("success", 0),
            "preprocessing_failed": status_counts.get("preprocessing_failed", 0),
            "inference_failed": status_counts.get("inference_failed", 0),
        },
        "coverage": {
            "processed_rate": len(predictions) / total_expected if total_expected else 0.0,
            "inference_success_rate": status_counts.get("success", 0) / total_expected
            if total_expected
            else 0.0,
            "parse_success_rate": parse_counts.get("success", 0) / total_expected
            if total_expected
            else 0.0,
            "denominator": "all selected benchmark QA rows",
        },
        "scores": {
            "mra_all_inputs": float(np.mean(all_scores)) if all_scores else None,
            "mra_successful_inference": float(np.mean(successful_scores))
            if successful_scores
            else None,
            "by_category_all_inputs": {
                category: float(np.mean(values))
                for category, values in sorted(category_scores.items())
            },
        },
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "error_tag_counts": dict(sorted(error_counts.items())),
        "created_at": _utc_now(),
    }


def _write_compatible_csv(
    path: Path,
    records: list[dict[str, Any]],
    source_rows: list[list[str]],
    output_columns: list[tuple[str, int]],
    predictions: list[dict[str, Any]],
    model: str,
) -> None:
    by_qa_id = {prediction["qa_id"]: prediction for prediction in predictions}
    output_records = []
    for record in records:
        prediction = by_qa_id.get(record["qa_id"], {})
        parsed = prediction.get("parsed") or {}
        output_records.append(
            {
                "response": prediction.get("raw_response") or "",
                "parsed_value": parsed.get("value"),
            }
        )
    released._write_predictions_csv(
        path,
        output_columns,
        source_rows,
        output_records,
        model,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.output_root / args.evaluation_run_id
    config_path = run_dir / "config.json"
    run_status_path = run_dir / "run.json"
    predictions_path = run_dir / "predictions.jsonl"
    attempts_path = run_dir / "attempts.jsonl"
    artifacts_path = run_dir / "artifacts.jsonl"
    cases_path = run_dir / "cases.jsonl"
    prediction_history_path = run_dir / "prediction_history.jsonl"

    records, source_rows, output_columns = released._load_csv(
        args.input_csv, args.video_dir, args.video_ext
    )
    args.source_qa_count = len(records)
    args.source_case_count = len({str(record.get("video_id")) for record in records})
    if args.limit is not None:
        records = records[: args.limit]
        source_rows = source_rows[: args.limit]
    args.selected_qa_count = len(records)
    args.selected_case_count = len({str(record.get("video_id")) for record in records})
    _decorate_records(records, source_rows, output_columns)

    input_csv_descriptor = _file_descriptor(
        args.input_csv,
        _dataset_uri(args.input_csv, args.input_csv.parent, "quantiphy"),
    )
    config = _run_config(args, input_csv_descriptor)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(
            f"run directory already exists: {run_dir}; choose a new --evaluation-run-id "
            "or pass --resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        _validate_resume_config(existing_config, config)
        config = existing_config
    else:
        _write_json(config_path, config)
    args.code_commit = config["code"]["repository"]["commit"]
    args.config_uri = _repo_relative(config_path)

    _write_json(
        run_status_path,
        {
            "evaluation_run_id": args.evaluation_run_id,
            "run_id": args.processing_run_id,
            "status": "running",
            "failure_reason": None,
            "created_at": config["created_at"],
            "updated_at": _utc_now(),
        },
    )

    existing_predictions = _read_jsonl(predictions_path)
    predictions_by_qa = {record["qa_id"]: record for record in existing_predictions}
    existing_artifacts = _read_jsonl(artifacts_path)
    artifacts_by_id = {record["artifact_id"]: record for record in existing_artifacts}
    source_checksum_cache: dict[str, dict[str, Any]] = {}
    cases_by_id = {
        record["case_id"]: record for record in _read_jsonl(cases_path)
    }
    qa_records = [_qa_record(record) for record in records]

    try:
        format_prompt = Template(
            released.FORMAT_PROMPT.read_text(encoding="utf-8").strip()
        )
        tokenizer, processor = released._load_model_tools(args.model_path)
        engine = released._build_engine(
            args.model,
            args.model_path,
            gdn_prefill_backend=args.gdn_prefill_backend,
        )
        sampling = SamplingParams(
            **released.SAMPLING_CONFIG,
            seed=released.SEED,
            detokenize=True,
            logit_bias=released._logit_bias(processor),
        )
    except BaseException as exc:
        _write_json(
            run_status_path,
            {
                "evaluation_run_id": args.evaluation_run_id,
                "run_id": args.processing_run_id,
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "created_at": config["created_at"],
                "updated_at": _utc_now(),
            },
        )
        raise

    try:
        pending_records = [
            record
            for record in records
            if record["qa_id"] not in predictions_by_qa
            or (args.retry_failed and predictions_by_qa[record["qa_id"]].get("status") != "success")
        ]
        if args.retry_failed:
            for record in pending_records:
                previous = predictions_by_qa.get(record["qa_id"])
                if previous is not None and previous.get("status") != "success":
                    _append_jsonl(
                        prediction_history_path,
                        {
                            **previous,
                            "superseded_at": _utc_now(),
                            "superseded_reason": "retry_failed",
                        },
                    )
        for offset in range(0, len(pending_records), args.batch_size):
            batch_records = pending_records[offset : offset + args.batch_size]
            prepared: list[dict[str, Any]] = []
            for record in batch_records:
                started = time.perf_counter()
                source_descriptor = None
                try:
                    source_descriptor = _source_video_descriptor(
                        record, args.video_dir, source_checksum_cache
                    )
                    multimodal_data, processor_kwargs = released._video_input(record)
                    messages, rendered_prompt = _messages_and_prompt(
                        record, processor, format_prompt
                    )
                    prompt_ids = tokenizer.encode(
                        rendered_prompt, add_special_tokens=False
                    )[: released.MAX_PROMPT_LENGTH]
                    frame_manifest = None
                    sampled_frames_manifest_uri = None
                    if not args.no_save_frames:
                        frame_manifest, new_artifacts = _save_sampled_frames(
                            run_dir,
                            record,
                            multimodal_data,
                            args.evaluation_run_id,
                            args.processing_run_id,
                            source_descriptor,
                        )
                        sampled_frames_manifest_uri = _repo_relative(
                            run_dir / "media" / record["case_id"] / "sampled_frames.json"
                        )
                        for artifact in new_artifacts:
                            if artifact["artifact_id"] not in artifacts_by_id:
                                _append_jsonl(artifacts_path, artifact)
                                artifacts_by_id[artifact["artifact_id"]] = artifact
                    cases_by_id[record["case_id"]] = _case_record(
                        record,
                        args.video_dir,
                        source_descriptor,
                        frame_manifest,
                    )
                    created_at = _utc_now()
                    model_run_id = _stable_id(
                        "modelrun",
                        record["qa_id"],
                        f"{args.evaluation_run_id}|{record['qa_id']}|{args.model_id}",
                    )
                    prepared.append(
                        {
                            "record": record,
                            "source_descriptor": source_descriptor,
                            "messages": messages,
                            "rendered_prompt": rendered_prompt,
                            "sampled_frames_manifest_uri": sampled_frames_manifest_uri,
                            "engine_input": {
                                "prompt_token_ids": prompt_ids,
                                "multi_modal_data": multimodal_data,
                                "mm_processor_kwargs": processor_kwargs,
                            },
                            "preprocessing_seconds": time.perf_counter() - started,
                            "model_run_id": model_run_id,
                            "created_at": created_at,
                        }
                    )
                except Exception as exc:  # keep failed benchmark cases visible
                    elapsed = time.perf_counter() - started
                    cases_by_id[record["case_id"]] = _case_record(
                        record, args.video_dir, source_descriptor
                    )
                    prediction = _failure_prediction(
                        record,
                        args,
                        "preprocessing_failed",
                        f"{type(exc).__name__}: {exc}",
                        preprocessing_seconds=elapsed,
                    )
                    _append_jsonl(predictions_path, prediction)
                    predictions_by_qa[record["qa_id"]] = prediction

            if not prepared:
                continue

            attempt_ids_by_qa: dict[str, list[str]] = {
                item["record"]["qa_id"]: [] for item in prepared
            }
            generated_outputs = None
            selected_attempt_ids: dict[str, str] = {}
            batch_seconds = 0.0
            last_error = "unknown inference failure"
            for attempt_number in range(1, args.max_retries + 2):
                batch_id = f"batch_{offset // args.batch_size:06d}_attempt_{attempt_number:02d}"
                started = time.perf_counter()
                try:
                    generated_outputs = engine.generate(
                        [item["engine_input"] for item in prepared], sampling
                    )
                    if len(generated_outputs) != len(prepared):
                        raise RuntimeError(
                            "vLLM returned a different number of outputs than inputs: "
                            f"{len(generated_outputs)} != {len(prepared)}"
                        )
                    batch_seconds = time.perf_counter() - started
                    for item, output in zip(prepared, generated_outputs):
                        qa_id = item["record"]["qa_id"]
                        attempt_id = f"attempt_{uuid.uuid4().hex}"
                        attempt_ids_by_qa[qa_id].append(attempt_id)
                        selected_attempt_ids[qa_id] = attempt_id
                        completion = output.outputs[0]
                        _append_jsonl(
                            attempts_path,
                            {
                                "attempt_id": attempt_id,
                                "model_run_id": item["model_run_id"],
                                "run_id": args.processing_run_id,
                                "evaluation_run_id": args.evaluation_run_id,
                                "case_id": item["record"]["case_id"],
                                "qa_id": qa_id,
                                "attempt_number": attempt_number,
                                "batch_id": batch_id,
                                "status": "success",
                                "failure_reason": None,
                                "raw_response": completion.text,
                                "parsed": _parse_answer(completion.text),
                                "batch_wall_seconds": batch_seconds,
                                "created_at": _utc_now(),
                            },
                        )
                    break
                except Exception as exc:
                    batch_seconds = time.perf_counter() - started
                    last_error = f"{type(exc).__name__}: {exc}"
                    for item in prepared:
                        qa_id = item["record"]["qa_id"]
                        attempt_id = f"attempt_{uuid.uuid4().hex}"
                        attempt_ids_by_qa[qa_id].append(attempt_id)
                        _append_jsonl(
                            attempts_path,
                            {
                                "attempt_id": attempt_id,
                                "model_run_id": item["model_run_id"],
                                "run_id": args.processing_run_id,
                                "evaluation_run_id": args.evaluation_run_id,
                                "case_id": item["record"]["case_id"],
                                "qa_id": qa_id,
                                "attempt_number": attempt_number,
                                "batch_id": batch_id,
                                "status": "failed",
                                "failure_reason": last_error,
                                "raw_response": None,
                                "parsed": None,
                                "batch_wall_seconds": batch_seconds,
                                "created_at": _utc_now(),
                            },
                        )

            if generated_outputs is None:
                for item in prepared:
                    record = item["record"]
                    prediction = _failure_prediction(
                        record,
                        args,
                        "inference_failed",
                        last_error,
                        preprocessing_seconds=item["preprocessing_seconds"],
                        attempt_ids=attempt_ids_by_qa[record["qa_id"]],
                    )
                    _append_jsonl(predictions_path, prediction)
                    predictions_by_qa[record["qa_id"]] = prediction
            else:
                for item, output in zip(prepared, generated_outputs):
                    qa_id = item["record"]["qa_id"]
                    prediction = _successful_prediction(
                        item,
                        output,
                        attempt_ids_by_qa[qa_id],
                        selected_attempt_ids[qa_id],
                        batch_seconds,
                        len(prepared),
                        args,
                    )
                    _append_jsonl(predictions_path, prediction)
                    predictions_by_qa[qa_id] = prediction
            print(
                f"[case-eval] processed {min(offset + len(batch_records), len(pending_records))}/"
                f"{len(pending_records)} pending QA rows",
                flush=True,
            )

        # Preserve benchmark order and collapse superseded failed records after --retry-failed.
        ordered_predictions = [
            predictions_by_qa[record["qa_id"]]
            for record in records
            if record["qa_id"] in predictions_by_qa
        ]
        _write_jsonl(predictions_path, ordered_predictions)
        for record in records:
            if record["case_id"] not in cases_by_id:
                descriptor = _source_video_descriptor(record, args.video_dir, source_checksum_cache)
                cases_by_id[record["case_id"]] = _case_record(
                    record, args.video_dir, descriptor
                )
        qa_ids_by_case: dict[str, list[str]] = defaultdict(list)
        for qa_record in qa_records:
            qa_ids_by_case[qa_record["case_id"]].append(qa_record["qa_id"])
        common_record_fields = {
            "evaluation_run_id": args.evaluation_run_id,
            "run_id": args.processing_run_id,
            "code_commit": config["code"]["repository"]["commit"],
            "config_path": _repo_relative(config_path),
        }
        for case_record in cases_by_id.values():
            case_record.update(common_record_fields)
            case_record["qa_ids"] = qa_ids_by_case[case_record["case_id"]]
            case_record["created_at"] = config["created_at"]
        for qa_record in qa_records:
            qa_record.update(common_record_fields)
            qa_record["created_at"] = config["created_at"]
        _write_jsonl(cases_path, cases_by_id.values())
        _write_jsonl(run_dir / "qa.jsonl", qa_records)
        metrics = _prediction_metrics(ordered_predictions, len(records), args)
        _write_json(run_dir / "metrics.json", metrics)
        _write_compatible_csv(
            run_dir / "predictions.csv",
            records,
            source_rows,
            output_columns,
            ordered_predictions,
            args.model,
        )
        aggregate_artifacts = {
            "evaluation_config": config_path,
            "case_manifest": cases_path,
            "qa_manifest": run_dir / "qa.jsonl",
            "model_predictions": predictions_path,
            "model_attempts": attempts_path,
            "evaluation_metrics": run_dir / "metrics.json",
            "compatible_predictions_csv": run_dir / "predictions.csv",
        }
        for kind, path in aggregate_artifacts.items():
            if not path.is_file():
                continue
            descriptor = _file_descriptor(path, _repo_relative(path))
            artifact_id = _artifact_id(args.evaluation_run_id, kind, descriptor["uri"])
            artifacts_by_id[artifact_id] = {
                "artifact_id": artifact_id,
                "case_id": None,
                "run_id": args.processing_run_id,
                "evaluation_run_id": args.evaluation_run_id,
                "stage": "benchmark_evaluation",
                "kind": kind,
                **descriptor,
                "status": "success",
                "created_at": _utc_now(),
                "producer": "code_as_world.case_evaluation",
                "config_path": _repo_relative(config_path),
                "parent_artifact_ids": [],
                "preview_uri": None,
            }
        _write_jsonl(run_dir / "artifacts.jsonl", artifacts_by_id.values())
        _write_json(
            run_status_path,
            {
                "evaluation_run_id": args.evaluation_run_id,
                "run_id": args.processing_run_id,
                "status": metrics["status"],
                "failure_reason": None,
                "created_at": config["created_at"],
                "updated_at": _utc_now(),
                "metrics_uri": _repo_relative(run_dir / "metrics.json"),
            },
        )
        return metrics
    except BaseException as exc:
        _write_json(
            run_status_path,
            {
                "evaluation_run_id": args.evaluation_run_id,
                "run_id": args.processing_run_id,
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "created_at": config["created_at"],
                "updated_at": _utc_now(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run traceable, case-level QuantiPhy evaluation"
    )
    parser.add_argument("model", choices=tuple(released.MODEL_SETTINGS))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--input-csv", type=Path, default=released._env_path("QUANTIPHY_INPUT_CSV"))
    parser.add_argument("--video-dir", type=Path, default=released._env_path("QUANTIPHY_VIDEO_DIR"))
    parser.add_argument("--video-ext", default=released.DEFAULT_VIDEO_EXT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evaluation-run-id")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--gdn-prefill-backend",
        choices=("flashinfer", "triton"),
        help=(
            "Override the vLLM GDN prefill backend. Use triton on CUDA runtime "
            "containers that do not include nvcc."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--no-save-frames", action="store_true")
    parser.add_argument("--benchmark-version")
    parser.add_argument("--benchmark-repository", default="https://github.com/Paulineli/QuantiPhy.git")
    parser.add_argument("--benchmark-repo-path", type=Path)
    parser.add_argument("--benchmark-commit")
    parser.add_argument("--dataset-uri", default="hf://datasets/PaulineLi/QuantiPhy-validation")
    parser.add_argument("--dataset-license")
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.input_csv is None:
        raise SystemExit(
            "pass --input-csv /path/to/quantiphy_validation.csv or set QUANTIPHY_INPUT_CSV"
        )
    args.input_csv = args.input_csv.expanduser().resolve()
    if not args.input_csv.is_file():
        raise SystemExit(f"input CSV not found: {args.input_csv}")
    args.video_dir = args.video_dir.expanduser().resolve() if args.video_dir else None
    if args.video_dir is not None and not args.video_dir.is_dir():
        raise SystemExit(f"video directory not found: {args.video_dir}")
    args.model_path = (
        args.model_path or released.REPOSITORY_ROOT / "weights" / args.model
    ).expanduser().resolve()
    if not args.model_path.is_dir():
        raise SystemExit(f"model not found: {args.model_path}")
    args.output_root = args.output_root.expanduser().resolve()
    args.benchmark_repo_path = (
        args.benchmark_repo_path.expanduser().resolve() if args.benchmark_repo_path else None
    )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    args.evaluation_run_id = args.evaluation_run_id or _default_run_id(args.model)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.evaluation_run_id) is None:
        raise SystemExit(
            "--evaluation-run-id may contain only letters, digits, underscores and hyphens"
        )
    args.model_id = args.model_id or MODEL_IDS[args.model]
    args.processing_run_id = f"run_{args.evaluation_run_id}"
    args.resolved_model_revision = args.model_revision or _infer_huggingface_revision(
        args.model_path
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _resolve_args(build_parser().parse_args(argv))
    metrics = evaluate(args)
    print(
        f"[case-eval] status={metrics['status']} "
        f"mra_all_inputs={metrics['scores']['mra_all_inputs']}",
        flush=True,
    )
    print(f"[case-eval] output={args.output_root / args.evaluation_run_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
