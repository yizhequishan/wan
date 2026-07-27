"""Run the SVG2 sidecar over a manifest-driven denoising trajectory.

The trajectory itself is deliberately supplied by a driver callable so this
entry point can be shared by Flow inversion, controlled forward-noise, and
generation smoke tests without pretending that generated frames align with
Kubric ground-truth masks.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# Allow direct execution from a source checkout without installing Wan.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.analysis import SVG2Sidecar, SVG2SidecarConfig  # noqa: E402


Driver = Callable[..., Any]


def _load_driver(spec: str) -> Driver:
    try:
        module_name, function_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError("--driver must use the form package.module:function") from exc
    module = importlib.import_module(module_name)
    driver = getattr(module, function_name)
    if not callable(driver):
        raise TypeError(f"{spec} is not callable")
    return driver


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            video_id = str(record.get("video_id", ""))
            if not video_id:
                raise ValueError(f"{path}:{line_number} is missing video_id")
            if video_id in seen:
                raise ValueError(f"Duplicate video_id: {video_id}")
            seen.add(video_id)
            record["_manifest_line"] = line_number
            records.append(record)
    return records


def _resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _load_entity_ids(
    manifest_path: Path,
    record: dict[str, Any],
) -> torch.Tensor | None:
    value = record.get("entity_ids")
    if value is None:
        return None
    path = _resolve_path(manifest_path, str(value))
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{record['video_id']}: entity_ids must be a safe .npy file")
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=False)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{path} must contain integer entity IDs")
    return torch.from_numpy(np.asarray(array, dtype=np.int64)).flatten()


def _grid_size(record: dict[str, Any]) -> tuple[int, int, int] | None:
    raw = record.get("grid_size")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{record['video_id']}: grid_size must be [F,H,W]")
    grid = tuple(int(value) for value in raw)
    if min(grid) <= 0:
        raise ValueError(f"{record['video_id']}: grid_size values must be positive")
    return grid


def _validate_record(
    manifest_path: Path,
    record: dict[str, Any],
) -> tuple[torch.Tensor | None, tuple[int, int, int] | None]:
    entity_ids = _load_entity_ids(manifest_path, record)
    grid_size = _grid_size(record)
    if entity_ids is not None and grid_size is None:
        raise ValueError(f"{record['video_id']}: grid_size is required with entity_ids")
    if (
        entity_ids is not None
        and grid_size is not None
        and entity_ids.numel() != int(np.prod(grid_size))
    ):
        raise ValueError(
            f"{record['video_id']}: entity token count "
            f"{entity_ids.numel()} != grid product {grid_size}"
        )
    return entity_ids, grid_size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cached SVG2 cluster/entity graphs from Wan."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--driver",
        help=(
            "Trajectory callable as package.module:function. It receives "
            "record=..., sidecar=..., driver_args=...."
        ),
    )
    parser.add_argument(
        "--driver-args",
        default="{}",
        help="JSON mapping forwarded to the trajectory driver.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow a manifest size different from expected_videos.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = SVG2SidecarConfig.from_yaml(args.config)
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir))

    manifest_path = args.manifest.resolve()
    records = _load_manifest(manifest_path)
    if len(records) != config.expected_videos and not args.allow_count_mismatch:
        raise ValueError(
            f"Manifest has {len(records)} videos; config expects "
            f"{config.expected_videos}. Use --allow-count-mismatch only "
            "for smoke tests."
        )

    validated: list[
        tuple[dict[str, Any], torch.Tensor | None, tuple[int, int, int] | None]
    ] = []
    for record in records:
        entity_ids, grid_size = _validate_record(manifest_path, record)
        validated.append((record, entity_ids, grid_size))

    if args.check_only:
        print(
            f"Validated {len(validated)} records for "
            f"Cq={config.cq}, Ck={config.ck}."
        )
        return 0
    if not args.driver:
        raise ValueError("--driver is required unless --check-only is used")

    driver_args = json.loads(args.driver_args)
    if not isinstance(driver_args, dict):
        raise TypeError("--driver-args must decode to a JSON object")
    driver = _load_driver(args.driver)
    sidecar = SVG2Sidecar(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for experiment execution") from exc
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            asdict(config),
            handle,
            allow_unicode=True,
            sort_keys=True,
        )

    for index, (record, entity_ids, grid_size) in enumerate(validated, start=1):
        video_id = str(record["video_id"])
        sidecar.start_video(
            video_id,
            entity_ids=entity_ids,
            grid_size=grid_size,
            metadata={
                "manifest": str(manifest_path),
                "manifest_line": record["_manifest_line"],
            },
        )
        print(f"[{index}/{len(validated)}] {video_id}", flush=True)
        records_before = len(sidecar.records)
        try:
            driver(
                record=record,
                sidecar=sidecar,
                driver_args=driver_args,
            )
        finally:
            # Idempotent if a Wan generate() wrapper already finished it.
            sidecar.finish_video(video_id)
        if len(sidecar.records) == records_before:
            raise RuntimeError(
                f"Driver produced no recorded sidecar observations for "
                f"{video_id}. Check analysis_ctx, layer selection, and "
                "record_steps."
            )

    print(f"Results written to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
