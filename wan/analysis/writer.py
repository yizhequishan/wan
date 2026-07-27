"""Streaming result writer for SVG2 sidecar observations."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _safe_component(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return normalized or "unnamed"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class SidecarWriter:
    """Write small metadata rows and entity-level arrays incrementally."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.output_dir / "records.jsonl"

    def write_observation(
        self,
        *,
        video_id: str,
        branch: str,
        step_index: int,
        layer_idx: int,
        record: Mapping[str, Any],
        arrays: Mapping[str, Any],
    ) -> Path:
        video_dir = self.output_dir / _safe_component(video_id)
        video_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"step_{int(step_index):03d}"
            f"_layer_{int(layer_idx):02d}"
            f"_{_safe_component(branch)}"
        )
        array_path = video_dir / f"{stem}.npz"

        numpy_arrays: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            numpy_arrays[name] = np.asarray(value)
        np.savez_compressed(array_path, **numpy_arrays)

        row = dict(record)
        row["array_file"] = str(array_path.relative_to(self.output_dir))
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        return array_path
