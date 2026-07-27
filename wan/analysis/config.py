"""Configuration for the SVG2 read-only attention sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple, got {type(value)!r}")
    return tuple(int(item) for item in value)


@dataclass(frozen=True)
class SVG2SidecarConfig:
    """All controls needed by the cached K-means sidecar."""

    enabled: bool = True
    observe_branch: str = "cond"
    layers: tuple[int, ...] = (0, 6, 12, 17, 23, 29)
    record_steps: tuple[int, ...] = (0, 5, 10, 20, 35, 49)
    update_cache_every_step: bool = True
    cq: int = 128
    ck: int = 512
    kmeans_init_iters: int = 50
    kmeans_cached_iters: int = 2
    kmeans_tol: float = 1e-4
    visualization_ids: tuple[str, ...] = ()
    save_full_labels: str = "visualization_only"
    save_centroids_for_visualizations: bool = True
    save_cross_frame_graph: bool = True
    compute_hungarian: bool = True
    output_dir: str = "results/svg2_pilot_c128_k512"
    expected_videos: int = 20
    score_chunk_q: int = 32
    seed: int = 20260727

    def __post_init__(self) -> None:
        if self.observe_branch not in {"cond", "uncond", "all"}:
            raise ValueError("observe_branch must be one of: cond, uncond, all")
        if not self.layers:
            raise ValueError("layers cannot be empty")
        if not self.record_steps:
            raise ValueError("record_steps cannot be empty")
        if min(self.layers) < 0 or min(self.record_steps) < 0:
            raise ValueError("layers and record_steps must be non-negative")
        if self.cq <= 0 or self.ck <= 0:
            raise ValueError("cq and ck must be positive")
        if self.kmeans_init_iters <= 0 or self.kmeans_cached_iters <= 0:
            raise ValueError("K-means iteration counts must be positive")
        if self.kmeans_tol < 0:
            raise ValueError("kmeans_tol must be non-negative")
        if self.save_full_labels not in {
            "never",
            "visualization_only",
            "always",
        }:
            raise ValueError(
                "save_full_labels must be never, visualization_only, or always"
            )
        if self.expected_videos <= 0:
            raise ValueError("expected_videos must be positive")
        if self.score_chunk_q <= 0:
            raise ValueError("score_chunk_q must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SVG2SidecarConfig":
        """Build a validated config from a YAML/JSON mapping."""

        values = dict(raw)
        for name in ("layers", "record_steps", "visualization_ids"):
            if name in values:
                if name == "visualization_ids":
                    values[name] = tuple(str(item) for item in values[name])
                else:
                    values[name] = _int_tuple(values[name], name)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SVG2SidecarConfig":
        """Load a config without making PyYAML a core Wan dependency."""

        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for SVG2 experiment configs. "
                "Install requirements-analysis.txt."
            ) from exc

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise TypeError(f"{config_path} must contain a YAML mapping")
        return cls.from_mapping(raw)

    def should_observe_branch(self, branch: str) -> bool:
        return self.observe_branch == "all" or branch == self.observe_branch

    def should_save_labels(self, video_id: str) -> bool:
        if self.save_full_labels == "always":
            return True
        if self.save_full_labels == "never":
            return False
        return video_id in self.visualization_ids
