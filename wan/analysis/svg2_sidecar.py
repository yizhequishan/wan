"""Read-only SVG2 K-means observer for native Wan self-attention."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .cluster_metrics import (
    adjusted_rand_index,
    adjacent_frame_js,
    background_mixing,
    centroid_continuity,
    clustering_scores,
    fragmentation_at_mass,
    tensor_summary,
)
from .config import SVG2SidecarConfig
from .entity_graph import (
    approximate_entity_graph,
    build_cluster_entity_contingency,
    latent_frame_ids,
)
from .svg2_kmeans_backend import (
    KMeansResult,
    SVG2CachedKMeans,
    stable_seed,
)
from .writer import SidecarWriter


@dataclass
class _VideoState:
    entity_ids: torch.Tensor | None
    grid_size: tuple[int, int, int] | None
    visualization: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    device_entities: dict[str, torch.Tensor] = field(default_factory=dict)
    device_frames: dict[str, torch.Tensor] = field(default_factory=dict)


def _context_value(
    context: Mapping[str, Any] | object,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _scalar(value: Any) -> float | int | str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return str(value.shape)
        return float(value.detach().cpu().item())
    if isinstance(value, (float, int, str)) or value is None:
        return value
    return str(value)


def _size_metrics(result: KMeansResult) -> dict[str, Any]:
    sizes = result.sizes[0].to(torch.float64)
    token_count = sizes.sum(dim=1).clamp_min(1)
    return {
        "empty_fraction": tensor_summary((sizes == 0).double().mean(dim=1)),
        "largest_cluster_fraction": tensor_summary(
            sizes.max(dim=1).values / token_count
        ),
        "smallest_nonempty_cluster": [
            int(row[row > 0].min().item()) if torch.any(row > 0) else 0 for row in sizes
        ],
    }


class SVG2Sidecar:
    """Observe RoPE-applied Q/K while leaving dense attention untouched."""

    def __init__(
        self,
        config: SVG2SidecarConfig,
        *,
        backend: SVG2CachedKMeans | None = None,
        writer: SidecarWriter | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or SVG2CachedKMeans(
            cq=config.cq,
            ck=config.ck,
            init_iters=config.kmeans_init_iters,
            step_iters=config.kmeans_cached_iters,
            tol=config.kmeans_tol,
        )
        self.writer = writer
        if self.writer is None and config.output_dir:
            self.writer = SidecarWriter(config.output_dir)

        self.records: list[dict[str, Any]] = []
        self._videos: dict[str, _VideoState] = {}
        self._previous_centroids: dict[tuple[str, str, int, str], torch.Tensor] = {}
        self._previous_labels: dict[tuple[str, str, int, str], torch.Tensor] = {}

    def start_video(
        self,
        video_id: str,
        *,
        entity_ids: torch.Tensor | None = None,
        grid_size: tuple[int, int, int] | None = None,
        visualization: bool | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Register aligned entity IDs before processing a denoising path."""

        video_id = str(video_id)
        self.backend.reset_video(video_id)
        self._clear_previous(video_id)

        normalized_entities: torch.Tensor | None = None
        if entity_ids is not None:
            normalized_entities = torch.as_tensor(
                entity_ids,
                dtype=torch.long,
                device="cpu",
            ).flatten()
            if torch.any(normalized_entities < -2):
                raise ValueError(
                    "entity_ids may only use -2 boundary, -1 background, "
                    "and non-negative entity IDs"
                )

        normalized_grid: tuple[int, int, int] | None = None
        if grid_size is not None:
            normalized_grid = tuple(int(value) for value in grid_size)
            if len(normalized_grid) != 3 or min(normalized_grid) <= 0:
                raise ValueError("grid_size must be a positive (F,H,W) tuple")
            if (
                normalized_entities is not None
                and math.prod(normalized_grid) != normalized_entities.numel()
            ):
                raise ValueError("entity_ids length does not match grid_size product")

        if visualization is None:
            visualization = self.config.should_save_labels(video_id)
        self._videos[video_id] = _VideoState(
            entity_ids=normalized_entities,
            grid_size=normalized_grid,
            visualization=bool(visualization),
            metadata=dict(metadata or {}),
        )

    def finish_video(self, video_id: str) -> None:
        """Release per-video centroid and persistence state."""

        video_id = str(video_id)
        self.backend.reset_video(video_id)
        self._clear_previous(video_id)
        self._videos.pop(video_id, None)

    def clear(self) -> None:
        self.backend.clear()
        self._videos.clear()
        self._previous_centroids.clear()
        self._previous_labels.clear()
        self.records.clear()

    def _clear_previous(self, video_id: str) -> None:
        for mapping in (
            self._previous_centroids,
            self._previous_labels,
        ):
            stale = [key for key in mapping if key[0] == video_id]
            for key in stale:
                del mapping[key]

    def should_observe(
        self,
        *,
        layer_idx: int,
        context: Mapping[str, Any] | object | None,
    ) -> bool:
        """Cheap preflight used before model.py copies Q/K for analysis."""

        if not self.config.enabled or context is None:
            return False
        branch = str(_context_value(context, "branch", "cond"))
        if not self.config.should_observe_branch(branch):
            return False
        if int(layer_idx) not in self.config.layers:
            return False
        step_index = int(_context_value(context, "step_index", -1))
        return (
            self.config.update_cache_every_step
            or step_index in self.config.record_steps
        )

    def _video_state(
        self,
        video_id: str,
        grid_size: tuple[int, int, int],
    ) -> _VideoState:
        state = self._videos.get(video_id)
        if state is None:
            self.start_video(video_id, grid_size=grid_size)
            return self._videos[video_id]
        if state.grid_size is None:
            state.grid_size = grid_size
        elif state.grid_size != grid_size:
            raise ValueError(
                f"Grid changed within video {video_id}: "
                f"{state.grid_size} -> {grid_size}"
            )
        return state

    def _entity_tensors(
        self,
        state: _VideoState,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if state.entity_ids is None or state.grid_size is None:
            return None
        key = str(device)
        if key not in state.device_entities:
            state.device_entities[key] = state.entity_ids.to(
                device=device,
                non_blocking=True,
            )
            state.device_frames[key] = latent_frame_ids(
                state.grid_size,
                device,
            )
        return state.device_entities[key], state.device_frames[key]

    def _persistence_metrics(
        self,
        *,
        video_id: str,
        branch: str,
        layer_idx: int,
        kind: str,
        result: KMeansResult,
        save_labels: bool,
        compute_metrics: bool,
    ) -> dict[str, Any] | None:
        key = (video_id, branch, layer_idx, kind)
        current_centroids = result.centroids[0].detach()
        previous_centroids = self._previous_centroids.get(key)
        output: dict[str, Any] | None = None
        if compute_metrics and previous_centroids is not None:
            output = centroid_continuity(
                previous_centroids,
                current_centroids,
                compute_hungarian=self.config.compute_hungarian,
            )

        if save_labels:
            current_labels = result.labels[0].detach()
            previous_labels = self._previous_labels.get(key)
            if compute_metrics and previous_labels is not None:
                if output is None:
                    output = {}
                output["label_ari"] = tensor_summary(
                    adjusted_rand_index(previous_labels, current_labels)
                )
            self._previous_labels[key] = current_labels

        self._previous_centroids[key] = current_centroids
        return output

    @torch.inference_mode()
    def observe(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_lens: torch.Tensor,
        grid_sizes: torch.Tensor,
        layer_idx: int,
        context: Mapping[str, Any] | object | None,
    ) -> None:
        """Run cached K-means and stream selected observations to disk."""

        layer_idx = int(layer_idx)
        if not self.should_observe(
            layer_idx=layer_idx,
            context=context,
        ):
            return

        branch = str(_context_value(context, "branch", "cond"))
        step_index = int(_context_value(context, "step_index", -1))
        is_record_step = step_index in self.config.record_steps

        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("sidecar expects matching q/k [B,L,H,D]")
        if q.shape[0] != 1:
            raise NotImplementedError(
                "The pilot sidecar currently requires batch size 1"
            )

        valid_len = int(seq_lens[0].detach().cpu().item())
        q = q[:, :valid_len].contiguous()
        k = k[:, :valid_len].contiguous()
        grid_size = tuple(int(value) for value in grid_sizes[0].detach().cpu().tolist())
        if math.prod(grid_size) != valid_len:
            raise ValueError(
                f"valid sequence length {valid_len} does not equal "
                f"grid product {grid_size}"
            )
        if self.config.cq > valid_len or self.config.ck > valid_len:
            raise ValueError(
                f"Cluster counts Cq={self.config.cq}, Ck={self.config.ck} "
                f"cannot exceed valid token count {valid_len}"
            )

        video_id = str(_context_value(context, "video_id", "unnamed"))
        analysis_seed = int(_context_value(context, "analysis_seed", self.config.seed))
        run_seed = stable_seed(
            analysis_seed,
            video_id,
            branch,
            layer_idx,
            self.config.cq,
            self.config.ck,
        )
        state = self._video_state(video_id, grid_size)
        cache_was_initialized = self.backend.has_cache(
            video_id,
            branch,
            layer_idx,
        )
        q_result, k_result = self.backend.run(
            q,
            k,
            video_id=video_id,
            branch=branch,
            layer_idx=layer_idx,
            seed=run_seed,
        )

        save_labels = state.visualization
        q_persistence = self._persistence_metrics(
            video_id=video_id,
            branch=branch,
            layer_idx=layer_idx,
            kind="q",
            result=q_result,
            save_labels=save_labels,
            compute_metrics=is_record_step,
        )
        k_persistence = self._persistence_metrics(
            video_id=video_id,
            branch=branch,
            layer_idx=layer_idx,
            kind="k",
            result=k_result,
            save_labels=save_labels,
            compute_metrics=is_record_step,
        )
        if not is_record_step:
            return

        record: dict[str, Any] = {
            "schema_version": 1,
            "video_id": video_id,
            "branch": branch,
            "step_index": step_index,
            "timestep": _scalar(_context_value(context, "timestep")),
            "layer_idx": layer_idx,
            "cq": self.config.cq,
            "ck": self.config.ck,
            "num_heads": q.shape[2],
            "head_dim": q.shape[3],
            "valid_tokens": valid_len,
            "grid_size": list(grid_size),
            "q_dtype": str(q.dtype),
            "cache_was_initialized": cache_was_initialized,
            "persistence_lag": (
                "previous_denoising_step"
                if self.config.update_cache_every_step
                else "previous_observed_step"
            ),
            "q_n_iters": q_result.n_iters,
            "k_n_iters": k_result.n_iters,
            "q_cluster_sizes": _size_metrics(q_result),
            "k_cluster_sizes": _size_metrics(k_result),
            "metadata": state.metadata,
        }
        arrays: dict[str, Any] = {
            "q_cluster_sizes": q_result.sizes[0],
            "k_cluster_sizes": k_result.sizes[0],
        }

        record["q_persistence"] = q_persistence
        record["k_persistence"] = k_persistence

        entity_tensors = self._entity_tensors(state, q.device)
        if entity_tensors is not None:
            entity_ids, frame_ids = entity_tensors
            if entity_ids.numel() != valid_len:
                raise ValueError(
                    f"Entity labels for {video_id} have "
                    f"{entity_ids.numel()} tokens, expected {valid_len}"
                )
            foreground = entity_ids[entity_ids >= 0]
            num_entities = int(foreground.max().item()) + 1 if foreground.numel() else 0
            num_categories = num_entities + 1
            q_contingency = build_cluster_entity_contingency(
                q_result.labels[0],
                entity_ids,
                frame_ids,
                num_clusters=self.config.cq,
                num_frames=grid_size[0],
                num_entity_categories=num_categories,
            )
            k_contingency = build_cluster_entity_contingency(
                k_result.labels[0],
                entity_ids,
                frame_ids,
                num_clusters=self.config.ck,
                num_frames=grid_size[0],
                num_entity_categories=num_categories,
            )
            arrays["q_entity_contingency"] = q_contingency
            arrays["k_entity_contingency"] = k_contingency

            for name, contingency in (
                ("q", q_contingency),
                ("k", k_contingency),
            ):
                global_contingency = contingency.sum(dim=2)
                scores = clustering_scores(global_contingency)
                record[f"{name}_entity_alignment"] = {
                    metric: tensor_summary(values) for metric, values in scores.items()
                }
                record[f"{name}_entity_alignment"].update(
                    {
                        "fragmentation_90": tensor_summary(
                            fragmentation_at_mass(global_contingency)
                        ),
                        "background_mixing": tensor_summary(
                            background_mixing(global_contingency)
                        ),
                        "adjacent_frame_js": tensor_summary(
                            adjacent_frame_js(contingency)
                        ),
                    }
                )

            graph = approximate_entity_graph(
                q_result.centroids[0],
                k_result.centroids[0],
                q_result.sizes[0],
                k_result.sizes[0],
                q_contingency,
                k_contingency,
                score_chunk_q=self.config.score_chunk_q,
            )
            arrays["ghat_same_frame"] = graph.same_frame
            if self.config.save_cross_frame_graph:
                arrays["ghat_cross_frame"] = graph.cross_frame
            record["entity_categories"] = {
                "background": 0,
                "entities": {str(entity): entity + 1 for entity in range(num_entities)},
                "boundary_input_id": -2,
            }
            row_mass = graph.cross_frame.sum(dim=(3, 4))
            valid_rows = q_contingency.sum(dim=1) > 0
            unassigned = torch.full(
                (row_mass.shape[0],),
                float("nan"),
                device=q.device,
            )
            for head in range(row_mass.shape[0]):
                head_rows = row_mass[head][valid_rows[head]]
                if head_rows.numel():
                    # Boundary keys are excluded from the entity graph, so
                    # their routing probability appears as unassigned mass.
                    unassigned[head] = (1.0 - head_rows).clamp_min(0).mean()
            record["ghat_unassigned_target_mass"] = tensor_summary(unassigned)

        if save_labels:
            label_shape = (
                q_result.labels.shape[1],
                *grid_size,
            )
            arrays["q_labels"] = (
                q_result.labels[0].reshape(label_shape).to(torch.uint16)
            )
            arrays["k_labels"] = (
                k_result.labels[0].reshape(label_shape).to(torch.uint16)
            )
            if self.config.save_centroids_for_visualizations:
                arrays["q_centroids"] = q_result.centroids[0].to(torch.float16)
                arrays["k_centroids"] = k_result.centroids[0].to(torch.float16)

        self.records.append(record)
        if self.writer is not None:
            self.writer.write_observation(
                video_id=video_id,
                branch=branch,
                step_index=step_index,
                layer_idx=layer_idx,
                record=record,
                arrays=arrays,
            )
