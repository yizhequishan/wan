"""Selected-row self-attention probe for a minimal Wan trajectory pilot.

The probe is intentionally much simpler than SVG2: it selects a handful of
query tokens after RoPE, computes only ``q_selected @ K.T``, and writes the
resulting per-head ``(F, H, W)`` probability volumes. It never replaces or
modifies Wan's dense FlashAttention output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .writer import SidecarWriter


def _context_value(
    context: Mapping[str, Any] | object | None,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


@dataclass(frozen=True)
class DecodedVideoShape:
    """Decoded video dimensions in ``(frames, height, width)`` order."""

    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if min(self.frames, self.height, self.width) <= 0:
            raise ValueError("decoded video dimensions must all be positive")

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | Sequence[int],
    ) -> "DecodedVideoShape":
        if isinstance(value, Mapping):
            return cls(
                frames=int(value["frames"]),
                height=int(value["height"]),
                width=int(value["width"]),
            )
        if isinstance(value, (str, bytes)) or len(value) != 3:
            raise ValueError(
                "video_shape must be {frames,height,width} or [frames,height,width]"
            )
        return cls(*(int(item) for item in value))

    def to_dict(self) -> dict[str, int]:
        return {
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
        }


@dataclass(frozen=True)
class QueryPoint:
    """A query selected either in decoded pixels or directly in token space."""

    name: str
    frame: int | None = None
    x: float | None = None
    y: float | None = None
    token: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        has_token = self.token is not None
        has_pixel = self.frame is not None or self.x is not None or self.y is not None
        if has_token == has_pixel:
            raise ValueError(
                f"query {self.name!r} must specify exactly one of "
                "token=[f,h,w] or frame/x/y"
            )
        if has_pixel and None in (self.frame, self.x, self.y):
            raise ValueError(f"query {self.name!r} requires frame, x, and y")
        if has_token and len(self.token or ()) != 3:
            raise ValueError(f"query {self.name!r} token must be [f,h,w]")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        default_name: str,
    ) -> "QueryPoint":
        name = str(value.get("name", default_name)).strip()
        if not name:
            raise ValueError("query names may not be empty")
        if "token" in value:
            token_value = value["token"]
            if (
                isinstance(token_value, (str, bytes))
                or not isinstance(token_value, Sequence)
                or len(token_value) != 3
            ):
                raise ValueError(f"query {name!r} token must be [f,h,w]")
            return cls(
                name=name,
                token=tuple(int(item) for item in token_value),
            )
        missing = [key for key in ("frame", "x", "y") if key not in value]
        if missing:
            raise ValueError(
                f"query {name!r} is missing decoded coordinate(s): {missing}"
            )
        return cls(
            name=name,
            frame=int(value["frame"]),
            x=float(value["x"]),
            y=float(value["y"]),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.token is not None:
            return {"name": self.name, "token": list(self.token)}
        return {
            "name": self.name,
            "frame": int(self.frame),
            "x": float(self.x),
            "y": float(self.y),
        }


@dataclass(frozen=True)
class RowProbeRequest:
    """Generation replay and manually selected query points."""

    video_id: str
    prompt: str
    negative_prompt: str
    seed: int
    video_shape: DecodedVideoShape
    queries: tuple[QueryPoint, ...]
    generation: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RowProbeRequest":
        raw_queries = value.get("queries")
        if (
            isinstance(raw_queries, (str, bytes))
            or not isinstance(raw_queries, Sequence)
            or not raw_queries
        ):
            raise ValueError("query file must contain a non-empty queries list")
        queries: list[QueryPoint] = []
        names: set[str] = set()
        for index, raw_query in enumerate(raw_queries):
            if not isinstance(raw_query, Mapping):
                raise TypeError(f"queries[{index}] must be a JSON object")
            query = QueryPoint.from_mapping(
                raw_query,
                default_name=f"query_{index:02d}",
            )
            if query.name in names:
                raise ValueError(f"duplicate query name: {query.name}")
            names.add(query.name)
            queries.append(query)

        video_id = str(value.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("query file is missing video_id")
        generation = value.get("generation", {})
        if not isinstance(generation, Mapping):
            raise TypeError("generation must be a JSON object")
        return cls(
            video_id=video_id,
            prompt=str(value.get("prompt", "")),
            negative_prompt=str(value.get("negative_prompt", "")),
            seed=int(value.get("seed", 0)),
            video_shape=DecodedVideoShape.from_value(value["video_shape"]),
            queries=tuple(queries),
            generation=dict(generation),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "video_shape": self.video_shape.to_dict(),
            "queries": [query.to_dict() for query in self.queries],
            "generation": dict(self.generation),
        }


def load_row_probe_request(path: str | Path) -> RowProbeRequest:
    """Load and validate a row-probe JSON file."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return RowProbeRequest.from_mapping(value)


def decoded_frame_to_token_frame(
    frame: int,
    *,
    decoded_frames: int,
    token_frames: int,
    temporal_stride: int = 4,
) -> int:
    """Map a decoded frame to Wan's causal VAE latent-frame index."""

    if not 0 <= frame < decoded_frames:
        raise ValueError(f"decoded frame {frame} is outside [0, {decoded_frames - 1}]")
    expected_frames = 1 + temporal_stride * (token_frames - 1)
    if decoded_frames != expected_frames:
        raise ValueError(
            f"decoded/token temporal shapes are inconsistent: {decoded_frames} "
            f"decoded frames, {token_frames} token frames, stride {temporal_stride}"
        )
    return 0 if frame == 0 else (frame + temporal_stride - 1) // temporal_stride


def resolve_query_tokens(
    queries: Sequence[QueryPoint],
    *,
    decoded_shape: DecodedVideoShape,
    grid_size: tuple[int, int, int],
    temporal_stride: int = 4,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Resolve manual decoded coordinates to flattened ``(F,H,W)`` tokens."""

    token_frames, token_height, token_width = grid_size
    resolved: list[dict[str, Any]] = []
    flat_indices: list[int] = []
    for query in queries:
        if query.token is not None:
            token_f, token_h, token_w = query.token
        else:
            assert query.frame is not None
            assert query.x is not None
            assert query.y is not None
            if not 0 <= query.x < decoded_shape.width:
                raise ValueError(
                    f"query {query.name!r} x={query.x} is outside decoded width "
                    f"{decoded_shape.width}"
                )
            if not 0 <= query.y < decoded_shape.height:
                raise ValueError(
                    f"query {query.name!r} y={query.y} is outside decoded height "
                    f"{decoded_shape.height}"
                )
            token_f = decoded_frame_to_token_frame(
                query.frame,
                decoded_frames=decoded_shape.frames,
                token_frames=token_frames,
                temporal_stride=temporal_stride,
            )
            token_h = min(
                int(query.y * token_height / decoded_shape.height),
                token_height - 1,
            )
            token_w = min(
                int(query.x * token_width / decoded_shape.width),
                token_width - 1,
            )

        token = (int(token_f), int(token_h), int(token_w))
        if not (
            0 <= token[0] < token_frames
            and 0 <= token[1] < token_height
            and 0 <= token[2] < token_width
        ):
            raise ValueError(
                f"query {query.name!r} token {token} is outside grid {grid_size}"
            )
        flat_index = (token[0] * token_height + token[1]) * token_width + token[2]
        flat_indices.append(flat_index)
        resolved.append(
            {
                "name": query.name,
                "source": query.to_dict(),
                "token_fhw": list(token),
                "flat_index": flat_index,
            }
        )
    return resolved, flat_indices


class SelectedRowAttentionProbe:
    """Compute exact selected attention rows without materializing ``L x L``."""

    def __init__(
        self,
        *,
        queries: Sequence[QueryPoint],
        decoded_shape: DecodedVideoShape,
        layers: Sequence[int],
        record_steps: Sequence[int],
        output_dir: str | Path,
        branch: str = "cond",
        temporal_stride: int = 4,
        top_k: int = 16,
    ) -> None:
        if not queries:
            raise ValueError("at least one query is required")
        if not layers or min(int(layer) for layer in layers) < 0:
            raise ValueError("layers must contain non-negative indices")
        if not record_steps or min(int(step) for step in record_steps) < 0:
            raise ValueError("record_steps must contain non-negative indices")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self.queries = tuple(queries)
        self.decoded_shape = decoded_shape
        self.layers = frozenset(int(layer) for layer in layers)
        self.record_steps = frozenset(int(step) for step in record_steps)
        self.branch = str(branch)
        self.temporal_stride = int(temporal_stride)
        self.top_k = int(top_k)
        self.writer = SidecarWriter(output_dir)
        self.records: list[dict[str, Any]] = []
        self._observed: set[tuple[str, str, int, int]] = set()

    def should_observe(
        self,
        *,
        layer_idx: int,
        context: Mapping[str, Any] | object | None,
    ) -> bool:
        if context is None or int(layer_idx) not in self.layers:
            return False
        branch = str(_context_value(context, "branch", "cond"))
        step_index = int(_context_value(context, "step_index", -1))
        return branch == self.branch and step_index in self.record_steps

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
        layer_idx = int(layer_idx)
        if not self.should_observe(layer_idx=layer_idx, context=context):
            return
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError("row probe expects matching q/k shaped [B,L,H,D]")
        if q.shape[0] != 1:
            raise NotImplementedError("row probe currently requires batch size 1")

        valid_len = int(seq_lens[0].detach().cpu().item())
        grid_size = tuple(int(value) for value in grid_sizes[0].detach().cpu().tolist())
        if math.prod(grid_size) != valid_len:
            raise ValueError(
                f"valid sequence length {valid_len} does not equal "
                f"grid product {grid_size}"
            )
        if valid_len > q.shape[1]:
            raise ValueError(
                f"valid sequence length {valid_len} exceeds q/k length {q.shape[1]}"
            )

        video_id = str(_context_value(context, "video_id", "unnamed"))
        branch = str(_context_value(context, "branch", "cond"))
        step_index = int(_context_value(context, "step_index", -1))
        observation_key = (video_id, branch, step_index, layer_idx)
        if observation_key in self._observed:
            raise RuntimeError(f"duplicate row-probe observation: {observation_key}")

        resolved_queries, flat_indices = resolve_query_tokens(
            self.queries,
            decoded_shape=self.decoded_shape,
            grid_size=grid_size,
            temporal_stride=self.temporal_stride,
        )
        query_indices = torch.tensor(
            flat_indices,
            dtype=torch.long,
            device=q.device,
        )
        selected_q = q[0].index_select(0, query_indices)
        valid_k = k[0, :valid_len]

        # Match FlashAttention's default scale. The matmul stays in the model's
        # Q/K dtype; only the small [queries, heads, L] logits and softmax use
        # float32. No [L,L] tensor is ever constructed.
        scale = q.shape[-1] ** -0.5
        logits = torch.einsum("qhd,lhd->qhl", selected_q, valid_k)
        probabilities = torch.softmax(logits.float() * scale, dim=-1)
        volume = probabilities.reshape(
            len(self.queries),
            q.shape[2],
            *grid_size,
        )

        temporal_mass = volume.sum(dim=(-1, -2))
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        normalized_entropy = entropy / math.log(valid_len)
        top_count = min(self.top_k, valid_len)
        top_probability, top_index = probabilities.topk(top_count, dim=-1)

        token_coordinate_values = [
            tuple(int(value) for value in item["token_fhw"])
            for item in resolved_queries
        ]
        token_coordinates = torch.tensor(
            token_coordinate_values,
            dtype=torch.long,
            device=q.device,
        )
        query_frame_mass = torch.stack(
            [
                temporal_mass[index, :, int(token[0])]
                for index, token in enumerate(token_coordinate_values)
            ],
            dim=0,
        )
        local_same_frame: list[torch.Tensor] = []
        local_3d: list[torch.Tensor] = []
        fixed_spatial_tube: list[torch.Tensor] = []
        selected_anchor_same_frame: list[torch.Tensor] = []
        selected_anchor_3d: list[torch.Tensor] = []
        for query_index, token in enumerate(token_coordinate_values):
            token_f, token_h, token_w = token
            h0, h1 = max(0, token_h - 1), min(grid_size[1], token_h + 2)
            w0, w1 = max(0, token_w - 1), min(grid_size[2], token_w + 2)
            f0, f1 = max(0, token_f - 1), min(grid_size[0], token_f + 2)
            local_same_frame.append(
                volume[query_index, :, token_f, h0:h1, w0:w1].sum(dim=(-1, -2))
            )
            local_3d.append(
                volume[query_index, :, f0:f1, h0:h1, w0:w1].sum(dim=(-1, -2, -3))
            )
            fixed_spatial_tube.append(
                volume[query_index, :, :, h0:h1, w0:w1].sum(dim=(-1, -2, -3))
            )
            anchor_same_for_query: list[torch.Tensor] = []
            anchor_3d_for_query: list[torch.Tensor] = []
            for anchor in token_coordinate_values:
                anchor_f, anchor_h, anchor_w = anchor
                anchor_h0 = max(0, anchor_h - 1)
                anchor_h1 = min(grid_size[1], anchor_h + 2)
                anchor_w0 = max(0, anchor_w - 1)
                anchor_w1 = min(grid_size[2], anchor_w + 2)
                anchor_f0 = max(0, anchor_f - 1)
                anchor_f1 = min(grid_size[0], anchor_f + 2)
                anchor_same_for_query.append(
                    volume[
                        query_index,
                        :,
                        anchor_f,
                        anchor_h0:anchor_h1,
                        anchor_w0:anchor_w1,
                    ].sum(dim=(-1, -2))
                )
                anchor_3d_for_query.append(
                    volume[
                        query_index,
                        :,
                        anchor_f0:anchor_f1,
                        anchor_h0:anchor_h1,
                        anchor_w0:anchor_w1,
                    ].sum(dim=(-1, -2, -3))
                )
            selected_anchor_same_frame.append(
                torch.stack(anchor_same_for_query, dim=-1)
            )
            selected_anchor_3d.append(torch.stack(anchor_3d_for_query, dim=-1))

        local_same_frame_tensor = torch.stack(local_same_frame)
        local_3d_tensor = torch.stack(local_3d)
        fixed_spatial_tube_tensor = torch.stack(fixed_spatial_tube)
        selected_anchor_same_frame_tensor = torch.stack(selected_anchor_same_frame)
        selected_anchor_3d_tensor = torch.stack(selected_anchor_3d)
        top_f = top_index // (grid_size[1] * grid_size[2])
        top_remainder = top_index.remainder(grid_size[1] * grid_size[2])
        top_h = top_remainder // grid_size[2]
        top_w = top_remainder.remainder(grid_size[2])
        top_token_fhw = torch.stack((top_f, top_h, top_w), dim=-1)

        attention_sum_error = (
            (probabilities.sum(dim=-1) - 1.0).abs().max().detach().cpu().item()
        )
        record: dict[str, Any] = {
            "schema_version": 1,
            "probe": "selected_attention_rows",
            "video_id": video_id,
            "branch": branch,
            "step_index": step_index,
            "timestep": float(_context_value(context, "timestep", float("nan"))),
            "layer_idx": layer_idx,
            "grid_size": list(grid_size),
            "decoded_video_shape": self.decoded_shape.to_dict(),
            "valid_tokens": valid_len,
            "num_heads": q.shape[2],
            "head_dim": q.shape[3],
            "qk_dtype": str(q.dtype),
            "softmax_dtype": str(probabilities.dtype),
            "softmax_scale": scale,
            "attention_sum_max_abs_error": attention_sum_error,
            "queries": resolved_queries,
            "display_note": (
                "Raw arrays are linear probabilities. Rendered PNG/MP4 heatmaps "
                "use one log-probability scale per query/head volume."
            ),
        }
        arrays = {
            "attention_per_head": volume,
            "attention_head_mean": volume.mean(dim=1),
            "temporal_mass_per_head": temporal_mass,
            "entropy_per_head": entropy,
            "normalized_entropy_per_head": normalized_entropy,
            "query_frame_mass_per_head": query_frame_mass,
            "local_3x3_same_frame_mass_per_head": local_same_frame_tensor,
            "local_3x3x3_mass_per_head": local_3d_tensor,
            "fixed_3x3_spatial_tube_mass_per_head": fixed_spatial_tube_tensor,
            "selected_anchor_3x3_same_frame_mass_per_head": (
                selected_anchor_same_frame_tensor
            ),
            "selected_anchor_3x3x3_mass_per_head": selected_anchor_3d_tensor,
            "topk_probabilities": top_probability,
            "topk_flat_indices": top_index,
            "topk_token_fhw": top_token_fhw,
            "query_flat_indices": query_indices,
            "query_tokens_fhw": token_coordinates,
        }

        self.writer.write_observation(
            video_id=video_id,
            branch=branch,
            step_index=step_index,
            layer_idx=layer_idx,
            record=record,
            arrays=arrays,
        )
        self.records.append(record)
        self._observed.add(observation_key)

    def finish_video(self, video_id: str) -> None:
        """Compatibility hook called by the Wan generation loop."""

    def clear(self) -> None:
        self.records.clear()
        self._observed.clear()
