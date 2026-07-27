"""Online cluster/entity contingency and centroid-level graph projection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class EntityGraphResult:
    """Entity graph including background as category zero."""

    cross_frame: torch.Tensor
    same_frame: torch.Tensor


def latent_frame_ids(
    grid_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    frames, height, width = (int(value) for value in grid_size)
    return torch.arange(
        frames,
        device=device,
        dtype=torch.long,
    ).repeat_interleave(height * width)


def build_cluster_entity_contingency(
    labels: torch.Tensor,
    entity_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    num_clusters: int,
    num_frames: int,
    num_entity_categories: int,
) -> torch.Tensor:
    """Count ``[head, cluster, frame, entity_category]`` online.

    ``entity_ids`` uses ``-2`` for excluded boundary, ``-1`` for background,
    and non-negative values for entities. Output category zero is background;
    entity ``e`` is stored at category ``e + 1``.
    """

    if labels.ndim != 2:
        raise ValueError("labels must have shape [H,L]")
    if entity_ids.ndim != 1 or frame_ids.ndim != 1:
        raise ValueError("entity_ids and frame_ids must have shape [L]")
    if labels.shape[1] != entity_ids.numel():
        raise ValueError("labels and entity_ids token counts differ")
    if frame_ids.numel() != entity_ids.numel():
        raise ValueError("frame_ids and entity_ids token counts differ")

    entity_ids = entity_ids.to(labels.device, dtype=torch.long)
    frame_ids = frame_ids.to(labels.device, dtype=torch.long)
    valid = entity_ids >= -1
    categories = entity_ids[valid] + 1
    valid_frames = frame_ids[valid]
    valid_labels = labels[:, valid].long()

    if categories.numel() == 0:
        return torch.zeros(
            labels.shape[0],
            num_clusters,
            num_frames,
            num_entity_categories,
            device=labels.device,
            dtype=torch.int64,
        )
    if int(categories.max()) >= num_entity_categories:
        raise ValueError("entity id exceeds declared category count")

    num_heads = labels.shape[0]
    head_ids = torch.arange(
        num_heads,
        device=labels.device,
        dtype=torch.long,
    ).unsqueeze(1)
    flat = (
        (head_ids * num_clusters + valid_labels) * num_frames
        + valid_frames.unsqueeze(0)
    ) * num_entity_categories + categories.unsqueeze(0)

    counts = torch.bincount(
        flat.reshape(-1),
        minlength=(num_heads * num_clusters * num_frames * num_entity_categories),
    )
    return counts.reshape(
        num_heads,
        num_clusters,
        num_frames,
        num_entity_categories,
    )


def approximate_entity_graph(
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    q_sizes: torch.Tensor,
    k_sizes: torch.Tensor,
    q_contingency: torch.Tensor,
    k_contingency: torch.Tensor,
    *,
    score_chunk_q: int = 32,
) -> EntityGraphResult:
    """Project SVG2 centroid routing into a frame/entity interaction graph.

    Inputs omit batch and have shapes:

    - centroids: ``[H,C,D]``
    - sizes: ``[H,C]``
    - contingency: ``[H,C,F,E]``
    """

    if q_centroids.ndim != 3 or k_centroids.ndim != 3:
        raise ValueError("centroids must have shape [H,C,D]")
    if q_centroids.shape[0] != k_centroids.shape[0]:
        raise ValueError("Q and K head counts differ")
    if q_centroids.shape[-1] != k_centroids.shape[-1]:
        raise ValueError("Q and K head dimensions differ")
    if q_contingency.ndim != 4 or k_contingency.ndim != 4:
        raise ValueError("contingencies must have shape [H,C,F,E]")

    heads, cq, head_dim = q_centroids.shape
    _, ck, _, _ = k_contingency.shape
    num_frames = q_contingency.shape[2]
    num_categories = q_contingency.shape[3]
    if k_contingency.shape[2:] != (num_frames, num_categories):
        raise ValueError("Q and K contingency grids differ")
    if q_contingency.shape[:2] != (heads, cq):
        raise ValueError("Q centroid and contingency shapes differ")
    if k_centroids.shape[:2] != (heads, ck):
        raise ValueError("K centroid and contingency shapes differ")
    if q_sizes.shape != (heads, cq) or k_sizes.shape != (heads, ck):
        raise ValueError("cluster size shapes do not match centroids")

    q_centroids = q_centroids.float()
    k_centroids = k_centroids.float()
    q_sizes = q_sizes.float()
    k_sizes = k_sizes.float()
    q_counts = q_contingency.float()
    k_counts = k_contingency.float()
    if torch.any(q_counts.sum(dim=(2, 3)) > q_sizes):
        raise ValueError("Q contingency exceeds cluster sizes")
    if torch.any(k_counts.sum(dim=(2, 3)) > k_sizes):
        raise ValueError("K contingency exceeds cluster sizes")

    # Fraction of each key cluster belonging to a frame/entity category.
    k_fraction = k_counts / k_sizes[..., None, None].clamp_min(1)
    q_source = q_counts.permute(0, 2, 3, 1)
    q_source = q_source / q_source.sum(dim=-1, keepdim=True).clamp_min(1)

    graph = torch.zeros(
        heads,
        num_frames,
        num_categories,
        num_frames,
        num_categories,
        device=q_centroids.device,
        dtype=torch.float32,
    )

    log_k_sizes = torch.where(
        k_sizes > 0,
        k_sizes.log(),
        torch.full_like(k_sizes, float("-inf")),
    )
    scale = 1.0 / math.sqrt(head_dim)

    for start in range(0, cq, score_chunk_q):
        end = min(start + score_chunk_q, cq)
        scores = torch.einsum(
            "had,hbd->hab",
            q_centroids[:, start:end],
            k_centroids,
        )
        routing = torch.softmax(
            scores * scale + log_k_sizes[:, None, :],
            dim=-1,
        )
        target = torch.einsum(
            "hab,hbtv->hatv",
            routing,
            k_fraction,
        )
        graph += torch.einsum(
            "hsua,hatv->hsutv",
            q_source[..., start:end],
            target,
        )

    same_frame = torch.stack(
        [graph[:, frame, :, frame, :] for frame in range(num_frames)],
        dim=1,
    )
    return EntityGraphResult(cross_frame=graph, same_frame=same_frame)
