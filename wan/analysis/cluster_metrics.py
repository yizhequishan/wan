"""Metrics computed from cluster labels, centroids, and entity contingency."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x.log(), torch.zeros_like(x))


def clustering_scores(contingency: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute homogeneity, completeness, and V-measure per head.

    Args:
        contingency: Counts shaped ``[heads, clusters, entity_categories]``.
    """

    if contingency.ndim != 3:
        raise ValueError("contingency must have shape [H,C,E]")

    counts = contingency.to(torch.float64)
    total = counts.sum(dim=(1, 2), keepdim=True)
    joint = counts / total.clamp_min(1)
    cluster_prob = joint.sum(dim=2)
    entity_prob = joint.sum(dim=1)

    entity_entropy = -(entity_prob * _safe_log(entity_prob)).sum(dim=1)
    cluster_entropy = -(cluster_prob * _safe_log(cluster_prob)).sum(dim=1)

    entity_given_cluster = -(
        joint * (_safe_log(joint) - _safe_log(cluster_prob).unsqueeze(-1))
    ).sum(dim=(1, 2))
    cluster_given_entity = -(
        joint * (_safe_log(joint) - _safe_log(entity_prob).unsqueeze(1))
    ).sum(dim=(1, 2))

    homogeneity = torch.where(
        entity_entropy > 0,
        1.0 - entity_given_cluster / entity_entropy,
        torch.ones_like(entity_entropy),
    )
    completeness = torch.where(
        cluster_entropy > 0,
        1.0 - cluster_given_entity / cluster_entropy,
        torch.ones_like(cluster_entropy),
    )
    v_measure = torch.where(
        homogeneity + completeness > 0,
        2 * homogeneity * completeness / (homogeneity + completeness),
        torch.zeros_like(homogeneity),
    )

    has_data = total.flatten() > 0
    nan = torch.full_like(homogeneity, float("nan"))
    return {
        "homogeneity": torch.where(has_data, homogeneity, nan),
        "completeness": torch.where(has_data, completeness, nan),
        "v_measure": torch.where(has_data, v_measure, nan),
    }


def fragmentation_at_mass(
    contingency: torch.Tensor,
    mass: float = 0.9,
    exclude_background: bool = True,
) -> torch.Tensor:
    """Mean number of clusters needed to cover ``mass`` of each entity."""

    if not 0 < mass <= 1:
        raise ValueError("mass must be in (0, 1]")
    if contingency.ndim != 3:
        raise ValueError("contingency must have shape [H,C,E]")

    counts = contingency.to(torch.float64)
    if exclude_background and counts.shape[-1] > 1:
        counts = counts[..., 1:]

    per_head: list[torch.Tensor] = []
    for head_counts in counts:
        values: list[torch.Tensor] = []
        for entity_counts in head_counts.transpose(0, 1):
            total = entity_counts.sum()
            if total <= 0:
                continue
            ordered = entity_counts.sort(descending=True).values
            covered = ordered.cumsum(0) >= mass * total
            first = torch.nonzero(covered, as_tuple=False)[0, 0] + 1
            values.append(first.to(torch.float64))
        if values:
            per_head.append(torch.stack(values).mean())
        else:
            per_head.append(torch.tensor(float("nan"), device=counts.device))
    return torch.stack(per_head)


def background_mixing(contingency: torch.Tensor) -> torch.Tensor:
    """Fraction of tokens in foreground/background-mixed cluster mass.

    Category zero is assumed to be background. For each cluster this counts the
    smaller of background and foreground mass, then normalizes by all tokens.
    """

    if contingency.ndim != 3:
        raise ValueError("contingency must have shape [H,C,E]")
    if contingency.shape[-1] < 2:
        return torch.zeros(
            contingency.shape[0],
            dtype=torch.float64,
            device=contingency.device,
        )

    counts = contingency.to(torch.float64)
    background = counts[..., 0]
    foreground = counts[..., 1:].sum(dim=-1)
    mixed = torch.minimum(background, foreground).sum(dim=-1)
    return mixed / counts.sum(dim=(1, 2)).clamp_min(1)


def adjacent_frame_js(contingency: torch.Tensor) -> torch.Tensor:
    """Mean adjacent-frame JS divergence of ``p(cluster | entity, frame)``."""

    if contingency.ndim != 4:
        raise ValueError("contingency must have shape [H,C,F,E]")

    counts = contingency.to(torch.float64)
    num_heads, _, num_frames, num_entities = counts.shape
    output = torch.full(
        (num_heads,),
        float("nan"),
        dtype=torch.float64,
        device=counts.device,
    )

    for head in range(num_heads):
        divergences: list[torch.Tensor] = []
        for frame in range(num_frames - 1):
            for entity in range(1, num_entities):
                left = counts[head, :, frame, entity]
                right = counts[head, :, frame + 1, entity]
                if left.sum() <= 0 or right.sum() <= 0:
                    continue
                left = left / left.sum()
                right = right / right.sum()
                middle = 0.5 * (left + right)
                js = 0.5 * (
                    (left * (_safe_log(left) - _safe_log(middle))).sum()
                    + (right * (_safe_log(right) - _safe_log(middle))).sum()
                )
                divergences.append(js)
        if divergences:
            output[head] = torch.stack(divergences).mean()
    return output


def adjusted_rand_index(
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
) -> torch.Tensor:
    """Adjusted Rand index per head for labels shaped ``[H, L]``."""

    if labels_a.shape != labels_b.shape or labels_a.ndim != 2:
        raise ValueError("labels must have identical [H,L] shapes")

    results: list[torch.Tensor] = []
    for left, right in zip(labels_a.long(), labels_b.long()):
        _, left_inv = torch.unique(left, return_inverse=True)
        _, right_inv = torch.unique(right, return_inverse=True)
        num_left = int(left_inv.max().item()) + 1
        num_right = int(right_inv.max().item()) + 1
        flat = left_inv * num_right + right_inv
        table = (
            torch.bincount(
                flat,
                minlength=num_left * num_right,
            )
            .reshape(num_left, num_right)
            .to(torch.float64)
        )

        def choose_two(value: torch.Tensor) -> torch.Tensor:
            return value * (value - 1) / 2

        cells = choose_two(table).sum()
        rows = choose_two(table.sum(dim=1)).sum()
        cols = choose_two(table.sum(dim=0)).sum()
        total_pairs = choose_two(
            torch.tensor(left.numel(), dtype=torch.float64, device=left.device)
        )
        if total_pairs <= 0:
            results.append(torch.tensor(1.0, device=left.device))
            continue
        expected = rows * cols / total_pairs
        maximum = 0.5 * (rows + cols)
        denominator = maximum - expected
        ari = (
            (cells - expected) / denominator
            if denominator != 0
            else torch.tensor(1.0, device=left.device)
        )
        results.append(ari)
    return torch.stack(results)


def centroid_continuity(
    previous: torch.Tensor,
    current: torch.Tensor,
    compute_hungarian: bool = True,
) -> dict[str, Any]:
    """Compare cached centroid identity across denoising observations.

    Args:
        previous/current: Tensors shaped ``[heads, clusters, dim]``.
    """

    if previous.shape != current.shape or previous.ndim != 3:
        raise ValueError("centroids must have identical [H,C,D] shapes")

    previous = F.normalize(previous.float(), dim=-1)
    current = F.normalize(current.float(), dim=-1)
    similarity = torch.einsum("hcd,hkd->hck", previous, current)
    cluster_count = similarity.shape[1]
    indices = torch.arange(cluster_count, device=similarity.device)
    same_index = similarity[:, indices, indices].mean(dim=1)
    nearest = similarity.max(dim=2).values.mean(dim=1)

    hungarian: list[float] | None = None
    if compute_hungarian:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            hungarian = None
        else:
            hungarian = []
            for head_similarity in similarity:
                matrix = head_similarity.detach().cpu().numpy()
                rows, columns = linear_sum_assignment(-matrix)
                hungarian.append(float(matrix[rows, columns].mean()))

    return {
        "same_index_cosine": same_index.detach().cpu().tolist(),
        "nearest_cosine": nearest.detach().cpu().tolist(),
        "hungarian_cosine": hungarian,
    }


def tensor_summary(values: torch.Tensor) -> dict[str, float | list[float]]:
    """JSON-friendly per-head values plus a finite mean."""

    cpu = values.detach().to(torch.float64).cpu()
    finite = cpu[torch.isfinite(cpu)]
    mean = float(finite.mean()) if finite.numel() else math.nan
    return {"per_head": cpu.tolist(), "mean": mean}
