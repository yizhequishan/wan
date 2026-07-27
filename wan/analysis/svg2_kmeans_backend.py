"""Thin adapter around SVG2's official Flash-KMeans implementation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import torch


KMeansCallable = Callable[..., tuple]


@dataclass
class KMeansResult:
    """Normalized output shared by bundled SVG2 and flash-kmeans releases."""

    labels: torch.Tensor
    centroids: torch.Tensor
    sizes: torch.Tensor
    n_iters: int


@dataclass
class _CentroidState:
    q: torch.Tensor
    k: torch.Tensor


def _load_flash_kmeans() -> KMeansCallable:
    try:
        from flash_kmeans import batch_kmeans_Euclid
    except ImportError as exc:
        raise RuntimeError(
            "The SVG2 sidecar requires flash-kmeans. Install it with "
            '`python -m pip install "flash-kmeans==0.2.0"` or install '
            "requirements-analysis.txt."
        ) from exc
    return batch_kmeans_Euclid


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a process-stable seed without Python's randomized hash()."""

    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def cluster_sizes(labels: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """Count cluster assignments for labels shaped ``[batch, tokens]``."""

    labels = labels.long()
    sizes = torch.zeros(
        labels.shape[0],
        num_clusters,
        device=labels.device,
        dtype=torch.int64,
    )
    sizes.scatter_add_(
        1,
        labels,
        torch.ones_like(labels, dtype=torch.int64),
    )
    return sizes


class SVG2CachedKMeans:
    """SVG2 Q/K K-means with centroid warm starts and isolated cache keys."""

    def __init__(
        self,
        cq: int = 128,
        ck: int = 512,
        init_iters: int = 50,
        step_iters: int = 2,
        tol: float = 1e-4,
        kmeans_fn: KMeansCallable | None = None,
    ) -> None:
        if cq <= 0 or ck <= 0:
            raise ValueError("cq and ck must be positive")
        if init_iters <= 0 or step_iters <= 0:
            raise ValueError("iteration counts must be positive")

        self.cq = int(cq)
        self.ck = int(ck)
        self.init_iters = int(init_iters)
        self.step_iters = int(step_iters)
        self.tol = float(tol)
        self._kmeans_fn = kmeans_fn
        self._cache: dict[tuple[object, ...], _CentroidState] = {}

    @property
    def kmeans_fn(self) -> KMeansCallable:
        if self._kmeans_fn is None:
            self._kmeans_fn = _load_flash_kmeans()
        return self._kmeans_fn

    def clear(self) -> None:
        self._cache.clear()

    def reset_video(self, video_id: str) -> None:
        stale = [key for key in self._cache if key[0] == video_id]
        for key in stale:
            del self._cache[key]

    def has_cache(
        self,
        video_id: str,
        branch: str,
        layer_idx: int,
    ) -> bool:
        return self._cache_key(video_id, branch, layer_idx) in self._cache

    def _cache_key(
        self,
        video_id: str,
        branch: str,
        layer_idx: int,
    ) -> tuple[object, ...]:
        return (video_id, branch, int(layer_idx), self.cq, self.ck)

    def _fit(
        self,
        x: torch.Tensor,
        num_clusters: int,
        init_centroids: torch.Tensor | None,
        max_iters: int,
    ) -> KMeansResult:
        output = self.kmeans_fn(
            x,
            n_clusters=num_clusters,
            max_iters=max_iters,
            tol=self.tol,
            init_centroids=init_centroids,
            verbose=False,
        )
        if not isinstance(output, tuple):
            output = tuple(output)

        # Standalone flash-kmeans returns (labels, centroids, n_iters).
        # Sparse-VideoGen's bundled copy additionally returns cluster sizes.
        if len(output) == 3:
            labels, centroids, n_iters = output
            sizes = cluster_sizes(labels, num_clusters)
        elif len(output) == 4:
            labels, centroids, sizes, n_iters = output
        else:
            raise RuntimeError(
                "Unexpected batch_kmeans_Euclid return signature: "
                f"{len(output)} values"
            )

        return KMeansResult(
            labels=labels,
            centroids=centroids,
            sizes=sizes,
            n_iters=int(n_iters),
        )

    @torch.inference_mode()
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        video_id: str,
        branch: str,
        layer_idx: int,
        seed: int,
    ) -> tuple[KMeansResult, KMeansResult]:
        """Cluster native Wan Q/K tensors shaped ``[B, L, H, D]``."""

        if q.shape != k.shape or q.ndim != 4:
            raise ValueError(
                f"q and k must have identical [B,L,H,D] shapes, got "
                f"{tuple(q.shape)} and {tuple(k.shape)}"
            )

        batch, seq_len, num_heads, head_dim = q.shape
        q_flat = (
            q.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch * num_heads, seq_len, head_dim)
        )
        k_flat = (
            k.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch * num_heads, seq_len, head_dim)
        )

        cache_key = self._cache_key(video_id, branch, layer_idx)
        previous = self._cache.get(cache_key)
        max_iters = self.init_iters if previous is None else self.step_iters
        q_init = None if previous is None else previous.q
        k_init = None if previous is None else previous.k

        cuda_devices: list[int] = []
        if q.is_cuda:
            cuda_devices = [
                (
                    q.device.index
                    if q.device.index is not None
                    else torch.cuda.current_device()
                )
            ]

        # Flash-KMeans samples initial centers with torch.randint. Restoring
        # RNG state is part of the sidecar's "read only" contract.
        with torch.random.fork_rng(devices=cuda_devices):
            if previous is None:
                torch.manual_seed(seed)
                if q.is_cuda:
                    torch.cuda.manual_seed(seed)
            q_result = self._fit(q_flat, self.cq, q_init, max_iters)
            k_result = self._fit(k_flat, self.ck, k_init, max_iters)

        self._cache[cache_key] = _CentroidState(
            q=q_result.centroids.detach(),
            k=k_result.centroids.detach(),
        )

        q_result.labels = q_result.labels.view(batch, num_heads, seq_len)
        k_result.labels = k_result.labels.view(batch, num_heads, seq_len)
        q_result.centroids = q_result.centroids.view(
            batch, num_heads, self.cq, head_dim
        )
        k_result.centroids = k_result.centroids.view(
            batch, num_heads, self.ck, head_dim
        )
        q_result.sizes = q_result.sizes.view(batch, num_heads, self.cq)
        k_result.sizes = k_result.sizes.view(batch, num_heads, self.ck)
        return q_result, k_result
