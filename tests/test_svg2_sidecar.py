from pathlib import Path

import numpy as np
import torch

from wan.analysis import SVG2Sidecar, SVG2SidecarConfig
from wan.analysis.svg2_kmeans_backend import SVG2CachedKMeans


def _fake_kmeans(
    x,
    n_clusters,
    max_iters,
    tol,
    init_centroids,
    verbose,
):
    labels = (
        torch.arange(x.shape[1], device=x.device)
        .remainder(n_clusters)
        .expand(x.shape[0], -1)
        .clone()
    )
    if init_centroids is None:
        centroids = x[:, :n_clusters].clone()
    else:
        centroids = init_centroids
    return labels, centroids, max_iters


def _sidecar(tmp_path: Path) -> SVG2Sidecar:
    config = SVG2SidecarConfig(
        layers=(0,),
        record_steps=(0,),
        cq=2,
        ck=2,
        visualization_ids=("visual",),
        compute_hungarian=False,
        output_dir=str(tmp_path),
        expected_videos=1,
        score_chunk_q=1,
    )
    backend = SVG2CachedKMeans(
        cq=2,
        ck=2,
        init_iters=3,
        step_iters=1,
        kmeans_fn=_fake_kmeans,
    )
    return SVG2Sidecar(config, backend=backend)


def _observe(sidecar: SVG2Sidecar, video_id: str) -> None:
    q = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.1]], [[0.0, 1.0]], [[0.1, 1.0]]]])
    sidecar.observe(
        q=q,
        k=q,
        seq_lens=torch.tensor([4]),
        grid_sizes=torch.tensor([[1, 2, 2]]),
        layer_idx=0,
        context={
            "video_id": video_id,
            "branch": "cond",
            "step_index": 0,
            "timestep": 999.0,
            "analysis_seed": 7,
        },
    )


def test_only_visualization_video_saves_full_labels(tmp_path):
    sidecar = _sidecar(tmp_path)
    entities = torch.tensor([-1, 0, 1, -2])

    sidecar.start_video(
        "visual",
        entity_ids=entities,
        grid_size=(1, 2, 2),
    )
    _observe(sidecar, "visual")
    visual_file = next((tmp_path / "visual").glob("*.npz"))
    with np.load(visual_file, allow_pickle=False) as arrays:
        assert "q_labels" in arrays
        assert "k_labels" in arrays
        assert "ghat_same_frame" in arrays

    sidecar.finish_video("visual")
    sidecar.start_video(
        "aggregate",
        entity_ids=entities,
        grid_size=(1, 2, 2),
    )
    _observe(sidecar, "aggregate")
    aggregate_file = next((tmp_path / "aggregate").glob("*.npz"))
    with np.load(aggregate_file, allow_pickle=False) as arrays:
        assert "q_labels" not in arrays
        assert "k_labels" not in arrays
        assert "q_entity_contingency" in arrays
        assert "ghat_same_frame" in arrays


def test_non_record_step_updates_cache_without_writing(tmp_path):
    config = SVG2SidecarConfig(
        layers=(0,),
        record_steps=(2,),
        cq=2,
        ck=2,
        compute_hungarian=False,
        output_dir=str(tmp_path),
        expected_videos=1,
    )
    backend = SVG2CachedKMeans(
        cq=2,
        ck=2,
        kmeans_fn=_fake_kmeans,
    )
    sidecar = SVG2Sidecar(config, backend=backend)
    q = torch.randn(1, 4, 1, 2)
    sidecar.observe(
        q=q,
        k=q,
        seq_lens=torch.tensor([4]),
        grid_sizes=torch.tensor([[1, 2, 2]]),
        layer_idx=0,
        context={
            "video_id": "video",
            "branch": "cond",
            "step_index": 0,
        },
    )

    assert backend.has_cache("video", "cond", 0)
    assert sidecar.records == []
    assert not list(tmp_path.rglob("*.npz"))
