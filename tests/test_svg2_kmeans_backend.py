import torch

from wan.analysis.svg2_kmeans_backend import SVG2CachedKMeans


def test_cached_kmeans_shapes_cache_and_rng_isolation():
    calls = []

    def fake_kmeans(
        x,
        n_clusters,
        max_iters,
        tol,
        init_centroids,
        verbose,
    ):
        calls.append(
            {
                "max_iters": max_iters,
                "init": init_centroids,
                "tol": tol,
            }
        )
        # Exercise fork_rng: this random draw must not escape the backend.
        torch.randint(0, x.shape[1], (1,))
        labels = (
            torch.arange(x.shape[1], device=x.device)
            .remainder(n_clusters)
            .expand(x.shape[0], -1)
            .clone()
        )
        if init_centroids is None:
            indices = torch.arange(n_clusters, device=x.device)
            centroids = x[:, indices].clone()
        else:
            centroids = init_centroids + 0.01
        return labels, centroids, max_iters

    backend = SVG2CachedKMeans(
        cq=2,
        ck=3,
        init_iters=5,
        step_iters=2,
        kmeans_fn=fake_kmeans,
    )
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn_like(q)

    rng_before = torch.random.get_rng_state().clone()
    q0, k0 = backend.run(
        q,
        k,
        video_id="video",
        branch="cond",
        layer_idx=0,
        seed=123,
    )
    rng_after = torch.random.get_rng_state()

    assert torch.equal(rng_before, rng_after)
    assert q0.labels.shape == (1, 2, 8)
    assert k0.labels.shape == (1, 2, 8)
    assert q0.centroids.shape == (1, 2, 2, 4)
    assert k0.centroids.shape == (1, 2, 3, 4)
    assert torch.all(q0.sizes.sum(dim=-1) == 8)
    assert calls[0]["init"] is None
    assert calls[1]["init"] is None
    assert calls[0]["max_iters"] == 5

    backend.run(
        q + 0.1,
        k + 0.1,
        video_id="video",
        branch="cond",
        layer_idx=0,
        seed=123,
    )
    assert calls[2]["init"] is not None
    assert calls[3]["init"] is not None
    assert calls[2]["max_iters"] == 2


def test_cfg_branches_do_not_share_centroids():
    initializations = []

    def fake_kmeans(
        x,
        n_clusters,
        max_iters,
        tol,
        init_centroids,
        verbose,
    ):
        initializations.append(init_centroids)
        labels = torch.zeros(
            x.shape[:2],
            dtype=torch.long,
            device=x.device,
        )
        centers = x[:, :n_clusters].clone()
        return labels, centers, 1

    backend = SVG2CachedKMeans(cq=1, ck=1, kmeans_fn=fake_kmeans)
    q = torch.randn(1, 4, 1, 2)
    backend.run(
        q,
        q,
        video_id="v",
        branch="cond",
        layer_idx=0,
        seed=0,
    )
    backend.run(
        q,
        q,
        video_id="v",
        branch="uncond",
        layer_idx=0,
        seed=0,
    )

    assert all(value is None for value in initializations)
