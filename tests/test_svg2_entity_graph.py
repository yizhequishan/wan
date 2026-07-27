import torch

from wan.analysis.cluster_metrics import clustering_scores
from wan.analysis.entity_graph import (
    approximate_entity_graph,
    build_cluster_entity_contingency,
)


def test_perfect_cluster_entity_alignment_scores_one():
    contingency = torch.tensor(
        [
            [
                [4, 0],
                [0, 4],
            ]
        ]
    )
    scores = clustering_scores(contingency)
    assert torch.allclose(scores["homogeneity"], torch.ones_like(scores["homogeneity"]))
    assert torch.allclose(
        scores["completeness"], torch.ones_like(scores["completeness"])
    )
    assert torch.allclose(scores["v_measure"], torch.ones_like(scores["v_measure"]))


def test_contingency_excludes_boundary_and_maps_background():
    labels = torch.tensor([[0, 0, 1, 1]])
    entity_ids = torch.tensor([-1, 0, 1, -2])
    frame_ids = torch.tensor([0, 0, 0, 0])
    counts = build_cluster_entity_contingency(
        labels,
        entity_ids,
        frame_ids,
        num_clusters=2,
        num_frames=1,
        num_entity_categories=3,
    )

    assert counts.shape == (1, 2, 1, 3)
    assert counts.sum() == 3
    assert counts[0, 0, 0, 0] == 1
    assert counts[0, 0, 0, 1] == 1
    assert counts[0, 1, 0, 2] == 1


def test_entity_graph_has_normalized_cross_frame_rows():
    q_centroids = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    k_centroids = q_centroids.clone()
    q_sizes = torch.tensor([[2, 2]])
    k_sizes = torch.tensor([[2, 2]])

    # [head, cluster, frame, category], category 0 is background.
    q_contingency = torch.tensor(
        [
            [
                [[0, 2, 0]],
                [[0, 0, 2]],
            ]
        ]
    )
    k_contingency = q_contingency.clone()
    graph = approximate_entity_graph(
        q_centroids,
        k_centroids,
        q_sizes,
        k_sizes,
        q_contingency,
        k_contingency,
        score_chunk_q=1,
    )

    assert graph.cross_frame.shape == (1, 1, 3, 1, 3)
    assert graph.same_frame.shape == (1, 1, 3, 3)
    entity_rows = graph.cross_frame[0, 0, 1:, 0]
    assert torch.allclose(
        entity_rows.sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )
