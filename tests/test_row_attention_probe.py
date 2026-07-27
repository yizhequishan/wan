import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wan.analysis import (
    DecodedVideoShape,
    QueryPoint,
    SelectedRowAttentionProbe,
    decoded_frame_to_token_frame,
    load_row_probe_request,
    resolve_query_tokens,
)


def test_decoded_frame_mapping_matches_causal_vae_chunks():
    assert (
        decoded_frame_to_token_frame(
            0,
            decoded_frames=81,
            token_frames=21,
        )
        == 0
    )
    assert (
        decoded_frame_to_token_frame(
            1,
            decoded_frames=81,
            token_frames=21,
        )
        == 1
    )
    assert (
        decoded_frame_to_token_frame(
            4,
            decoded_frames=81,
            token_frames=21,
        )
        == 1
    )
    assert (
        decoded_frame_to_token_frame(
            5,
            decoded_frames=81,
            token_frames=21,
        )
        == 2
    )
    assert (
        decoded_frame_to_token_frame(
            80,
            decoded_frames=81,
            token_frames=21,
        )
        == 20
    )


def test_resolve_decoded_and_direct_query_tokens():
    shape = DecodedVideoShape(frames=5, height=4, width=4)
    queries = (
        QueryPoint(name="pixel", frame=4, x=3.5, y=0.5),
        QueryPoint(name="direct", token=(0, 1, 0)),
    )
    resolved, indices = resolve_query_tokens(
        queries,
        decoded_shape=shape,
        grid_size=(2, 2, 2),
    )

    assert resolved[0]["token_fhw"] == [1, 0, 1]
    assert resolved[1]["token_fhw"] == [0, 1, 0]
    assert indices == [5, 2]


def test_selected_attention_row_matches_manual_softmax(tmp_path: Path):
    torch.manual_seed(7)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    queries = (
        QueryPoint(name="object_a", frame=4, x=3.5, y=0.5),
        QueryPoint(name="object_b", token=(0, 1, 0)),
    )
    probe = SelectedRowAttentionProbe(
        queries=queries,
        decoded_shape=DecodedVideoShape(frames=5, height=4, width=4),
        layers=(1,),
        record_steps=(2,),
        output_dir=tmp_path,
        top_k=3,
    )

    assert not probe.should_observe(
        layer_idx=0,
        context={"branch": "cond", "step_index": 2},
    )
    assert not probe.should_observe(
        layer_idx=1,
        context={"branch": "uncond", "step_index": 2},
    )
    assert probe.should_observe(
        layer_idx=1,
        context={"branch": "cond", "step_index": 2},
    )

    probe.observe(
        q=q,
        k=k,
        seq_lens=torch.tensor([8]),
        grid_sizes=torch.tensor([[2, 2, 2]]),
        layer_idx=1,
        context={
            "video_id": "two_objects",
            "branch": "cond",
            "step_index": 2,
            "timestep": 500.0,
        },
    )

    array_path = next((tmp_path / "two_objects").glob("*.npz"))
    with np.load(array_path, allow_pickle=False) as arrays:
        actual = torch.from_numpy(arrays["attention_per_head"])
        expected = torch.softmax(
            torch.einsum("qhd,lhd->qhl", q[0, [5, 2]], k[0]) / 2.0,
            dim=-1,
        ).reshape(2, 2, 2, 2, 2)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual.sum(dim=(-1, -2, -3)),
            torch.ones(2, 2),
        )
        assert arrays["attention_per_head"].shape == (2, 2, 2, 2, 2)
        assert arrays["temporal_mass_per_head"].shape == (2, 2, 2)
        assert arrays["topk_token_fhw"].shape == (2, 2, 3, 3)
        assert arrays["selected_anchor_3x3_same_frame_mass_per_head"].shape == (2, 2, 2)
        assert arrays["query_flat_indices"].tolist() == [5, 2]

    record = json.loads((tmp_path / "records.jsonl").read_text(encoding="utf-8"))
    assert record["array_file"].endswith(".npz")
    assert record["queries"][0]["token_fhw"] == [1, 0, 1]
    assert record["probe"] == "selected_attention_rows"
    assert record["attention_sum_max_abs_error"] < 1e-6


def test_query_file_round_trip(tmp_path: Path):
    path = tmp_path / "queries.json"
    value = {
        "video_id": "sample",
        "prompt": "two moving objects",
        "negative_prompt": "",
        "seed": 9,
        "video_shape": {"frames": 5, "height": 4, "width": 4},
        "generation": {"task": "t2v-1.3B", "sampling_steps": 50},
        "queries": [
            {"name": "a", "frame": 1, "x": 1, "y": 2},
            {"name": "b", "token": [1, 0, 0]},
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")

    request = load_row_probe_request(path)
    assert request.video_id == "sample"
    assert request.video_shape == DecodedVideoShape(5, 4, 4)
    assert request.queries[1].token == (1, 0, 0)
    assert request.generation["sampling_steps"] == 50
    assert request.to_dict() == value


def test_inconsistent_temporal_shape_is_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        decoded_frame_to_token_frame(
            4,
            decoded_frames=6,
            token_frames=2,
        )


def test_example_query_config_is_valid():
    example = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "configs"
        / "row_attention_queries.example.json"
    )
    request = load_row_probe_request(example)
    assert request.video_shape == DecodedVideoShape(81, 480, 832)
    assert request.generation["shift"] == 8.0
    assert len(request.queries) == 6
