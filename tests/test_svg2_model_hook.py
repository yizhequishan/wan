import pytest
import torch

pytest.importorskip("diffusers")

import wan.modules.model as model_module  # noqa: E402
from wan.modules.model import WanSelfAttention  # noqa: E402


class RecordingSidecar:
    def __init__(self):
        self.calls = []

    def observe(self, **kwargs):
        self.calls.append(kwargs)


def test_sidecar_does_not_change_dense_attention_output(monkeypatch):
    monkeypatch.setattr(
        model_module,
        "rope_apply",
        lambda tensor, grid_sizes, freqs: tensor.float(),
    )
    monkeypatch.setattr(
        model_module,
        "flash_attention",
        lambda q, k, v, **kwargs: q + k + v,
    )

    attention = WanSelfAttention(dim=4, num_heads=2).eval()
    x = torch.randn(1, 4, 4)
    seq_lens = torch.tensor([4])
    grid_sizes = torch.tensor([[1, 2, 2]])

    dense = attention(x, seq_lens, grid_sizes, freqs=None)
    observer = RecordingSidecar()
    observed = attention(
        x,
        seq_lens,
        grid_sizes,
        freqs=None,
        layer_idx=0,
        analysis_ctx={"video_id": "v", "step_index": 0},
        analysis_sidecar=observer,
    )

    assert torch.equal(dense, observed)
    assert len(observer.calls) == 1
    assert observer.calls[0]["q"].dtype == torch.bfloat16
