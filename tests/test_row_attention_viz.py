import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from wan.analysis.row_attention_viz import (
    render_probe_directory,
    write_query_sheets,
)


def _tiny_video(path: Path) -> None:
    frames = []
    for index in range(5):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, index % 3] = 40 + index * 30
        frames.append(frame)
    imageio.mimsave(path, frames, format="GIF", duration=0.1)


def test_query_sheet_and_static_probe_render(tmp_path: Path):
    video_path = tmp_path / "video.gif"
    _tiny_video(video_path)
    sheets, shape = write_query_sheets(
        video_path=video_path,
        frame_indices=(0, 4),
        output_dir=tmp_path / "sheets",
        spatial_token_pixels=2,
        label_every_tokens=1,
    )
    assert shape.to_dict() == {"frames": 5, "height": 4, "width": 4}
    assert len(sheets) == 2
    assert all(path.exists() for path in sheets)

    results_dir = tmp_path / "raw"
    array_dir = results_dir / "sample"
    array_dir.mkdir(parents=True)
    probabilities = np.arange(1, 17, dtype=np.float32).reshape(1, 2, 8)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    volume = probabilities.reshape(1, 2, 2, 2, 2)
    array_path = array_dir / "step_000_layer_01_cond.npz"
    np.savez_compressed(
        array_path,
        attention_per_head=volume,
        attention_head_mean=volume.mean(axis=1),
        temporal_mass_per_head=volume.sum(axis=(-1, -2)),
        normalized_entropy_per_head=np.full((1, 2), 0.5, dtype=np.float32),
    )
    record = {
        "probe": "selected_attention_rows",
        "video_id": "sample",
        "branch": "cond",
        "step_index": 0,
        "timestep": 999.0,
        "layer_idx": 1,
        "grid_size": [2, 2, 2],
        "decoded_video_shape": {"frames": 5, "height": 4, "width": 4},
        "queries": [
            {
                "name": "object",
                "source": {"name": "object", "token": [1, 0, 1]},
                "token_fhw": [1, 0, 1],
                "flat_index": 5,
            }
        ],
        "array_file": str(array_path.relative_to(results_dir)),
    }
    (results_dir / "records.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    outputs = render_probe_directory(
        results_dir=results_dir,
        video_path=video_path,
        output_dir=tmp_path / "rendered",
        write_head_videos=False,
    )
    assert len(outputs) == 3
    contact = tmp_path / "rendered" / "step_000_layer_01_cond" / "object_mean_fhw.png"
    assert contact.exists()
    with Image.open(contact) as image:
        assert image.size == (512, 160)
    index = json.loads(
        (tmp_path / "rendered" / "render_index.json").read_text(encoding="utf-8")
    )
    assert index[0]["all_heads_video"] is None
