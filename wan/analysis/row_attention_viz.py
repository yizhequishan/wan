"""Visualization helpers for selected-row Wan attention volumes."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .row_attention_probe import DecodedVideoShape


_COLOR_STOPS = np.asarray(
    [
        [5, 8, 24],
        [31, 67, 172],
        [20, 174, 194],
        [244, 222, 66],
        [204, 31, 35],
    ],
    dtype=np.float32,
)


def _safe_component(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return normalized or "unnamed"


def _read_video(path: str | Path) -> list[np.ndarray]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video visualization requires imageio and imageio-ffmpeg."
        ) from exc

    frames: list[np.ndarray] = []
    reader = imageio.get_reader(str(path))
    try:
        for frame in reader:
            rgb = np.asarray(frame)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                raise ValueError(f"unsupported video frame shape: {rgb.shape}")
            frames.append(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8))
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"video has no readable frames: {path}")
    return frames


def _write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Video visualization requires imageio and imageio-ffmpeg."
        ) from exc

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
    )
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _to_rgb(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB")


def _log_normalize(volume: np.ndarray) -> np.ndarray:
    """Normalize one complete volume without rescaling individual frames."""

    values = np.asarray(volume, dtype=np.float32)
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values)
    log_values = np.log10(np.maximum(values, np.finfo(np.float32).tiny))
    positive_logs = np.log10(positive)
    low = float(np.percentile(positive_logs, 2.0))
    high = float(np.percentile(positive_logs, 99.75))
    if high <= low + 1e-8:
        return np.full_like(values, 0.5)
    return np.clip((log_values - low) / (high - low), 0.0, 1.0)


def _colorize(normalized: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    scaled = values * (_COLOR_STOPS.shape[0] - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.minimum(lower + 1, _COLOR_STOPS.shape[0] - 1)
    fraction = (scaled - lower)[..., None]
    rgb = _COLOR_STOPS[lower] * (1.0 - fraction) + _COLOR_STOPS[upper] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _resize_rgb(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    return np.asarray(image.resize(size, resample=Image.Resampling.BILINEAR))


def _overlay(
    frame: np.ndarray,
    normalized_heatmap: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.uint8)
    heatmap = _colorize(normalized_heatmap)
    heatmap = _resize_rgb(heatmap, (frame.shape[1], frame.shape[0]))
    blended = frame.astype(np.float32) * (1.0 - alpha)
    blended += heatmap.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _representative_decoded_frame(
    token_frame: int,
    *,
    decoded_frames: int,
    temporal_stride: int = 4,
) -> int:
    if token_frame == 0:
        return 0
    return min(
        temporal_stride * token_frame - temporal_stride // 2,
        decoded_frames - 1,
    )


def _query_pixel(
    query_record: Mapping[str, Any],
    *,
    decoded_shape: DecodedVideoShape,
    grid_size: tuple[int, int, int],
) -> tuple[float, float]:
    source = query_record.get("source", {})
    if "x" in source and "y" in source:
        return float(source["x"]), float(source["y"])
    _, token_h, token_w = query_record["token_fhw"]
    return (
        (float(token_w) + 0.5) * decoded_shape.width / grid_size[2],
        (float(token_h) + 0.5) * decoded_shape.height / grid_size[1],
    )


def _annotated_panel(
    image: np.ndarray,
    *,
    label: str,
    panel_size: tuple[int, int],
    marker: tuple[float, float] | None = None,
    marker_source_size: tuple[int, int] | None = None,
    highlighted: bool = False,
) -> np.ndarray:
    panel = _to_rgb(image).resize(panel_size, resample=Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle((0, 0, panel_size[0], 21), fill=(0, 0, 0, 190))
    draw.text((5, 5), label, fill=(255, 255, 255, 255))
    border = (255, 215, 0, 255) if highlighted else (255, 255, 255, 100)
    draw.rectangle(
        (0, 0, panel_size[0] - 1, panel_size[1] - 1),
        outline=border,
        width=2 if highlighted else 1,
    )
    if marker is not None and marker_source_size is not None:
        marker_x = marker[0] * panel_size[0] / marker_source_size[0]
        marker_y = marker[1] * panel_size[1] / marker_source_size[1]
        radius = 5
        draw.ellipse(
            (
                marker_x - radius,
                marker_y - radius,
                marker_x + radius,
                marker_y + radius,
            ),
            outline=(255, 255, 255, 255),
            width=2,
        )
    return np.asarray(panel)


def write_query_sheets(
    *,
    video_path: str | Path,
    frame_indices: Sequence[int],
    output_dir: str | Path,
    spatial_token_pixels: int = 16,
    label_every_tokens: int = 4,
) -> tuple[list[Path], DecodedVideoShape]:
    """Write decoded frames with Wan token-cell grid coordinates."""

    if spatial_token_pixels <= 0 or label_every_tokens <= 0:
        raise ValueError("grid spacing and label interval must be positive")
    frames = _read_video(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for frame_index in frame_indices:
        if not 0 <= frame_index < len(frames):
            raise ValueError(
                f"frame {frame_index} is outside video range [0,{len(frames) - 1}]"
            )
        image = _to_rgb(frames[frame_index])
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        for x in range(0, width, spatial_token_pixels):
            line_width = 2 if x % (spatial_token_pixels * 4) == 0 else 1
            draw.line((x, 0, x, height), fill=(0, 220, 255, 140), width=line_width)
        for y in range(0, height, spatial_token_pixels):
            line_width = 2 if y % (spatial_token_pixels * 4) == 0 else 1
            draw.line((0, y, width, y), fill=(0, 220, 255, 140), width=line_width)

        label_stride = spatial_token_pixels * label_every_tokens
        for y in range(0, height, label_stride):
            for x in range(0, width, label_stride):
                draw.rectangle((x + 1, y + 1, x + 49, y + 13), fill=(0, 0, 0, 150))
                draw.text(
                    (x + 3, y + 2),
                    f"h{y // spatial_token_pixels},w{x // spatial_token_pixels}",
                    fill=(255, 255, 255, 255),
                )
        token_frame = 0 if frame_index == 0 else math.ceil(frame_index / 4)
        draw.rectangle((0, height - 24, width, height), fill=(0, 0, 0, 190))
        draw.text(
            (6, height - 19),
            (
                f"decoded frame={frame_index} -> token f={token_frame}; "
                f"pixel (x,y) maps to token "
                f"(f, floor(y/{spatial_token_pixels}), "
                f"floor(x/{spatial_token_pixels}))"
            ),
            fill=(255, 255, 255, 255),
        )
        output_path = output_dir / f"frame_{frame_index:03d}_token_grid.png"
        image.save(output_path)
        outputs.append(output_path)
    shape = DecodedVideoShape(
        frames=len(frames),
        height=int(frames[0].shape[0]),
        width=int(frames[0].shape[1]),
    )
    return outputs, shape


def _contact_sheet(
    *,
    normalized_volume: np.ndarray,
    linear_volume: np.ndarray,
    video_frames: Sequence[np.ndarray],
    query_record: Mapping[str, Any],
    decoded_shape: DecodedVideoShape,
    grid_size: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    token_frames = normalized_volume.shape[0]
    columns = 7 if token_frames > 7 else token_frames
    rows = math.ceil(token_frames / columns)
    panel_size = (256, 160)
    canvas = Image.new(
        "RGB",
        (columns * panel_size[0], rows * panel_size[1]),
        color=(12, 12, 16),
    )
    query_token_f = int(query_record["token_fhw"][0])
    marker = _query_pixel(
        query_record,
        decoded_shape=decoded_shape,
        grid_size=grid_size,
    )
    for token_f in range(token_frames):
        decoded_f = _representative_decoded_frame(
            token_f,
            decoded_frames=len(video_frames),
        )
        overlay = _overlay(
            video_frames[decoded_f],
            normalized_volume[token_f],
            alpha=alpha,
        )
        mass = float(linear_volume[token_f].sum())
        panel = _annotated_panel(
            overlay,
            label=f"key f={token_f:02d} | frame={decoded_f:03d} | mass={mass:.3f}",
            panel_size=panel_size,
            marker=marker if token_f == query_token_f else None,
            marker_source_size=(decoded_shape.width, decoded_shape.height),
            highlighted=token_f == query_token_f,
        )
        canvas.paste(
            _to_rgb(panel),
            (
                (token_f % columns) * panel_size[0],
                (token_f // columns) * panel_size[1],
            ),
        )
    return np.asarray(canvas)


def _head_video_frames(
    *,
    per_head_volume: np.ndarray,
    video_frames: Sequence[np.ndarray],
    query_record: Mapping[str, Any],
    decoded_shape: DecodedVideoShape,
    grid_size: tuple[int, int, int],
    entropy_per_head: np.ndarray,
    alpha: float,
) -> Iterable[np.ndarray]:
    num_heads, token_frames = per_head_volume.shape[:2]
    columns = 4
    rows = math.ceil(num_heads / columns)
    panel_size = (256, 160)
    normalized_heads = np.stack(
        [_log_normalize(per_head_volume[head]) for head in range(num_heads)]
    )
    query_token_f = int(query_record["token_fhw"][0])
    marker = _query_pixel(
        query_record,
        decoded_shape=decoded_shape,
        grid_size=grid_size,
    )
    for token_f in range(token_frames):
        decoded_f = _representative_decoded_frame(
            token_f,
            decoded_frames=len(video_frames),
        )
        canvas = Image.new(
            "RGB",
            (columns * panel_size[0], rows * panel_size[1]),
            color=(12, 12, 16),
        )
        for head in range(num_heads):
            overlay = _overlay(
                video_frames[decoded_f],
                normalized_heads[head, token_f],
                alpha=alpha,
            )
            mass = float(per_head_volume[head, token_f].sum())
            panel = _annotated_panel(
                overlay,
                label=(
                    f"h{head:02d} | f={token_f:02d} | "
                    f"mass={mass:.3f} | Hn={entropy_per_head[head]:.2f}"
                ),
                panel_size=panel_size,
                marker=marker if token_f == query_token_f else None,
                marker_source_size=(decoded_shape.width, decoded_shape.height),
                highlighted=token_f == query_token_f,
            )
            canvas.paste(
                _to_rgb(panel),
                (
                    (head % columns) * panel_size[0],
                    (head // columns) * panel_size[1],
                ),
            )
        yield np.asarray(canvas)


def _temporal_mass_chart(
    *,
    temporal_mass: np.ndarray,
    query_token_frame: int,
    size: tuple[int, int] = (900, 300),
) -> np.ndarray:
    """Draw all heads plus their mean without requiring matplotlib."""

    values = np.asarray(temporal_mass, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("temporal_mass must be [heads, token_frames]")
    width, height = size
    margin = (52, 20, 20, 34)
    plot_width = width - margin[0] - margin[2]
    plot_height = height - margin[1] - margin[3]
    canvas = Image.new("RGB", size, color=(250, 250, 248))
    draw = ImageDraw.Draw(canvas, "RGBA")
    maximum = max(float(values.max()), 1e-8)
    token_frames = values.shape[1]

    def point(token_f: int, mass: float) -> tuple[float, float]:
        x = margin[0] + token_f * plot_width / max(token_frames - 1, 1)
        y = margin[1] + plot_height * (1.0 - mass / maximum)
        return x, y

    for level in range(5):
        y = margin[1] + level * plot_height / 4
        draw.line(
            (margin[0], y, width - margin[2], y),
            fill=(0, 0, 0, 35),
            width=1,
        )
    query_x, _ = point(query_token_frame, 0.0)
    draw.line(
        (query_x, margin[1], query_x, margin[1] + plot_height),
        fill=(220, 40, 35, 190),
        width=2,
    )
    for head_values in values:
        points = [point(index, float(mass)) for index, mass in enumerate(head_values)]
        draw.line(points, fill=(35, 99, 180, 65), width=2)
    mean_values = values.mean(axis=0)
    mean_points = [point(index, float(mass)) for index, mass in enumerate(mean_values)]
    draw.line(mean_points, fill=(10, 30, 60, 255), width=4)
    draw.text((6, 8), f"mass (max={maximum:.3f})", fill=(10, 10, 10, 255))
    draw.text(
        (margin[0], height - 24),
        ("key-frame index; blue=individual heads, black=head mean, " "red=query frame"),
        fill=(10, 10, 10, 255),
    )
    return np.asarray(canvas)


def _load_probe_records(results_dir: Path) -> list[dict[str, Any]]:
    records_path = results_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"missing probe records: {records_path}")
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("probe") == "selected_attention_rows":
                records.append(record)
            elif "probe" in record:
                continue
            else:
                raise ValueError(
                    f"{records_path}:{line_number} is not a row-probe record"
                )
    if not records:
        raise ValueError(f"no selected-attention-row records in {records_path}")
    return records


def render_probe_directory(
    *,
    results_dir: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
    write_head_videos: bool = True,
    head_video_fps: float = 4.0,
    alpha: float = 0.58,
) -> list[Path]:
    """Render head-mean contact sheets and optional all-head videos."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_frames = _read_video(video_path)
    records = _load_probe_records(results_dir)
    outputs: list[Path] = []
    index_rows: list[dict[str, Any]] = []

    for record in records:
        decoded_shape = DecodedVideoShape.from_value(record["decoded_video_shape"])
        if len(video_frames) != decoded_shape.frames or video_frames[0].shape[:2] != (
            decoded_shape.height,
            decoded_shape.width,
        ):
            probe_shape = (
                decoded_shape.frames,
                decoded_shape.height,
                decoded_shape.width,
            )
            raise ValueError(
                "render video shape does not match probe metadata: "
                f"video={(len(video_frames), *video_frames[0].shape[:2])}, "
                f"probe={probe_shape}"
            )
        grid_size = tuple(int(value) for value in record["grid_size"])
        array_path = results_dir / record["array_file"]
        observation_name = (
            f"step_{int(record['step_index']):03d}"
            f"_layer_{int(record['layer_idx']):02d}"
            f"_{_safe_component(record['branch'])}"
        )
        observation_dir = output_dir / observation_name
        observation_dir.mkdir(parents=True, exist_ok=True)
        with np.load(array_path, allow_pickle=False) as arrays:
            per_head = np.asarray(arrays["attention_per_head"], dtype=np.float32)
            head_mean = np.asarray(arrays["attention_head_mean"], dtype=np.float32)
            temporal_mass = np.asarray(
                arrays["temporal_mass_per_head"],
                dtype=np.float32,
            )
            normalized_entropy = np.asarray(
                arrays["normalized_entropy_per_head"],
                dtype=np.float32,
            )

        for query_index, query_record in enumerate(record["queries"]):
            query_name = _safe_component(query_record["name"])
            normalized_mean = _log_normalize(head_mean[query_index])
            contact_sheet = _contact_sheet(
                normalized_volume=normalized_mean,
                linear_volume=head_mean[query_index],
                video_frames=video_frames,
                query_record=query_record,
                decoded_shape=decoded_shape,
                grid_size=grid_size,
                alpha=alpha,
            )
            contact_path = observation_dir / f"{query_name}_mean_fhw.png"
            _to_rgb(contact_sheet).save(contact_path)
            outputs.append(contact_path)

            mass_chart = _temporal_mass_chart(
                temporal_mass=temporal_mass[query_index],
                query_token_frame=int(query_record["token_fhw"][0]),
            )
            mass_path = observation_dir / f"{query_name}_temporal_mass.png"
            _to_rgb(mass_chart).save(mass_path)
            outputs.append(mass_path)

            head_video_path: Path | None = None
            if write_head_videos:
                head_video_path = observation_dir / f"{query_name}_12heads_fhw.mp4"
                _write_video(
                    head_video_path,
                    _head_video_frames(
                        per_head_volume=per_head[query_index],
                        video_frames=video_frames,
                        query_record=query_record,
                        decoded_shape=decoded_shape,
                        grid_size=grid_size,
                        entropy_per_head=normalized_entropy[query_index],
                        alpha=alpha,
                    ),
                    fps=head_video_fps,
                )
                outputs.append(head_video_path)

            index_rows.append(
                {
                    "step_index": int(record["step_index"]),
                    "timestep": record["timestep"],
                    "layer_idx": int(record["layer_idx"]),
                    "branch": record["branch"],
                    "query": query_record,
                    "contact_sheet": str(contact_path.relative_to(output_dir)),
                    "temporal_mass_chart": str(mass_path.relative_to(output_dir)),
                    "all_heads_video": (
                        str(head_video_path.relative_to(output_dir))
                        if head_video_path is not None
                        else None
                    ),
                    "normalization": (
                        "log10 probability, one robust scale for the complete "
                        "query/head volume; raw NPZ values remain linear"
                    ),
                }
            )

    index_path = output_dir / "render_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    outputs.append(index_path)
    return outputs
