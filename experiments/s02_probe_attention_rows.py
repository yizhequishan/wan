"""Replay Wan generation and save only manually selected attention rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Allow direct execution from a source checkout without installing Wan.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.analysis import (  # noqa: E402
    SelectedRowAttentionProbe,
    load_row_probe_request,
)
from wan.analysis.row_attention_viz import render_probe_directory  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one Wan T2V sample and compute q_selected @ K^T rows after RoPE."
        )
    )
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--size", type=str, default=None)
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument("--sample-solver", type=str, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--guide-scale", type=float, default=None)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=(14,),
        help="Zero-based transformer block indices.",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        nargs="+",
        default=(10, 25, 40),
        help="Zero-based denoising-step indices.",
    )
    parser.add_argument("--branch", choices=("cond", "uncond"), default="cond")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument(
        "--offload-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--render-video",
        type=Path,
        default=None,
        help=(
            "Optional decoded reference video for overlays. By default the "
            "deterministic replay is saved and used."
        ),
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--head-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a 12-panel MP4 for every selected query and observation.",
    )
    parser.add_argument("--head-video-fps", type=float, default=4.0)
    return parser.parse_args()


def _setting(
    cli_value: Any,
    generation: dict[str, Any],
    name: str,
    default: Any,
    cast: Callable[[Any], Any],
) -> Any:
    value = cli_value if cli_value is not None else generation.get(name, default)
    return cast(value)


def main() -> None:
    args = _parse_args()
    request = load_row_probe_request(args.query_file)
    if not request.prompt.strip():
        raise ValueError("query file prompt is empty")

    # Keep heavyweight Wan/diffusers/torchvision imports after argument parsing
    # so ``--help`` and lightweight query-file tools work without model deps.
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import cache_video

    generation = dict(request.generation)
    task = _setting(args.task, generation, "task", "t2v-1.3B", str)
    size_name = _setting(args.size, generation, "size", "832*480", str)
    frame_num = _setting(
        args.frame_num,
        generation,
        "frame_num",
        request.video_shape.frames,
        int,
    )
    sample_solver = _setting(
        args.sample_solver,
        generation,
        "sample_solver",
        "unipc",
        str,
    )
    sampling_steps = _setting(
        args.sampling_steps,
        generation,
        "sampling_steps",
        50,
        int,
    )
    shift = _setting(args.shift, generation, "shift", 8.0, float)
    guide_scale = _setting(
        args.guide_scale,
        generation,
        "guide_scale",
        6.0,
        float,
    )

    if task not in {"t2v-1.3B", "t2v-14B"}:
        raise ValueError("row probe currently supports t2v-1.3B or t2v-14B")
    if size_name not in SIZE_CONFIGS:
        raise ValueError(f"unknown size: {size_name}")
    if sample_solver not in {"unipc", "dpm++"}:
        raise ValueError(f"unsupported sample solver: {sample_solver}")
    if frame_num != request.video_shape.frames:
        raise ValueError(
            f"frame_num={frame_num} differs from query video "
            f"frames={request.video_shape.frames}"
        )
    width, height = SIZE_CONFIGS[size_name]
    if (height, width) != (
        request.video_shape.height,
        request.video_shape.width,
    ):
        raise ValueError(
            f"size {size_name} resolves to {(width, height)}, but query video "
            f"is {(request.video_shape.width, request.video_shape.height)}"
        )

    config = WAN_CONFIGS[task]
    invalid_layers = [
        layer for layer in args.layers if not 0 <= int(layer) < int(config.num_layers)
    ]
    if invalid_layers:
        raise ValueError(
            f"layer indices {invalid_layers} are outside "
            f"[0,{int(config.num_layers) - 1}]"
        )
    invalid_steps = [
        step for step in args.record_steps if not 0 <= int(step) < sampling_steps
    ]
    if invalid_steps:
        raise ValueError(
            f"denoising steps {invalid_steps} are outside " f"[0,{sampling_steps - 1}]"
        )

    raw_dir = args.output_dir / "raw"
    records_path = raw_dir / "records.jsonl"
    if records_path.exists():
        raise FileExistsError(
            f"{records_path} already exists; use a new --output-dir to avoid "
            "mixing observations from different runs"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe = SelectedRowAttentionProbe(
        queries=request.queries,
        decoded_shape=request.video_shape,
        layers=args.layers,
        record_steps=args.record_steps,
        output_dir=raw_dir,
        branch=args.branch,
        top_k=args.top_k,
    )

    run_config = {
        "query_file": str(args.query_file.resolve()),
        "checkpoint_dir": str(args.ckpt_dir.resolve()),
        "video_id": request.video_id,
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "seed": request.seed,
        "task": task,
        "size": size_name,
        "frame_num": frame_num,
        "sample_solver": sample_solver,
        "sampling_steps": sampling_steps,
        "shift": shift,
        "guide_scale": guide_scale,
        "layers": sorted(set(int(layer) for layer in args.layers)),
        "record_steps": sorted(set(int(step) for step in args.record_steps)),
        "branch": args.branch,
        "num_queries": len(request.queries),
        "attention_formula": (
            "softmax((q_selected @ K^T) / sqrt(head_dim)), per head, after RoPE"
        ),
    }
    run_config_path = args.output_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"Loading {task}; observing layers={run_config['layers']} at "
        f"steps={run_config['record_steps']} ({args.branch} branch)."
    )
    pipeline = WanT2V(
        config,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )
    model = (
        pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
    )
    model.analysis_sidecar = probe
    try:
        video = pipeline.generate(
            input_prompt=request.prompt,
            size=SIZE_CONFIGS[size_name],
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=request.negative_prompt,
            seed=request.seed,
            offload_model=args.offload_model,
            analysis_video_id=request.video_id,
        )
    finally:
        model.analysis_sidecar = None

    expected_observations = len(set(args.layers)) * len(set(args.record_steps))
    if len(probe.records) != expected_observations:
        raise RuntimeError(
            f"expected {expected_observations} observations, "
            f"but probe recorded {len(probe.records)}"
        )

    replay_path = args.output_dir / "replay.mp4"
    saved_path = cache_video(
        tensor=video[None],
        save_file=str(replay_path),
        fps=config.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    if saved_path is None:
        raise RuntimeError(f"failed to save deterministic replay to {replay_path}")
    print(f"Saved replay: {replay_path}")
    print(f"Saved {len(probe.records)} raw attention observation(s): {raw_dir}")

    if not args.no_render:
        render_video = args.render_video or replay_path
        outputs = render_probe_directory(
            results_dir=raw_dir,
            video_path=render_video,
            output_dir=args.output_dir / "rendered",
            write_head_videos=args.head_videos,
            head_video_fps=args.head_video_fps,
        )
        print(f"Rendered {len(outputs)} artifact(s): {args.output_dir / 'rendered'}")


if __name__ == "__main__":
    main()
