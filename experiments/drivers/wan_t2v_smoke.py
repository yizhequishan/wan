"""Standard Wan T2V generation driver for sidecar wiring smoke tests only."""

from __future__ import annotations

from typing import Any

from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.text2video import WanT2V


_PIPELINE: WanT2V | None = None
_PIPELINE_KEY: tuple[Any, ...] | None = None


def _pipeline(driver_args: dict[str, Any]) -> WanT2V:
    global _PIPELINE, _PIPELINE_KEY

    checkpoint_dir = str(driver_args["checkpoint_dir"])
    task = str(driver_args.get("task", "t2v-1.3B"))
    if task not in {"t2v-1.3B", "t2v-14B"}:
        raise ValueError("wan_t2v_smoke only supports Wan text-to-video")
    key = (
        checkpoint_dir,
        task,
        int(driver_args.get("device_id", 0)),
        bool(driver_args.get("t5_cpu", False)),
    )
    if _PIPELINE is None or _PIPELINE_KEY != key:
        _PIPELINE = WanT2V(
            WAN_CONFIGS[task],
            checkpoint_dir=checkpoint_dir,
            device_id=key[2],
            t5_cpu=key[3],
            use_usp=False,
            t5_fsdp=False,
            dit_fsdp=False,
        )
        _PIPELINE_KEY = key
    return _PIPELINE


def run_record(*, record, sidecar, driver_args):
    """Generate one video while observing Q/K.

    This driver does not reconstruct a Kubric video. It intentionally refuses
    entity labels unless the caller explicitly opts into an unmatched-mask
    wiring test.
    """

    if record.get("entity_ids") is not None and not driver_args.get(
        "allow_unmatched_entity_labels", False
    ):
        raise ValueError(
            "wan_t2v_smoke generates a new video, so Kubric entity labels do "
            "not match it. Remove entity_ids or use the inversion driver."
        )

    pipeline = _pipeline(driver_args)
    model = (
        pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
    )
    model.analysis_sidecar = sidecar

    size_name = str(driver_args.get("size", "832*480"))
    if size_name not in SIZE_CONFIGS:
        raise ValueError(f"Unknown Wan size: {size_name}")
    size = SIZE_CONFIGS[size_name]

    return pipeline.generate(
        input_prompt=str(record.get("prompt", "")),
        size=size,
        frame_num=int(driver_args.get("frame_num", 81)),
        shift=float(driver_args.get("shift", 5.0)),
        sample_solver=str(driver_args.get("sample_solver", "unipc")),
        sampling_steps=int(driver_args.get("sampling_steps", 50)),
        guide_scale=float(driver_args.get("guide_scale", 5.0)),
        n_prompt=str(record.get("negative_prompt", "")),
        seed=int(record.get("seed", 0)),
        offload_model=bool(driver_args.get("offload_model", True)),
        analysis_video_id=str(record["video_id"]),
    )
