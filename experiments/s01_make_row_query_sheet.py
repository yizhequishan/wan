"""Make token-grid frames for manually selecting Wan attention queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from a source checkout without installing Wan.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.analysis.row_attention_viz import write_query_sheets  # noqa: E402


DEFAULT_PROMPT = (
    "A locked-off camera view of two glossy balls on a plain light-gray floor. "
    "A red ball moves from left to right and a blue ball moves from right to "
    "left. They collide once near the center and bounce apart. Both balls "
    "remain visible, continuous motion, simple empty background, no cuts, "
    "no camera movement."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay Wan's 16x16 spatial token grid on selected decoded frames."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=(0, 20, 40, 60, 80),
        help="Zero-based decoded frame indices to export.",
    )
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=20250308)
    parser.add_argument("--task", type=str, default="t2v-1.3B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--sample-solver", type=str, default="unipc")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--guide-scale", type=float, default=6.0)
    parser.add_argument(
        "--template-file",
        type=Path,
        default=None,
        help="Default: OUTPUT_DIR/queries.template.json",
    )
    parser.add_argument(
        "--overwrite-template",
        action="store_true",
        help="Replace an existing template file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs, video_shape = write_query_sheets(
        video_path=args.video,
        frame_indices=args.frames,
        output_dir=args.output_dir,
    )
    template_path = args.template_file
    if template_path is None:
        template_path = args.output_dir / "queries.template.json"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    if template_path.exists() and not args.overwrite_template:
        print(f"Kept existing query template: {template_path}")
    else:
        template = {
            "video_id": args.video_id or args.video.stem,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "video_shape": video_shape.to_dict(),
            "generation": {
                "task": args.task,
                "size": args.size,
                "frame_num": video_shape.frames,
                "sample_solver": args.sample_solver,
                "sampling_steps": args.sampling_steps,
                "shift": args.shift,
                "guide_scale": args.guide_scale,
            },
            "queries": [],
        }
        with template_path.open("w", encoding="utf-8") as handle:
            json.dump(template, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Wrote query template: {template_path}")

    print(f"Wrote {len(outputs)} token-grid frame(s) to {args.output_dir}")
    print(
        "Fill queries with decoded coordinates, for example: "
        '{"name":"red_pre","frame":20,"x":240,"y":260}'
    )


if __name__ == "__main__":
    main()
