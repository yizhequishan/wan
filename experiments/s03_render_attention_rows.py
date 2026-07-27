"""Render previously saved selected-row attention volumes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution from a source checkout without installing Wan.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan.analysis.row_attention_viz import render_probe_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--head-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--head-video-fps", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.58)
    args = parser.parse_args()

    outputs = render_probe_directory(
        results_dir=args.results_dir,
        video_path=args.video,
        output_dir=args.output_dir,
        write_head_videos=args.head_videos,
        head_video_fps=args.head_video_fps,
        alpha=args.alpha,
    )
    print(f"Rendered {len(outputs)} artifact(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
