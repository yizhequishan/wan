# Wan selected-row attention pilot

This is a deliberately small falsification test. Wan still computes its
original dense FlashAttention output. A read-only observer runs after RoPE and
computes only

```text
softmax((q_selected @ K^T) / sqrt(head_dim))
```

for a few manually selected query tokens. It never constructs an `L x L`
attention matrix and does not use SVG2, K-means, SAM, or Kubric.

## 1. Generate one reference video with the existing entry point

Use the 1.3B model, 832x480, 81 frames, and a fixed seed. Keep every generation
argument unchanged for the replay.

```powershell
$WanCkpt = "D:\models\Wan2.1-T2V-1.3B"
$Prompt = "A locked-off camera view of two glossy balls on a plain light-gray floor. A red ball moves from left to right and a blue ball moves from right to left. They collide once near the center and bounce apart. Both balls remain visible, continuous motion, simple empty background, no cuts, no camera movement."

python generate.py `
  --task t2v-1.3B `
  --size "832*480" `
  --frame_num 81 `
  --ckpt_dir $WanCkpt `
  --offload_model True `
  --t5_cpu `
  --prompt $Prompt `
  --base_seed 20250308 `
  --sample_solver unipc `
  --sample_steps 50 `
  --sample_shift 8.0 `
  --sample_guide_scale 6.0 `
  --save_file "outputs\row_probe\reference.mp4"
```

Inspect the result first. If it does not contain two continuously visible
objects and a reasonably clear approach/contact/separation event, change only
the seed and regenerate. This pilot cannot answer the intended question from a
bad reference sample.

## 2. Export token-grid frames and select queries

Choose frames before, near, and after contact:

```powershell
python experiments\s01_make_row_query_sheet.py `
  --video "outputs\row_probe\reference.mp4" `
  --output-dir "outputs\row_probe\query_sheet" `
  --frames 16 24 32 40 48 56 64 `
  --video-id "two_balls_seed_20250308" `
  --prompt $Prompt `
  --seed 20250308
```

The script writes grid PNGs and
`outputs\row_probe\query_sheet\queries.template.json`. Fill its `queries`
array with points visibly inside the objects:

```json
{
  "name": "red_before",
  "frame": 24,
  "x": 238,
  "y": 267
}
```

Coordinates are zero-based decoded-video `(frame, x, y)` coordinates. At this
resolution, spatial cells are 16x16 pixels. Wan's causal VAE maps decoded frame
0 to key frame 0 and every later decoded frame to `ceil(frame / 4)`. The
probe performs this mapping itself. Prefer object interiors, at least one cell
away from boundaries. Pick approximately:

- one point in each object before contact;
- one point in each object at contact;
- one point in each object after contact.

The example file
`experiments/configs/row_attention_queries.example.json` is syntactically
complete, but its coordinates are illustrative and must be replaced after
looking at the actual generated video.

## 3. Replay the same sample and collect rows

For the first pass, keep one middle block (zero-based block 14) and inspect
three denoising stages:

```powershell
python experiments\s02_probe_attention_rows.py `
  --query-file "outputs\row_probe\query_sheet\queries.template.json" `
  --ckpt-dir $WanCkpt `
  --output-dir "outputs\row_probe\mid_layer_probe" `
  --layers 14 `
  --record-steps 10 25 40 `
  --t5-cpu `
  --offload-model
```

The query JSON supplies the prompt, seed, resolution, solver, step count,
shift, and guidance scale. The replay script rejects mismatched video shapes
and refuses to append to an existing `records.jsonl`.

For 832x480 and 81 frames, every raw observation contains:

```text
attention_per_head: [Q, 12, 21, 30, 52] float32
attention_head_mean: [Q, 21, 30, 52] float32
temporal_mass_per_head: [Q, 12, 21] float32
```

Every `[Q, head]` attention row sums to one. Outputs are:

- `raw/<video_id>/step_...npz`: linear-probability scientific data;
- `raw/records.jsonl`: shapes, query-to-token mapping, layer/step, and scale;
- `replay.mp4`: the decoded video corresponding to the observed Q/K;
- `rendered/*_mean_fhw.png`: all 21 key-frame slices in one contact sheet;
- `rendered/*_12heads_fhw.mp4`: all 12 heads, animated over key frames;
- `rendered/*_temporal_mass.png`: attention mass assigned to each key frame.

Rendered heatmaps use one robust log-probability scale for the complete
query/head volume. They are not normalized separately per frame. Use the NPZ
arrays for quantitative comparisons.

Rendering can be repeated without another Wan run:

```powershell
python experiments\s03_render_attention_rows.py `
  --results-dir "outputs\row_probe\mid_layer_probe\raw" `
  --video "outputs\row_probe\mid_layer_probe\replay.mp4" `
  --output-dir "outputs\row_probe\mid_layer_probe\rendered_again"
```

## What counts as evidence

First inspect individual heads; a head mean can erase a sparse trajectory.

- Trajectory-like evidence: for a query inside object A at frame `f_q`, one or
  more heads put coherent mass on A's positions at several other key frames.
  The high-mass locations should move with A, rather than staying at the
  query's fixed image coordinate.
- Interaction-like evidence: for contact-frame queries, some otherwise
  object-A-following heads transfer a visible, reproducible fraction of mass
  to object B. Check nearby interior queries so a single boundary token does
  not decide the result. If both objects have selected contact-frame points,
  `selected_anchor_3x3_same_frame_mass_per_head[source_query, head,
  target_point]` gives a small numeric cross-check of the visible transfer.
- Pure locality: most mass remains in the query frame and the local 3x3 or
  3x3x3 neighborhood. The NPZ includes
  `local_3x3_same_frame_mass_per_head` and
  `local_3x3x3_mass_per_head` for this check.
- Diffuse rows: high normalized entropy, weak spatial peaks, and nearly flat
  temporal mass. Log-scaled pictures alone can make tiny fluctuations look
  salient, so always check the raw mass and entropy arrays.

Step 0 is the high-noise end of Wan's reverse path; larger step indices are
later/cleaner. If the middle-block pilot is negative, run one controlled
early/middle/late sweep:

```powershell
python experiments\s02_probe_attention_rows.py `
  --query-file "outputs\row_probe\query_sheet\queries.template.json" `
  --ckpt-dir $WanCkpt `
  --output-dir "outputs\row_probe\layer_step_sweep" `
  --layers 3 14 26 `
  --record-steps 5 15 25 35 45 `
  --t5-cpu `
  --offload-model
```

If nearby object-interior queries remain diffuse or purely local across that
sweep, the proposed entity-interaction graph lacks its necessary raw
attention structure in this model/sample. A positive result is only evidence
that the structure exists; it is not yet evidence of physical causality.
