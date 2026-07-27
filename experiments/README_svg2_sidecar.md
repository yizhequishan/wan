# SVG2 read-only sidecar pilot

The sidecar observes post-RoPE self-attention Q/K tensors. Dense
FlashAttention still produces the model output. For the six selected layers,
centroids are updated at every conditional denoising step; metrics are written
only at the six configured record steps.

## Manifest

Use one JSON object per line:

```json
{"video_id":"kubric_000","entity_ids":"tokens/kubric_000.npy","grid_size":[21,30,52],"trajectory":"inversion","prompt":"...","seed":0}
```

`entity_ids` is a flattened or `[F,H,W]` integer NumPy array:

- `-2`: boundary, excluded from statistics;
- `-1`: background;
- `0..E-1`: Kubric entities.

The labels must describe the exact latent trajectory processed by the driver.
A newly generated video must not be paired with masks from an unrelated
Kubric render.

## Validate before loading Wan

```bash
python experiments/s03a_extract_svg2_graph.py \
  --config experiments/configs/svg2_pilot_c128_k512.yaml \
  --manifest data/kubric/pilot20.jsonl \
  --check-only
```

## Trajectory driver contract

The formal pilot should pass the Phase-2 inversion driver:

```bash
python experiments/s03a_extract_svg2_graph.py \
  --config experiments/configs/svg2_pilot_c128_k512.yaml \
  --manifest data/kubric/pilot20.jsonl \
  --driver your_inversion_package.kubric_driver:run_record \
  --driver-args '{"checkpoint_dir":"..."}'
```

The callable signature is:

```python
def run_record(*, record, sidecar, driver_args):
    model = load_or_reuse_wan(driver_args)
    model.analysis_sidecar = sidecar
    replay_the_matching_trajectory(record, model)
```

The driver must pass `analysis_ctx` through the model, or call the modified
`WanT2V.generate(..., analysis_video_id=record["video_id"])` interface.
Generation is useful for wiring smoke tests; the entity-alignment claim
requires a Kubric-aligned inversion or controlled noising trajectory.

For a no-mask wiring test, use the included generation driver:

```bash
python experiments/s03a_extract_svg2_graph.py \
  --config experiments/configs/svg2_pilot_c128_k512.yaml \
  --manifest data/smoke.jsonl \
  --allow-count-mismatch \
  --driver experiments.drivers.wan_t2v_smoke:run_record \
  --driver-args '{"checkpoint_dir":"Wan2.1-T2V-1.3B"}'
```
