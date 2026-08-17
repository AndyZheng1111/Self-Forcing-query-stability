# 基于 Self-Forcing 的 Future Forcing pre-RoPE Query 统计规律简单复现

## A minimal reproduction of pre-/post-RoPE Query stability on Self-Forcing

This fork provides a small-scale reproduction of the Query-distribution
observation used to motivate *Future Forcing*. It runs the Self-Forcing
Wan2.1-T2V-1.3B model, captures Query dimension 0 before and after RoPE, and
compares the empirical distributions across generated latent frames.

## Scope

Implemented here:

- capture of one flattened Query dimension in every self-attention DiT block;
- capture during the clean-context forward that writes denoised chunks into the
  Self-Forcing KV cache;
- preservation of latent-frame and layer boundaries;
- frame-wise normalized Wasserstein-1 analysis and density plots;
- 21-latent-frame and 81-latent-frame T2V experiments.

Not implemented here:

- Future Forcing future-Query proxies;
- future-aware KV importance scoring or eviction;
- future-profile-aware KV merging;
- dynamic temporal RoPE correction.

Consequently, this repository is an observation-only reproduction rather than a
complete implementation of Future Forcing.

## Experimental setup

| Item | Value |
|---|---|
| Backbone | Self-Forcing DMD / Wan2.1-T2V-1.3B |
| Checkpoint | `self_forcing_dmd.pt`, EMA weights |
| Generation mode | Text-to-video |
| Seed | 0 |
| Query dimension | flattened dimension 0 (head 0, feature 0) |
| DiT blocks | all 30 blocks, analyzed independently |
| Tokens per latent frame | 1560 |
| Capture stage | clean-context KV-cache update forward |
| Current capture constraints | single process, batch size 1, T2V only |

The 81-latent-frame run uses Self-Forcing's rolling KV cache with
`local_attn_size: 21`. This remains FIFO/sliding-window eviction; it is not a
Future Forcing cache policy.

## Installation and checkpoints

Follow the upstream installation and checkpoint instructions in the main
[README](README.md). The expected relative paths are:

```text
wan_models/Wan2.1-T2V-1.3B/
checkpoints/self_forcing_dmd.pt
```

Run the lightweight tests before inference:

```bash
python -m unittest tests.test_query_capture
```

## Capture a 21-latent-frame run

From the repository root:

```bash
python inference.py \
  --config_path configs/self_forcing_dmd.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/future_forcing/prompt_0.txt \
  --output_folder output/query_video/prompt_0 \
  --num_output_frames 21 \
  --use_ema \
  --seed 0 \
  --num_samples 1 \
  --save_with_index \
  --query_capture_dir output/query_capture/prompt_0 \
  --query_dimension 0
```

Repeat with `prompt_1.txt`, `prompt_2.txt`, or `prompt_dynamic_0.txt`, changing
the corresponding output directories.

## Capture the 81-latent-frame rolling-cache run

```bash
python inference.py \
  --config_path configs/self_forcing_dmd_long.yaml \
  --checkpoint_path checkpoints/self_forcing_dmd.pt \
  --data_path prompts/future_forcing/prompt_dynamic_0.txt \
  --output_folder output/query_video/prompt_dynamic_81latent \
  --num_output_frames 81 \
  --use_ema \
  --seed 0 \
  --num_samples 1 \
  --save_with_index \
  --query_capture_dir output/query_capture/prompt_dynamic_81latent \
  --query_dimension 0
```

Here 21 latent frames decode to 81 video frames, while 81 latent frames decode
to 321 video frames in this pipeline.

## Analyze a capture

```bash
python scripts/analyze_query_stability.py \
  output/query_capture/prompt_0/*.pt \
  --output_dir output/query_analysis/prompt_0 \
  --plot_layer 15
```

Use `--no_plots` when Matplotlib is unavailable.

For a fixed layer and latent frame, the capture contains 1560 scalar values from
Query dimension 0. For every pair of frames \(i,j\), the analysis computes

\[
\widehat W_1(X_i,X_j)=
\frac{W_1(X_i,X_j)}
{\sqrt{(\operatorname{Var}(X_i)+\operatorname{Var}(X_j))/2}+10^{-8}},
\]

then averages over all frame pairs. It does not average Query tensors across
layers and does not track corresponding spatial tokens across time.

## Results

| Run | Latent frames | Layers with pre < post | Mean pre W1 | Mean post W1 | Post/pre |
|---|---:|---:|---:|---:|---:|
| Art gallery | 21 | 30/30 | 0.780 | 3.492 | 4.48 |
| Ocean outpost | 21 | 30/30 | 0.561 | 3.083 | 5.49 |
| Skateboard downhill | 21 | 30/30 | 0.591 | 3.599 | 6.09 |
| Rooftop parkour | 21 | 30/30 | 0.767 | 3.488 | 4.55 |
| Rooftop parkour, rolling cache | 81 | 30/30 | 0.594 | 3.697 | 6.23 |

Across all five runs and all 30 blocks, pre-RoPE normalized W1 was lower than
post-RoPE normalized W1 in 150/150 run-layer comparisons. This reproduces the
paper's qualitative observation, not its exact numerical table.

Derived CSV, JSON, and PNG outputs are retained under
`output/query_analysis/`. Raw `.pt` captures, videos, checkpoints, and Wan model
weights are intentionally excluded from Git.

## Limitations

- one checkpoint, one seed, and one Query dimension;
- T2V only; no I2V validation;
- the Future Forcing paper does not disclose which DiT block was used for its
  plotted distribution, so all blocks are reported independently here;
- frame pairs share frames and tokens and are not independent statistical
  samples;
- results should not be generalized to all prompts, heads, dimensions, models,
  or generation settings.

## References and attribution

- [Self-Forcing](https://github.com/guandeh17/Self-Forcing), the upstream code
  and model backbone.
- [Future Forcing](https://arxiv.org/abs/2605.30083), the source of the
  pre-/post-RoPE Query stability hypothesis examined here.

The original Self-Forcing license is preserved in [LICENSE](LICENSE).
