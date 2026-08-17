import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze frame-wise pre/post-RoPE query distribution stability."
    )
    parser.add_argument("captures", nargs="+", help="Query-capture .pt files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--plot_layer", type=int, default=15)
    parser.add_argument("--no_plots", action="store_true")
    return parser.parse_args()


def load_capture(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    for key in ("pre_query", "post_query", "layer_indices"):
        if key not in payload:
            raise ValueError(f"{path} is missing {key}")
    pre = payload["pre_query"].float()
    post = payload["post_query"].float()
    if pre.shape != post.shape or pre.ndim != 3:
        raise ValueError(
            f"Expected matching [layer, frame, token] tensors, got {pre.shape} and {post.shape}"
        )
    return payload, pre, post


def mean_pairwise_normalized_w1(values):
    """Return the mean and individual normalized W1 values for [frame, token]."""
    sorted_values = values.sort(dim=-1).values
    frame_variances = values.var(dim=-1, unbiased=False)
    pair_values = []
    for frame_i, frame_j in combinations(range(values.shape[0]), 2):
        w1 = (sorted_values[frame_i] - sorted_values[frame_j]).abs().mean()
        pooled_std = torch.sqrt(
            (frame_variances[frame_i] + frame_variances[frame_j]) / 2
        )
        pair_values.append(w1 / (pooled_std + 1e-8))
    pair_values = torch.stack(pair_values)
    return pair_values.mean().item(), pair_values


def analyze_capture(path):
    payload, pre, post = load_capture(path)
    layers = payload["layer_indices"].tolist()
    rows = []
    pairwise = {}
    for tensor_index, layer_index in enumerate(layers):
        pre_mean, pre_pairs = mean_pairwise_normalized_w1(pre[tensor_index])
        post_mean, post_pairs = mean_pairwise_normalized_w1(post[tensor_index])
        rows.append({
            "capture": str(path),
            "prompt_index": payload.get("metadata", {}).get("prompt_index"),
            "seed": payload.get("metadata", {}).get("seed"),
            "query_dimension": payload.get("query_dimension"),
            "layer": layer_index,
            "pre_normalized_w1": pre_mean,
            "post_normalized_w1": post_mean,
            "post_pre_ratio": post_mean / max(pre_mean, 1e-8),
        })
        pairwise[int(layer_index)] = {
            "pre": pre_pairs,
            "post": post_pairs,
        }
    return payload, pre, post, rows, pairwise


def save_plots(output_dir, capture_stem, payload, pre, post, rows, plot_layer):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib and numpy; rerun with --no_plots "
            "or install matplotlib"
        ) from exc

    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    layers = [row["layer"] for row in rows]
    pre_w1 = [row["pre_normalized_w1"] for row in rows]
    post_w1 = [row["post_normalized_w1"] for row in rows]
    ratios = [row["post_pre_ratio"] for row in rows]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(layers, pre_w1, marker="o", label="Pre-RoPE")
    axis.plot(layers, post_w1, marker="o", label="Post-RoPE")
    axis.set_xlabel("DiT block")
    axis.set_ylabel("Mean pairwise normalized W1")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{capture_stem}_per_layer_w1.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(layers, ratios)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xlabel("DiT block")
    axis.set_ylabel("Post-RoPE / Pre-RoPE W1")
    fig.tight_layout()
    fig.savefig(plot_dir / f"{capture_stem}_per_layer_ratio.png", dpi=180)
    plt.close(fig)

    layer_indices = payload["layer_indices"].tolist()
    if plot_layer not in layer_indices:
        raise ValueError(f"plot_layer={plot_layer} is not in {layer_indices}")
    tensor_index = layer_indices.index(plot_layer)
    colors = plt.cm.viridis(np.linspace(0, 1, pre.shape[1]))
    for name, tensor in (("pre", pre), ("post", post)):
        fig, axis = plt.subplots(figsize=(7, 4.5))
        frame_values = tensor[tensor_index].numpy()
        value_min = frame_values.min()
        value_max = frame_values.max()
        bins = np.linspace(value_min, value_max, 101)
        for frame_index, values in enumerate(frame_values):
            density, edges = np.histogram(values, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            axis.plot(
                centers,
                density,
                color=colors[frame_index],
                alpha=0.75,
                linewidth=1,
            )
        axis.set_title(
            f"{name.capitalize()}-RoPE query dim {payload['query_dimension']}, layer {plot_layer}"
        )
        axis.set_xlabel("Query value")
        axis.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(
            plot_dir / f"{capture_stem}_layer_{plot_layer}_{name}_density.png",
            dpi=180,
        )
        plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []

    for capture_path_string in args.captures:
        capture_path = Path(capture_path_string)
        payload, pre, post, rows, _ = analyze_capture(capture_path)
        all_rows.extend(rows)
        summaries.append({
            "capture": str(capture_path),
            "prompt": payload.get("metadata", {}).get("prompt"),
            "seed": payload.get("metadata", {}).get("seed"),
            "query_dimension": payload.get("query_dimension"),
            "num_layers": pre.shape[0],
            "num_frames": pre.shape[1],
            "tokens_per_frame": pre.shape[2],
            "layers_pre_less_than_post": sum(
                row["pre_normalized_w1"] < row["post_normalized_w1"]
                for row in rows
            ),
            "mean_pre_normalized_w1": sum(
                row["pre_normalized_w1"] for row in rows
            ) / len(rows),
            "mean_post_normalized_w1": sum(
                row["post_normalized_w1"] for row in rows
            ) / len(rows),
        })
        if not args.no_plots:
            save_plots(
                output_dir,
                capture_path.stem,
                payload,
                pre,
                post,
                rows,
                args.plot_layer,
            )

    csv_path = output_dir / "per_layer_per_prompt.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, ensure_ascii=False, indent=2)

    rows_by_layer = {}
    for row in all_rows:
        rows_by_layer.setdefault(row["layer"], []).append(row)
    aggregate_rows = []
    for layer, rows in sorted(rows_by_layer.items()):
        mean_pre = sum(row["pre_normalized_w1"] for row in rows) / len(rows)
        mean_post = sum(row["post_normalized_w1"] for row in rows) / len(rows)
        aggregate_rows.append({
            "layer": layer,
            "num_prompts": len(rows),
            "mean_pre_normalized_w1": mean_pre,
            "mean_post_normalized_w1": mean_post,
            "post_pre_ratio": mean_post / max(mean_pre, 1e-8),
            "prompts_pre_less_than_post": sum(
                row["pre_normalized_w1"] < row["post_normalized_w1"]
                for row in rows
            ),
        })
    aggregate_path = output_dir / "summary_by_layer.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Saved metrics to {csv_path}")
    print(f"Saved cross-prompt layer summary to {aggregate_path}")


if __name__ == "__main__":
    main()
