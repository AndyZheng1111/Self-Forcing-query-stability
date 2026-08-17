from pathlib import Path
from typing import Dict, Optional

import torch


class QueryDimensionCapture:
    """Collect one flattened query dimension during selected inference forwards."""

    def __init__(self, query_dimension: int = 0, expected_layers: Optional[int] = None):
        if query_dimension < 0:
            raise ValueError("query_dimension must be non-negative")
        self.query_dimension = query_dimension
        self.expected_layers = expected_layers
        self.reset()

    def reset(self) -> None:
        self.enabled = False
        self._active_chunk = None
        self._records: Dict[int, list] = {}

    def begin_chunk(self, start_frame: int, num_frames: int) -> None:
        if self.enabled:
            raise RuntimeError("A query-capture chunk is already active")
        if start_frame < 0 or num_frames <= 0:
            raise ValueError("Invalid query-capture frame range")
        self._active_chunk = {
            "start_frame": int(start_frame),
            "num_frames": int(num_frames),
            "layers": set(),
        }
        self.enabled = True

    def abort_chunk(self) -> None:
        self.enabled = False
        self._active_chunk = None

    def end_chunk(self) -> None:
        if not self.enabled or self._active_chunk is None:
            raise RuntimeError("No query-capture chunk is active")
        if self.expected_layers is not None:
            expected = set(range(self.expected_layers))
            missing = sorted(expected - self._active_chunk["layers"])
            if missing:
                raise RuntimeError(f"Missing query captures for layers: {missing}")
        self.enabled = False
        self._active_chunk = None

    def record(
        self,
        layer_index: int,
        pre_rope_query: torch.Tensor,
        post_rope_query: torch.Tensor,
        grid_sizes: torch.Tensor,
        current_start: int,
    ) -> None:
        if not self.enabled:
            return
        if self._active_chunk is None:
            raise RuntimeError("Query capture is enabled without an active chunk")
        if layer_index in self._active_chunk["layers"]:
            raise RuntimeError(
                f"Layer {layer_index} was captured more than once in one chunk"
            )
        if pre_rope_query.shape != post_rope_query.shape:
            raise ValueError("Pre- and post-RoPE query shapes differ")
        if pre_rope_query.ndim != 4:
            raise ValueError(
                "Expected query shape [batch, sequence, heads, head_dim], "
                f"got {tuple(pre_rope_query.shape)}"
            )
        if pre_rope_query.shape[0] != 1:
            raise ValueError("Query capture currently requires batch size 1")

        num_frames, height, width = [int(value) for value in grid_sizes[0].tolist()]
        frame_seq_length = height * width
        sequence_length = num_frames * frame_seq_length
        if pre_rope_query.shape[1] < sequence_length:
            raise ValueError("Query sequence is shorter than the spatiotemporal grid")
        if num_frames != self._active_chunk["num_frames"]:
            raise ValueError(
                f"Expected {self._active_chunk['num_frames']} frames, got {num_frames}"
            )

        actual_start_frame = int(current_start) // frame_seq_length
        if actual_start_frame != self._active_chunk["start_frame"]:
            raise ValueError(
                f"Expected start frame {self._active_chunk['start_frame']}, "
                f"got {actual_start_frame}"
            )

        num_heads = pre_rope_query.shape[2]
        head_dim = pre_rope_query.shape[3]
        flattened_dim = num_heads * head_dim
        if self.query_dimension >= flattened_dim:
            raise ValueError(
                f"query_dimension={self.query_dimension} exceeds flattened "
                f"query width {flattened_dim}"
            )
        head_index, feature_index = divmod(self.query_dimension, head_dim)

        def select_dimension(query: torch.Tensor) -> torch.Tensor:
            selected = query[
                :, :sequence_length, head_index, feature_index
            ].reshape(1, num_frames, frame_seq_length)
            return selected.detach().to(device="cpu", dtype=torch.float32).contiguous()

        record = {
            "start_frame": self._active_chunk["start_frame"],
            "pre": select_dimension(pre_rope_query),
            "post": select_dimension(post_rope_query),
        }
        self._records.setdefault(int(layer_index), []).append(record)
        self._active_chunk["layers"].add(int(layer_index))

    def tensors(self) -> Dict[str, torch.Tensor]:
        if self.enabled:
            raise RuntimeError("Cannot finalize query captures while a chunk is active")
        if not self._records:
            raise RuntimeError("No query captures were recorded")

        layer_indices = sorted(self._records)
        if self.expected_layers is not None:
            expected = list(range(self.expected_layers))
            if layer_indices != expected:
                raise RuntimeError(
                    f"Expected layers {expected}, captured {layer_indices}"
                )

        pre_layers = []
        post_layers = []
        expected_frame_starts = None
        for layer_index in layer_indices:
            records = sorted(
                self._records[layer_index], key=lambda item: item["start_frame"]
            )
            frame_starts = [item["start_frame"] for item in records]
            if expected_frame_starts is None:
                expected_frame_starts = frame_starts
            elif frame_starts != expected_frame_starts:
                raise RuntimeError(
                    f"Layer {layer_index} has inconsistent captured chunks"
                )

            expected_start = 0
            for item in records:
                if item["start_frame"] != expected_start:
                    raise RuntimeError(
                        f"Non-contiguous capture at frame {item['start_frame']}; "
                        f"expected {expected_start}"
                    )
                expected_start += item["pre"].shape[1]

            pre_layers.append(torch.cat([item["pre"] for item in records], dim=1))
            post_layers.append(torch.cat([item["post"] for item in records], dim=1))

        # The batch-size-one restriction lets downstream analysis use [layer, frame, token].
        pre = torch.stack(pre_layers, dim=1).squeeze(0)
        post = torch.stack(post_layers, dim=1).squeeze(0)
        return {
            "pre_query": pre,
            "post_query": post,
            "layer_indices": torch.tensor(layer_indices, dtype=torch.int64),
        }

    def save(self, output_path, metadata: Optional[dict] = None) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = self.tensors()
        payload = {
            "format_version": 1,
            "query_dimension": self.query_dimension,
            "capture_stage": "clean_context",
            "pre_query": tensors["pre_query"],
            "post_query": tensors["post_query"],
            "layer_indices": tensors["layer_indices"],
            "metadata": metadata or {},
        }
        torch.save(payload, output_path)
        return output_path
