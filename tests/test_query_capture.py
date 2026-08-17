import sys
import unittest
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analyze_query_stability import mean_pairwise_normalized_w1
from utils.query_capture import QueryDimensionCapture


class QueryDimensionCaptureTest(unittest.TestCase):
    def test_collects_layers_chunks_and_frames(self):
        capture = QueryDimensionCapture(query_dimension=0, expected_layers=2)
        grid_sizes = torch.tensor([[3, 2, 2]])

        for start_frame in (0, 3):
            capture.begin_chunk(start_frame=start_frame, num_frames=3)
            for layer_index in range(2):
                base = torch.arange(72, dtype=torch.float32).reshape(1, 12, 2, 3)
                pre = base + start_frame * 100 + layer_index * 10
                post = pre + 1
                capture.record(
                    layer_index=layer_index,
                    pre_rope_query=pre,
                    post_rope_query=post,
                    grid_sizes=grid_sizes,
                    current_start=start_frame * 4,
                )
            capture.end_chunk()

        tensors = capture.tensors()
        self.assertEqual(tuple(tensors["pre_query"].shape), (2, 6, 4))
        self.assertEqual(tuple(tensors["post_query"].shape), (2, 6, 4))
        torch.testing.assert_close(
            tensors["post_query"], tensors["pre_query"] + 1
        )

    def test_pairwise_metric_uses_all_frame_pairs(self):
        values = torch.tensor([
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ])
        mean, pairs = mean_pairwise_normalized_w1(values)
        self.assertEqual(pairs.numel(), 3)
        self.assertGreater(mean, 0)


if __name__ == "__main__":
    unittest.main()
