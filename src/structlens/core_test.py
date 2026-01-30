import numpy as np
import torch

from structlens.core import StructLens, mst
from structlens.utils.logging_config import get_logger, setup_logging

logger = get_logger("struct_lens.core_test")

setup_logging(log_level="INFO")


def test_mst_basic():
    scores = np.array(
        [
            [
                [0, 1, 1, 1],
                [0, 0, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 0, 4],
            ],
            [
                [4, 0, 0, 9],
                [3, 0, 0, 9],
                [2, 2, 0, 9],
                [9, 9, 9, 9],
            ],
        ]
    )
    expected = [[3, 0, 1, 3], [0, 2, 0, -1]]

    spanning_tree_list = mst(scores, num_nodes=[4, 3])
    argmax_heads = [
        spanning_tree["argmax_heads"].tolist() for spanning_tree in spanning_tree_list
    ]

    assert argmax_heads == expected


def test_struct_lens_compute_scores():
    from structlens.similarity import L2DistanceSimilarityFunction

    # 1 batch, 3 nodes, 2 features
    representations = torch.Tensor(
        [
            [
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 4.0],
            ]
        ]
    )
    mask = torch.Tensor(
        [
            [
                [float("-inf"), 1.0, 1.0],
                [float("-inf"), float("-inf"), 1.0],
                [float("-inf"), float("-inf"), float("-inf")],
            ]
        ]
    )
    root_selection_scores = torch.Tensor([[1, 1, 1]])

    # Distance:
    #  [
    #         [0, 3, 5],
    #         [3, 0, 4],
    #         [5, 4, 0],
    #  ]

    # Exponentiated and masked with root:
    expected = torch.Tensor(
        [
            [1.0, 0.25, 0.1666666667],
            [float("-inf"), 1.0, 0.2],
            [float("-inf"), float("-inf"), 1.0],
        ]
    )
    struct_lens = StructLens()
    scores = struct_lens.compute_scores(
        representations, mask, root_selection_scores, L2DistanceSimilarityFunction()
    )
    logger.debug(f"scores: {scores}")
    logger.debug(f"expected: {expected}")
    assert torch.allclose(scores, expected, atol=1e-4)


def test_struct_lens_mst():
    scores = torch.Tensor(
        [
            [
                [0, 1, 1, 1],
                [0, 0, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 0, 4],
            ],
            [
                [4, 0, 0, 9],
                [3, 0, 0, 9],
                [2, 2, 0, 9],
                [9, 9, 9, 9],
            ],
        ]
    )
    expected = [[3, 0, 1, 3], [0, 2, 0, -1]]

    struct_lens = StructLens()
    st_list = struct_lens.compute_mst(scores, num_nodes=[4, 3])
    argmax_heads = [spanning_tree["argmax_heads"].tolist() for spanning_tree in st_list]
    logger.debug(f"argmax_heads: {argmax_heads}")
    logger.debug(f"expected: {expected}")

    assert argmax_heads == expected
