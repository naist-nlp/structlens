from logging import Logger
from typing import TypedDict

import einops
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import torch
from tensorflow_text.python.ops import mst_ops
from torch import Tensor

from structlens.similarity import SimilarityFunction
from structlens.utils.logging_config import get_logger

default_logger = get_logger("struct_lens")


def set_seed(seed: int):
    """
    Set the seed for the random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class SpanningTree(TypedDict):
    """
    A spanning tree of a directed graph.

    max_score: The maximum score of the spanning tree.
    argmax_heads: The head of the edge in the maximum spanning tree.
        where argmax_heads[i] is the head of i-th node in the spanning tree.
        If argmax_heads[i] == i, then the i-th node is the root node.
        If argmax_heads[i] == -1, then the i-th node is padded.
    """

    max_score: npt.NDArray[np.float32]
    argmax_heads: npt.NDArray[np.int32]


def mst_tf(
    scores: npt.NDArray,
    num_nodes: list[int] | None = None,
    is_forest=False,
    seed: int = 42,
    logger: Logger = default_logger,
) -> list[SpanningTree]:
    """
    Computes the maximum spanning tree of a batch of graphs.

    Args:
        scores: A numpy array of shape (batch_size, num_nodes, num_nodes)
            where scores[i, j, k] is the score of the edge from node j to node k in batch i if i != j
            and the score of the root selection in batch i if i == j.
        num_nodes: The number of nodes in each batch.
        is_forest: Whether to compute the maximum spanning forest.
        seed: The seed for the random number generator.

    Returns:
        spanning_tree_list: SpanningTree objects for each batch.
    """
    set_seed(seed)
    assert scores.ndim == 3, (
        "scores must be a 3D numpy array (batch_size, num_nodes, num_nodes)"
    )
    assert scores.shape[1] == scores.shape[2], "scores must be a square matrix"

    batch_size = scores.shape[0]
    if num_nodes is None:
        num_nodes = [scores.shape[1]] * batch_size

    # NOTE: TensorFlow MST expects the scores to be in the format (batch_size, target, source)
    # Ref: <https://github.com/tensorflow/text/blob/aa839b19768fb359a64d26f36d740eb65103c1bf/tensorflow_text/core/ops/mst_ops.cc>
    score_t = einops.rearrange(scores, "b n1 n2 -> b n2 n1")
    tf_scores = tf.convert_to_tensor(score_t, dtype=tf.float32)
    tf_num_nodes = tf.convert_to_tensor(num_nodes, dtype=tf.int32)
    max_scores, argmax_heads = mst_ops.max_spanning_tree(
        tf_num_nodes, tf_scores, forest=is_forest
    )

    max_scores_np: npt.NDArray[np.float32] = max_scores.numpy()
    argmax_heads_np: npt.NDArray[np.int32] = argmax_heads.numpy()

    spanning_tree_list: list[SpanningTree] = []
    for i in range(batch_size):
        argmax_heads_i = argmax_heads_np[i]
        logger.debug(f"argmax_heads (tf): {argmax_heads_i}")
        spanning_tree_list.append(
            SpanningTree(
                max_score=max_scores_np[i],
                argmax_heads=argmax_heads_i,
            )
        )
    return spanning_tree_list


def mst(
    scores: npt.NDArray,
    num_nodes: list[int] | None = None,
    is_forest=False,
    seed: int = 42,
    logger: Logger = default_logger,
) -> list[SpanningTree]:
    return mst_tf(scores, num_nodes, is_forest, seed, logger)


class StructLens:
    def __init__(self, seed: int = 42, logger: Logger = default_logger):
        self.seed = seed
        self.logger = logger
        set_seed(seed)

    @torch.no_grad()
    def compute_scores(
        self,
        representations: Tensor,
        mask: Tensor,
        root_selection_scores: Tensor,
        similarity_function: SimilarityFunction,
    ) -> Tensor:
        """
        Compute the scores between the representations then mask the scores and apply the root selection scores.

        Args:
            representations: A tensor of shape (batch_size, num_nodes, num_features)
            mask: A tensor of shape (batch_size, num_nodes, num_nodes)
            root_selection_scores: A tensor of shape (batch_size, num_nodes)
            similarity_function: A function that takes a tensor of shape (batch_size, num_nodes, num_features) and returns a tensor of shape (batch_size, num_nodes, num_nodes)
        Returns:
            scores: A tensor of shape (batch_size, num_nodes, num_nodes)
        """
        assert (
            representations.shape[0] == mask.shape[0] == root_selection_scores.shape[0]
        ), (
            f"representations, mask, and root_selection_scores must have the same batch size, but got {representations.shape[0]}, {mask.shape[0]}, {root_selection_scores.shape[0]}"
        )
        assert (
            representations.shape[1] == mask.shape[1] == root_selection_scores.shape[1]
        ), (
            f"representations, mask, and root_selection_scores must have the same number of nodes, but got {representations.shape[1]}, {mask.shape[1]}, {root_selection_scores.shape[1]}"
        )
        assert mask.shape[1] == mask.shape[2], (
            f"mask must be a square matrix, but got {mask.shape[1]}x{mask.shape[2]}"
        )
        self.logger.debug(f"representations.shape: {representations.shape}")
        self.logger.debug(f"mask.shape: {mask.shape}")
        self.logger.debug(f"root_selection_scores.shape: {root_selection_scores.shape}")

        scores = similarity_function(
            representations
        )  # (batch_size, num_nodes (heads), num_nodes (modifiers))
        self.logger.debug(f"scores.shape: {scores.shape}")
        self.logger.debug(f"scores: {scores}")

        # NOTE: Reciprocal of the similarity to avoid negative values
        scores = 1.0 / (1.0 - scores)
        self.logger.debug(f"scores (reciprocal): {scores}")
        scores.mul_(mask)

        # Replace the diagonal elements of scores with root_selection_scores
        eye_mask = torch.eye(
            scores.shape[1], dtype=torch.bool, device=scores.device
        ).unsqueeze(0)
        scores.masked_fill_(eye_mask, 0)
        del eye_mask
        root_scores_device = root_selection_scores.to(scores.device)
        scores.add_(torch.diag_embed(root_scores_device))
        del root_scores_device

        return scores

    @torch.no_grad()
    def compute_mst(
        self, scores: Tensor, num_nodes: list[int], is_forest: bool = False
    ) -> list[SpanningTree]:
        """
        Compute the maximum spanning tree of a batch of graphs.

        Args:
            scores: A tensor of shape (batch_size, num_nodes (heads), num_nodes (modifiers))
            num_nodes: The number of actual nodes in each batch.

        Returns:
            spanning_tree_list: SpanningTree objects for each batch.
        """
        assert scores.ndim == 3, (
            f"scores must be a 3D tensor (batch_size, num_nodes, num_nodes), but got {scores.ndim}"
        )
        assert scores.shape[1] == scores.shape[2], (
            f"scores must be a square matrix, but got {scores.shape[1]}x{scores.shape[2]}"
        )
        assert len(num_nodes) == scores.shape[0], (
            f"num_nodes must be a list of length batch_size, but got {len(num_nodes)}"
        )
        self.logger.debug(f"scores.shape: {scores.shape}")
        self.logger.debug(f"num_nodes: {num_nodes}")

        scores_np = scores.float().cpu().numpy()
        del scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mst(
            scores_np,
            num_nodes,
            is_forest=is_forest,
            seed=self.seed,
            logger=self.logger,
        )
