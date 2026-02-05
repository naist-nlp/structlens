"""
This module implements Tree Edit Distance (TED) calculation functionality.
It provides functions to convert between different tree representations and compute
the edit distance between two trees, following Zhang-Shasha's algorithm.
"""

import functools
from logging import Logger
from typing import Any

from structlens import SpanningTree
from structlens.utils.logging_config import get_logger

default_logger = get_logger("structlens.metrics")


def parent_array_to_tree(
    parents: list[int],
) -> tuple[int | None, list[list[int]]]:
    """Convert a parent array representation to a tree structure.

    Args:
        parents: Array where parents[i] represents the parent of node i.

    Returns:
        A pair (root, children) where root is the root node index and
        children is a list of lists where children[i] contains the children of node i.

    Raises:
        ValueError: If the input represents a forest instead of a single tree.
    """
    n = len(parents)
    children: list[list[int]] = [[] for _ in range(n)]
    root = None
    for i, p in enumerate(parents):
        if i == p:  # root
            if root is not None:
                raise ValueError("Forest detected: multiple roots found")
            root = i
        else:
            children[p].append(i)
    return root, children


def count_subtree_sizes(
    n: int, root: int | None, children: list[list[int]]
) -> list[int]:
    """Calculate the size of each subtree in the tree using iterative post-order traversal.

    Args:
        n: Total number of nodes in the tree.
        root: Index of the root node.
        children: List of lists where children[i] contains the children of node i.

    Returns:
        Array where subtree_sizes[i] is the size of the subtree rooted at node i.
    """
    if root is None:
        return [0] * n

    subtree_sizes = [0] * n
    # Iterative post-order traversal to avoid recursion limit issues
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            subtree_sizes[node] = 1 + sum(subtree_sizes[ch] for ch in children[node])
        else:
            stack.append((node, True))
            for child in reversed(children[node]):
                stack.append((child, False))
    return subtree_sizes


def convert_tree_format(
    parents: list[int],
) -> tuple[int | None, list[list[int]], list[int]]:
    """Convert a parent array to an expanded tree representation.

    Args:
        parents: Array where parents[i] represents the parent of node i.

    Returns:
        A triple (root, children, subtree_sizes) containing the root node index,
        children lists, and subtree sizes for each node.
    """
    root, children = parent_array_to_tree(parents)
    subtree_sizes = count_subtree_sizes(len(parents), root, children)
    return root, children, subtree_sizes


def ted(
    parents1: list[int],
    parents2: list[int],
    labels1: list[Any] | None = None,
    labels2: list[Any] | None = None,
    logger: Logger = default_logger,
) -> int:
    """Calculate the Tree Edit Distance between two trees.

    Args:
        parents1: Parent array representation of the first tree.
        parents2: Parent array representation of the second tree.
        labels1: Optional labels for tree1 nodes.
            Defaults to indices [0..len(parents1)-1].
        labels2: Optional labels for tree2 nodes.
            Defaults to indices [0..len(parents2)-1].
        logger: Logger instance.

    Returns:
        The minimum edit distance between the two trees.
    """
    root1, children1, subtree_size1 = convert_tree_format(parents1)
    root2, children2, subtree_size2 = convert_tree_format(parents2)

    if root1 is None or root2 is None:
        raise ValueError("Both trees must have a root node")

    labels1 = labels1 or list(range(len(parents1)))
    labels2 = labels2 or list(range(len(parents2)))

    @functools.cache
    def distance(u: int, v: int) -> int:
        cost = 0 if labels1[u] == labels2[v] else 1

        # children
        c1 = children1[u]
        c2 = children2[v]

        # compute edit distance of (ordered) sequences of subtrees
        m, n = len(c1), len(c2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + subtree_size1[c1[i - 1]]  # remove only
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + subtree_size2[c2[j - 1]]  # insert only
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = min(
                    dp[i - 1][j] + subtree_size1[c1[i - 1]],  # remove
                    dp[i][j - 1] + subtree_size2[c2[j - 1]],  # insert
                    dp[i - 1][j - 1] + distance(c1[i - 1], c2[j - 1]),  # match
                )

        logger.debug(f"Distance {u}->{v}: {cost + dp[m][n]}")
        return cost + dp[m][n]

    return distance(root1, root2)


def edge_edit(parents1: list[int], parents2: list[int]) -> int:
    """Calculate the Edge-wise Edit Distance between two trees of the exact same set of vertices.

    Args:
        parents1: Parent array representation of the first tree.
        parents2: Parent array representation of the second tree.

    Returns:
        The number of differing edges between the two trees.
    """
    return sum(p1 != p2 for p1, p2 in zip(parents1, parents2, strict=True))


class StructLensDistance:
    def __init__(self, logger: Logger = default_logger):
        self.logger = logger

    def ted(
        self,
        spanning_tree1: SpanningTree,
        spanning_tree2: SpanningTree,
        labels1: list[Any] | None = None,
        labels2: list[Any] | None = None,
    ) -> int:
        """
        Compute the Tree Edit Distance between two spanning trees.

        Args:
            spanning_tree1: SpanningTree object for the first tree.
            spanning_tree2: SpanningTree object for the second tree.
            labels1: Labels for the nodes in the first tree. Defaults to None.
            labels2: Labels for the nodes in the second tree. Defaults to None.

        Returns:
            The Tree Edit Distance between the two spanning trees.
        """
        return ted(
            spanning_tree1.argmax_heads.tolist(),
            spanning_tree2.argmax_heads.tolist(),
            labels1,
            labels2,
            self.logger,
        )

    def edge_edit(
        self, spanning_tree1: SpanningTree, spanning_tree2: SpanningTree
    ) -> int:
        """
        Compute the Edge-wise Edit Distance between two spanning trees of the exact same set of vertices.

        Args:
            spanning_tree1: SpanningTree object for the first tree.
            spanning_tree2: SpanningTree object for the second tree.

        Returns:
            The Edge-wise Edit Distance between the two spanning trees.
        """
        if spanning_tree1.argmax_heads.shape != spanning_tree2.argmax_heads.shape:
            raise ValueError(
                f"spanning_tree1 and spanning_tree2 must have the same number of vertices, but got {spanning_tree1.argmax_heads.shape} and {spanning_tree2.argmax_heads.shape}"
            )
        return edge_edit(
            spanning_tree1.argmax_heads.tolist(), spanning_tree2.argmax_heads.tolist()
        )
