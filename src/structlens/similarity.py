from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class SimilarityFunction(Protocol):
    def __call__(self, representations: Tensor) -> Tensor:
        """
        Calculate the similarity between the representations.

        Args:
            representations: A tensor of shape (batch_size, num_tokens, num_features)

        Returns:
            The similarity between the representations.
                A tensor of shape (batch_size, num_tokens, num_tokens)
        """
        ...


def l2_distance(representations: Tensor) -> Tensor:
    """
    Calculate the negative L2 distance as similarity between the representations.

    Args:
        representations: A tensor of shape (batch_size, num_tokens, num_features)

    Returns:
        The negative L2 distance between the representations.
            A tensor of shape (batch_size, num_tokens, num_tokens)
    """
    if representations.ndim != 3:
        raise ValueError(
            f"representations must be a 3D tensor, but got {representations.ndim}"
        )
    representations = representations.float()
    distance = -torch.cdist(representations, representations, p=2)
    return distance
