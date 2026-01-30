from enum import Enum
from typing import Protocol

import torch
from torch import Tensor


class SimilarityType(Enum):
    """
    The type of similarity function to use for the MST.
    """

    L2_DISTANCE = "l2_distance"


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


class L2DistanceSimilarityFunction:
    def __call__(self, representations: Tensor) -> Tensor:
        """
        Calculate the negative L2 distance between the representations.

        Args:
            representations: A tensor of shape (batch_size, num_tokens, num_features)

        Returns:
            The negative L2 distance between the representations.
                A tensor of shape (batch_size, num_tokens, num_tokens)
        """
        representations = representations.float()
        distance = -torch.cdist(representations, representations, p=2)
        return distance


def get_similarity_function(similarity_type: SimilarityType) -> SimilarityFunction:
    """
    Get the similarity function based on the similarity type.
    """
    match similarity_type:
        case SimilarityType.L2_DISTANCE:
            return L2DistanceSimilarityFunction()
