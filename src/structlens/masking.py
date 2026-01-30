import torch
from torch import Tensor


def create_masks(num_nodes_per_layer: list[int], device: str | torch.device) -> Tensor:
    """
    Create a mask for the scores for each layer.

    Args:
        num_nodes_per_layer: A list of the number of nodes in each layer.
        device: The device to create the mask on.

    Returns:
        mask: A tensor of shape (num_layers, max_nodes_for_masks, max_nodes_for_masks)
    """
    max_nodes_for_masks = max(num_nodes_per_layer)
    tril_template = torch.tril(
        torch.ones(
            max_nodes_for_masks,
            max_nodes_for_masks,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=-1,
    )
    mask_layer = torch.ones(
        (max_nodes_for_masks, max_nodes_for_masks), dtype=torch.float32, device=device
    )
    mask_layer.masked_fill_(tril_template, float("-inf"))
    mask = torch.stack([mask_layer] * len(num_nodes_per_layer))
    return mask
