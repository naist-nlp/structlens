import torch
from torch import Tensor


def generate_mask(
    batch_size: int,
    num_layers: int,
    num_tokens_per_layer: int,
    mask_value: float = float(0),
    mask_same_layer: bool = True,
    mask_future_tokens: bool = True,
    mask_the_same_position: bool = False,
) -> Tensor:
    """
    Generates a mask for the scores.

    Args:
        batch_size: The number of batches
        num_layers: The number of layers
        num_tokens_per_layer: The number of tokens per layer
        mask_value: The value to mask the scores with
        mask_same_layer: Whether to mask connections in the same layer
        mask_future_tokens: Whether to mask connections from future tokens to past tokens across layers
        mask_the_same_position: Whether to mask connections from the same position to the same position

    Returns:
        mask: A tensor of shape (batch_size, num_layers * num_tokens_per_layer, num_layers * num_tokens_per_layer)
    """
    num_nodes = num_layers * num_tokens_per_layer
    mask = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.float32)

    # Precompute per-node layer and token indices for vectorized comparisons
    node_indices = torch.arange(num_nodes)
    layer_indices = node_indices // num_tokens_per_layer  # (num_nodes,)
    token_indices = node_indices % num_tokens_per_layer  # (num_nodes,)

    # Mask connections from later layers to earlier layers (row layer > col layer)
    future_layer_mask = layer_indices.unsqueeze(1) > layer_indices.unsqueeze(0)
    mask[:, future_layer_mask] = mask_value

    if mask_same_layer:
        # Mask connections within the same layer
        same_layer_mask = layer_indices.unsqueeze(1) == layer_indices.unsqueeze(0)
        mask[:, same_layer_mask] = mask_value

    if mask_future_tokens:
        # Mask connections where the source token position is later than the target
        future_token_mask = token_indices.unsqueeze(1) > token_indices.unsqueeze(0)
        mask[:, future_token_mask] = mask_value

    if mask_the_same_position:
        # Mask connections between nodes at the same token position
        same_position_mask = token_indices.unsqueeze(1) == token_indices.unsqueeze(0)
        mask[:, same_position_mask] = mask_value

    return mask
