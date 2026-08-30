import torch 
import torch.nn as nn
from myvllm.sampling import logits_to_token_ids


class SamplerLayer(nn.Module):
    """
    A custom sampler layer that selects elements from the input tensor
    based on provided indices.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        return logits_to_token_ids(logits, temperature)
