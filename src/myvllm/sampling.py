import torch


def greedy_token_ids(logits: torch.Tensor) -> torch.Tensor:
    """Deterministic greedy selection with an explicit exact-tie policy.

    CUDA/PyTorch ``argmax`` returns the first maximum, which makes exact BF16
    ties depend on an implicit library convention.  Mini-vLLM uses the highest
    token id among exactly equal maxima so target-only decode and speculative
    verification share one documented rule.
    """
    reversed_indices = logits.flip(dims=(-1,)).argmax(dim=-1)
    return logits.shape[-1] - 1 - reversed_indices


def logits_to_token_ids(
    logits: torch.Tensor, temperature: torch.Tensor
) -> torch.Tensor:
    """Shared target-only/speculative logits decision function.

    Temperature zero is exact deterministic argmax. Positive temperatures retain
    the existing exponential-race categorical sampler.
    """
    if temperature.ndim == 0:
        temperature = temperature.expand(logits.shape[0])
    if torch.all(temperature == 0):
        return greedy_token_ids(logits)
    greedy = temperature == 0
    safe_temperature = torch.where(greedy, torch.ones_like(temperature), temperature)
    scaled_logits = logits / safe_temperature.unsqueeze(-1)
    probs = torch.softmax(scaled_logits, dim=-1)
    sampled = probs.div(
        torch.empty_like(probs).exponential_().clamp_min_(1e-10)
    ).argmax(dim=-1)
    return torch.where(greedy, greedy_token_ids(logits), sampled)
