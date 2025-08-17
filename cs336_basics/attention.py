import torch
from einops import einsum

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim, keepdim=True).values
    x_stable = x - max_x

    numerator = torch.exp(x_stable)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator

def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.shape[-1]

    attention_scores = (
        einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        / d_k**0.5
    )
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -float("inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    return einsum(
        attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v"
    )