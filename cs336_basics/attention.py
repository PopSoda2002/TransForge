import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics.rope import RoPE
from cs336_basics.linear_module import LinearModule

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

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.wqkv = LinearModule(d_model, 3 * d_model)
        self.output_proj = LinearModule(d_model, d_model)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        transformed_q = rearrange(q, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=self.num_heads)
        token_positions = torch.arange(seq_len, device=x.device)
        if rope is not None:
            transformed_q = rope(transformed_q, token_positions)
        transformed_k = rearrange(k, "... sequence_length (num_heads d_k) -> ... num_heads sequence_length d_k", num_heads=self.num_heads)
        if rope is not None:
            transformed_k = rope(transformed_k, token_positions)
        transformed_v = rearrange(v, "... sequence_length (num_heads d_v) -> ... num_heads sequence_length d_v", num_heads=self.num_heads)
        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

        attention_output = scaled_dot_product_attention(transformed_q, transformed_k, transformed_v, mask)
        attention_output = rearrange(attention_output, "... num_heads sequence_length d_v -> ... sequence_length (num_heads d_v)")
        
        return self.output_proj(attention_output)
