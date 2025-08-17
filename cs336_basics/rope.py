import torch
import torch.nn as nn
from einops import rearrange, einsum

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self._init_rope(self.d_k)

    def _init_rope(self, d_k):
        position_ids = torch.arange(self.max_seq_len, device=self.device)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=self.device) / d_k))
        theta_i_k = torch.outer(position_ids, inv_freq)
        self.register_buffer("cos_cached", theta_i_k.cos(), persistent=False)
        self.register_buffer("sin_cached", theta_i_k.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        print(f"shape of x: {x.shape}")
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        print(f"shape of x after rearrange: {x.shape}")
        print(f"shape of x_even: {x_even.shape}")
        print(f"shape of x_odd: {x_odd.shape}")
        even_rot = x_even * cos - x_odd * sin
        odd_rot  = x_even * sin + x_odd * cos

        result = torch.empty_like(x)
        result[..., 0::2] = even_rot
        result[..., 1::2] = odd_rot
        return result