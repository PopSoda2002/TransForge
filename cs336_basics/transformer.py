import torch
import torch.nn as nn
from cs336_basics.attention import MultiheadAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm_module import RMSNorm
from cs336_basics.rope import RoPE

class Block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiheadAttention(d_model, num_heads)
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), rope)
        x = x + self.feed_forward(self.norm2(x))
        return x