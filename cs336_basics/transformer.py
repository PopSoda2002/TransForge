import torch
import torch.nn as nn
from cs336_basics.attention import MultiheadAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rmsnorm_module import RMSNorm
from cs336_basics.rope import RoPE
from cs336_basics.embedding_module import EmbeddingModule
from cs336_basics.linear_module import LinearModule

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

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff):
        super().__init__()
        self.embedding = EmbeddingModule(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.lm_head = LinearModule(d_model, vocab_size)

    def forward(self, x: torch.Tensor, rope: RoPE | None = None) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm(x)
        return self.lm_head(x)