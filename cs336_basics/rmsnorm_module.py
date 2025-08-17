import torch
import torch.nn as nn
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight
        result = x * norm
        return result.to(in_dtype)
