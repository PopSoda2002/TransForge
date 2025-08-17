import torch
import torch.nn as nn
from einops import einsum

def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.randn(d_ff, d_model))
        self.W3 = nn.Parameter(torch.randn(d_ff, d_model))
        self.W2 = nn.Parameter(torch.randn(d_model, d_ff))

    def forward(self, x: torch.Tensor):
        temp = SiLU(einsum(x, self.W1, "... d_model, d_ff d_model -> ... d_ff")) * einsum(x, self.W3, "... d_model, d_ff d_model -> ... d_ff")
        result = einsum(temp, self.W2, "... d_ff, d_model d_ff -> ... d_model")
        return result