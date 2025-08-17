import torch
import torch.nn as nn
from einops import rearrange, einsum

class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        # Per the formula in the image, we initialize weights from a truncated normal distribution.
        # The distribution is N(μ=0, σ^2 = 2 / (d_in + d_out)), truncated at [-3σ, 3σ].
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        # We create an empty tensor and then fill it with values from the truncated normal distribution.
        initial_weights = torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype)
        nn.init.trunc_normal_(initial_weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(initial_weights)

    def forward(self, x):
        return einsum(x, self.weight, "... dim_in, dim_out dim_in -> ... dim_out")

if __name__ == "__main__":
    linear_module = LinearModule(3, 4)
    print(linear_module(torch.randn(2, 3)))
