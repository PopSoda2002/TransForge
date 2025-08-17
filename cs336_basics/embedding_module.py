import torch
import torch.nn as nn
from einops import rearrange, einsum

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        std = 1
        initial_weights = torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype)
        nn.init.trunc_normal_(initial_weights, mean=0.0, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(initial_weights)

    def forward(self, x: torch.LongTensor):
        return self.weight[x]