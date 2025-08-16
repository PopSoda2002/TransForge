import torch

from einops import rearrange, einsum

def example_1():
    """
    Matrix multiplication
    """
    # batch, sequence, embedding_in
    x = torch.randn(2, 3, 6)
    # embedding_out, embedding_in
    y = torch.randn(5, 6)
    z = einsum(x, y, "bs sequence_length embedding_in, embedding_out embedding_in -> bs sequence_length embedding_out")
    print(z.shape)
    print(z)
    z2 = einsum(x, y, "... embedding_in, embedding_out embedding_in -> ... embedding_out")
    print(z2.shape)
    print(z2)

def example_2():
    """
    Broadcasted matrix multiplication
    """
    # bs, height, width, channels
    x = torch.randn(64, 100, 10, 3)
    # scale
    y = torch.linspace(0.0, 1.0, 5)
    z = einsum(x, y, "bs h w c, s -> bs s h w c")
    print(z.shape)

def example_3():
    """
    pixel-mixing
    """
    # bs, height, width, channels
    x = torch.randn(64, 100, 10, 3)
    B = torch.randn(100*10, 100*10)
    x_flat = rearrange(x, "bs h w c -> bs c (h w)")
    z = einsum(x_flat, B, "bs c pixel_in, pixel_in pixel_out -> bs c pixel_out")
    z = rearrange(z, "bs c (h w) -> bs h w c", h=100, w=10)
    print(z.shape)

if __name__ == "__main__":
    example_3()