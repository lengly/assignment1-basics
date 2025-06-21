import torch
import torch.nn as nn
import numpy as np
# from einops import rearrange, einsum

def InitParam(out_dim, in_dim, device=None, dtype=None):
    weight = nn.Parameter(torch.empty(out_dim, in_dim, device=device, dtype=dtype))
    weight_std = np.sqrt(2 / (in_dim + out_dim))
    nn.init.trunc_normal_(weight, std=weight_std, a=-3 * weight_std, b=3 * weight_std)
    return weight

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = InitParam(out_features, in_features, device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
        # return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = InitParam(num_embeddings, embedding_dim, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = x / ((x ** 2).mean(-1, keepdim=True) + self.eps) ** 0.5
        x = x * self.weight
        return x.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
        assert d_ff % 64 == 0, "d_ff must be divisible by 64"
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.silu = SiLU(device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))