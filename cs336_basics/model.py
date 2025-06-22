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

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs = 1 / (theta ** (torch.arange(0, d_k, 2)[: (d_k // 2)].float() / d_k))
        t = torch.arange(max_seq_len, device=freqs.device)
        freqs_cis = torch.outer(t, freqs).to(device)
        self.register_buffer("cos_cache", torch.cos(freqs_cis), persistent=False, device=device)
        self.register_buffer("sin_cache", torch.sin(freqs_cis), persistent=False, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        seq_len = x.shape[-2]
        cos = self.cos_cache[token_positions].to(x.device)
        sin = self.sin_cache[token_positions].to(x.device)
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated = torch.stack([x_rotated_even, x_rotated_odd], dim=-1)
        x_rotated = x_rotated.view(x.shape)
        return x_rotated

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        max_x = torch.max(x, dim=dim, keepdim=True).values
        exp_x = torch.exp(x - max_x)
        return exp_x / exp_x.sum(dim=dim, keepdim=True)



