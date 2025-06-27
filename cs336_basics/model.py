import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, einsum
from typing import Optional

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
        freqs = 1 / (theta ** (torch.arange(0, d_k, 2, device=device)[: (d_k // 2)].float() / d_k))
        t = torch.arange(max_seq_len, device=device)
        freqs_cis = torch.outer(t, freqs)
        self.register_buffer("cos_cache", torch.cos(freqs_cis), persistent=False)
        self.register_buffer("sin_cache", torch.sin(freqs_cis), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        d_k = x.shape[-1]
        cos = self.cos_cache[token_positions, :d_k // 2].to(x.device)
        sin = self.sin_cache[token_positions, :d_k // 2].to(x.device)
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

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q shape: (batch_size, ..., num_q, d_k)
        # k shape: (batch_size, ..., num_kv, d_k)
        # v shape: (batch_size, ..., num_kv, d_v)
        # mask shape: (num_q, num_kv)
        d_k = q.shape[-1]
        qk = einsum(q, k, "... num_q d_k, ... num_kv d_k -> ... num_q num_kv") / d_k ** 0.5
        if mask is not None:
            qk = qk.masked_fill(mask == 0, float("-inf"))
        return self.softmax(qk, dim=-1) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope: Optional[RoPE] = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        self.rope = rope
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.scale_dot_product_attention = ScaleDotProductAttention()
    
    def forward(self, x: torch.Tensor, casual_mask: torch.Tensor = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        # x.shape: (batch_size, sequence_length, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "bz s (h d) -> bz h s d", h=self.num_heads)
        k = rearrange(k, "bz s (h d) -> bz h s d", h=self.num_heads)
        v = rearrange(v, "bz s (h d) -> bz h s d", h=self.num_heads)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        x = self.scale_dot_product_attention(q, k, v, casual_mask)
        x = rearrange(x, "bz h s d -> bz s (h d)")
        return self.output_proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope: Optional[RoPE] = None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor, casual_mask: torch.Tensor = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), casual_mask, token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta, device=None, dtype=None, shared_lm_head=False):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length, device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype) \
                for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        if shared_lm_head:
            self.lm_head = None
        else:
            self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: (batch_size, sequence_length)
        seq_len = x.shape[-1]
        x = self.token_embeddings(x)
        casual_mask = 1 - torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
        token_positions = torch.arange(seq_len, device=self.device)
        for layer in self.layers:
            x = layer(x, casual_mask, token_positions)
        x = self.ln_final(x)
        if self.lm_head != None:
            x = self.lm_head(x)
        else:
            x = x @ self.token_embeddings.weight.T
        return x
