import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim)
        emb = torch.exp(torch.arange(0, half_dim, 2, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        max_log_val = math.log(10000)
        half_dim = dim // 2
        step_size = max_log_val / half_dim
        indices = torch.arange(0, half_dim, 2).float()
        inv_freq = torch.exp(-step_size * indices)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_length: int, device: torch.device):
        t = torch.arange(seq_length, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
