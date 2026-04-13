import torch
import torch.nn as nn
import math
from core.config import Config


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)

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


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.data.embd_dim)
        self.ln2 = nn.LayerNorm(config.data.embd_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.data.embd_dim,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.data.embd_dim, config.model.hidden_dim),
            nn.GELU(),
            nn.Linear(config.model.hidden_dim, config.data.embd_dim),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x, t_emb=None):

        if t_emb is not None:
            x = x + t_emb
        x_norm = self.ln1(x)

        attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x
