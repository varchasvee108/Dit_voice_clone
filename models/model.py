import torch
import torch.nn as nn
from core.config import Config
from models.layers import (
    SinusoidalEmbeddings,
    TransformerBlock,
)


class DiffusionTransformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.data.vocab_size, config.data.embd_dim)

        self.pos_embd = nn.Parameter(
            torch.randn(1, config.data.block_size, config.data.embd_dim)
        )

        self.time_embd = nn.Sequential(
            SinusoidalEmbeddings(config.data.embd_dim),
            nn.Linear(config.model.time_embed, config.model.hidden_dim),
            nn.GELU(),
            nn.Linear(config.model.hidden_dim, config.data.embd_dim),
        )

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.model.num_layers)]
        )

        self.ln_f = nn.LayerNorm(config.data.embd_dim)
        self.out_proj = nn.Linear(config.data.embd_dim, config.data.embd_dim)

        # self.lm_head = nn.Linear(
        #     config.data.embd_dim, config.data.vocab_size, bias=False
        # )

        # self.lm_head.weight = self.tok_embeddings.weight

    def forward(self, x, t):
        if x.dtype == torch.long:
            x = self.tok_embeddings(x)

        B, T, _ = x.shape

        x = x + self.pos_embd[:, :T, :]

        t_emb = self.time_embd(t)
        t_emb = t_emb.unsqueeze(1)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.ln_f(x)
        return self.out_proj(x)
