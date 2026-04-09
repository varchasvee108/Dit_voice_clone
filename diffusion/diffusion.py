import torch
import torch.nn as nn
from core.config import Config


class DiffusionModel(nn.Module):
    def __init__(self, config: Config, device):
        super().__init__()
        self.config = config
        self.betas = torch.linspace(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            config.diffusion.timesteps,
            device=device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
