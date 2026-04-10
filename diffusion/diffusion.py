import torch
from core.config import Config
from tqdm import tqdm


class DiffusionModel:
    def __init__(self, config: Config, device):
        self.config = config
        self.device = device
        self.betas = torch.linspace(
            config.diffusion.beta_start,
            config.diffusion.beta_end,
            config.diffusion.timesteps,
            device=device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int):
        return torch.randint(
            0,
            self.config.diffusion.timesteps,
            size=(batch_size,),
            device=self.device,
        )

    def q_sample(self, x0, t, noise):
        alpha_hat = self.alpha_cumprod[t]
        sqrt_alpha_hat = torch.sqrt(alpha_hat).view(-1, 1, 1)
        sqrt_one_minus = torch.sqrt(1 - alpha_hat).view(-1, 1, 1)

        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def p_sample(self, model, x, t):

        beta = self.betas[t].view(-1, 1, 1)
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_hat = self.alpha_cumprod[t].view(-1, 1, 1)
        noise_pred = model(x, t)

        mean = (1 / torch.sqrt(alpha)) * x - (
            (1 - alpha) / torch.sqrt(1 - alpha_hat)
        ) * noise_pred

        noise = torch.randn_like(x)
        mask = (t > 0).float().view(-1, 1, 1)
        return mean + mask * torch.sqrt(beta) * noise

    @torch.inference_mode()
    def sample(self, model, shape):
        model.eval()
        x = torch.randn(shape, device=self.device)
        pbar = tqdm(
            reversed(range(self.config.diffusion.timesteps)),
            desc="Sampling",
            total=self.config.diffusion.timesteps,
        )

        for t in pbar:
            t_batch = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        return x
