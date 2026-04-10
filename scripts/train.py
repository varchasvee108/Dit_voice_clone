import torch
import torch.nn as nn
from core.config import Config
from models.model import DiffusionTransformer
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusion.diffusion import Diffusion
import wandb
import os
import argparse
from torch.optim import AdamW


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text diffusion model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--save_every", type=int, default=500, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--eval_every", type=int, default=500, help="Evaluate model every N steps"
    )

    return parser.parse_args()


def save_checkpoint(model, optimizer, scaler, scheduler, step, loss, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def get_optimizer_and_scheduler(model, config, device):
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    scheduler = get_scheduler(
        name=config.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.num_warmup_steps,
        num_training_steps=config.training.num_training_steps,
    )
    scaler = GradScaler()
    return optimizer, scheduler, scaler


def train():
    args = parse_args()
    config: Config = Config(args.config)
    wandb.init(project="Text-Diffusion", config=config.to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoint", exist_ok=True)
    model = DiffusionTransformer(config).to(device)
    diffusion = Diffusion(config, device)

    optimizer, scheduler, scaler = get_optimizer_and_scheduler(model, config, device)
