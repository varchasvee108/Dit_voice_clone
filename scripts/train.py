import torch
import torch.nn as nn
from core.config import Config
from models.model import DiffusionTransformer
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler
from tqdm import tqdm
from diffusion.diffusion import Diffusion
import wandb
import os
import argparse
from torch.optim import AdamW
from data.dataloader import get_dataloader
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text diffusion model")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

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


def evaluate(model, dataloader, diffusion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            if x.dtype == torch.long:
                x = model.token_emb(x)
            t = diffusion.sample_timesteps(x.shape[0], device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)
            total_loss += loss.item()
    model.train()
    return total_loss / len(dataloader)


def train():
    args = parse_args()
    config: Config = Config(args.config)
    wandb.init(project="Text-Diffusion", config=config.to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoint", exist_ok=True)
    model = DiffusionTransformer(config).to(device)
    diffusion = Diffusion(config, device)

    optimizer, scheduler, scaler = get_optimizer_and_scheduler(model, config, device)
    train_dataloader, val_dataloader = get_dataloader(config)
    train_iter = iter(train_dataloader)
    best_loss = float("inf")

    pbar = tqdm(
        range(config.training.max_steps),
        desc="Training",
        unit="step",
        total=config.training.max_steps,
    )

    if args.resume and os.path.exists("checkpoints/best_model.pt"):
        checkpoint = torch.load("checkpoints/best_model.pt")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"] + 1
        best_loss = checkpoint["loss"]
        pbar.update(step)
        print(f"Resumed training from step {step}")
    else:
        step = 0

    model.train()

    for step in pbar:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            x = next(train_iter)

        x = x.to(device)

        if x.dtype == torch.long:
            x = model.token_emb(x)

        t = diffusion.sample_timesteps(x.shape[0], device)
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise)

        with autocast():
            pred_noise = model(x_t, t)
            loss = F.mse_loss(pred_noise, noise)

        optimizer.zero_grad(set_to_None=True)
        scaler.scale(loss).backward()
        scaler.unscale(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        wandb.log(
            {"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0]},
            step=step,
        )

        if step % args.eval_every == 0 and step > 0:
            eval_loss = evaluate(model, val_dataloader, diffusion, device)
            wandb.log({"eval/loss": eval_loss}, step=step)
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    step,
                    best_loss,
                    "checkpoints/best_model.pt",
                )

    wandb.finish()


if __name__ == "__main__":
    train()
