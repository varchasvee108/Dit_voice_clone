import torch
import torch.nn as nn
from core.config import Config
from models.model import DiffusionTransformer
from transformers import get_scheduler
from tqdm import tqdm
from diffusion.diffusion import DiffusionModel
import wandb
import os
import argparse
from torch.optim import AdamW
from data.dataloader import get_dataloader
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import tomllib


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text diffusion model")
    parser.add_argument(
        "--config", type=str, default="config/config.toml", help="Path to config file"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    parser.add_argument(
        "--eval_every", type=int, default=500, help="Evaluate model every N steps"
    )
    parser.add_argument("--resume", action="store_true")

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
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay,
    )
    scheduler = get_scheduler(
        name=config.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.max_steps,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())
    return optimizer, scheduler, scaler


def evaluate(model, dataloader, diffusion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            if x.dtype == torch.long:
                x = model.tok_embeddings(x)
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
    config: Config = Config.load_config(args.config)

    train_dataloader, val_dataloader = get_dataloader(config)
    config.data.vocab_size = train_dataloader.dataset.tokenizer.vocab_size
    with open(args.config, "rb") as f:
        config_dict = tomllib.load(f)
    wandb.init(project="Text-Diffusion", config=config_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)
    model = DiffusionTransformer(config).to(device)
    diffusion = DiffusionModel(config, device)

    optimizer, scheduler, scaler = get_optimizer_and_scheduler(model, config, device)
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
            x = model.tok_embeddings(x)

        t = diffusion.sample_timesteps(x.shape[0])
        noise = torch.randn_like(x)
        x_t = diffusion.q_sample(x, t, noise)

        with autocast(device_type="cuda", enabled=device.type == "cuda"):
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
