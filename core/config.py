from dataclasses import dataclass
import tomllib
from pathlib import Path


@dataclass
class ProjectConfig:
    name: str
    seed: int
    output_dir: str


@dataclass
class DataConfig:
    train_dir: str
    sample_rate: int
    hop_length: int
    max_duration: float
    batch_size: int


@dataclass
class ModelConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    patch_size: int
    latent_dim: int
    text_emb_dim: int
    speaker_emb_dim: int


@dataclass
class TrainingConfig:
    lr: float
    weight_decay: float
    betas: list[float]
    total_steps: int
    grad_accum_factor: int
    precision: str
    grad_clip: float


@dataclass
class DiffusionConfig:
    timesteps: int
    scheduler: str


@dataclass
class Config:
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    diffusion: DiffusionConfig

    @classmethod
    def load_config(cls, path: str | Path) -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        training_args = data["training"]

        if "training" in data and "betas" in data["training"]:
            training_args = {
                **training_args,
                "betas": tuple(data["training"]["betas"]),
            }

        return cls(
            project=ProjectConfig(**data["project"]),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**training_args),
            diffusion=DiffusionConfig(**data["diffusion"]),
        )
