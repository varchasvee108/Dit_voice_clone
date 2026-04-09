from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class ProjectConfig:
    name: str
    seed: int


@dataclass
class DataConfig:
    dataset: str
    block_size: int
    batch_size: int
    tokenizer: str
    vocab_size: int | None = None
    embd_dim: int


@dataclass
class ModelConfig:
    num_layers: int
    num_heads: int
    dropout: float
    hidden_dim: int
    time_embed: int


@dataclass
class TrainingConfig:
    lr: float
    max_steps: int
    warmup_steps: int
    betas: list[float]
    weight_decay: float
    grad_clip: float
    eval_interval: int
    save_interval: int
    scheduler: str


@dataclass
class DiffusionConfig:
    timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float


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

        return cls(
            project=ProjectConfig(**data["project"]),
            data=DataConfig(**data["data"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(
                **data["training"], betas=tuple(data["training"]["betas"])
            ),
            diffusion=DiffusionConfig(**data["diffusion"]),
        )
