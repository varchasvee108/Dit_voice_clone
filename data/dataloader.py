from text_dataset.dataset import TextDataset
from torch.utils.data import DataLoader
from core.config import Config


def get_dataloader(config: Config):
    train_dataset = TextDataset(config, "train")
    val_dataset = TextDataset(config, "val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=4,
    )
    return train_dataloader, val_dataloader
