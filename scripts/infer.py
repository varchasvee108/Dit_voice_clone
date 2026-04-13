import torch
from transformers import AutoTokenizer
from core.config import Config
from models.model import DiffusionTransformer
from diffusion.diffusion import DiffusionModel


@torch.inference_mode()
def infer(config_path, ckpt_path, num_samples=4, seq_len=None):
    config = Config.load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    config.data.vocab_size = tokenizer.vocab_size

    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()

    diffusion = DiffusionModel(config, device)

    T = seq_len or config.data.block_size
    shape = (num_samples, T, config.data.embd_dim)

    emb = diffusion.sample(model, shape)  # [B,T,D]

    vocab_emb = model.tok_embeddings.weight  # [V,D]
    dist = torch.cdist(emb, vocab_emb)  # [B,T,V]
    tokens = dist.argmin(dim=-1)  # [B,T]

    texts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    for i, t in enumerate(texts):
        print(f"[{i}] {t}\n")
