#!/usr/bin/env python3
"""Embed one text file and save embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm


def _import_transformers_auto():
    try:
        from transformers import AutoModel, AutoTokenizer
        return AutoModel, AutoTokenizer
    except Exception as e:
        if "GenerationMixin" in str(e):
            raise RuntimeError(
                "Failed to import transformers AutoModel/AutoTokenizer due to a broken "
                "transformers install (GenerationMixin import error).\n"
                "Fix in your env:\n"
                "  pip install --no-cache-dir --force-reinstall "
                "'setuptools>=70' 'transformers==4.57.6' 'tokenizers>=0.22,<0.23' "
                "'huggingface_hub>=0.34.0'\n"
            ) from e
        raise


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def _read_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_jsonl_texts(path: Path, text_field: str) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            value = item.get(text_field)
            if value is None:
                continue
            rows.append(str(value).strip())
    return [r for r in rows if r]


def load_texts(path: Path, text_field: str) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".txt":
        return _read_text_lines(path)
    if ext == ".jsonl":
        return _read_jsonl_texts(path, text_field=text_field)
    if ext == ".csv":
        out = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get(text_field)
                if value:
                    out.append(value.strip())
        return [x for x in out if x]
    raise ValueError(f"Unsupported file format for {path}. Use .txt, .jsonl, or .csv")


@torch.no_grad()
def batched_embed(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    emb_dim: int,
    show_progress: bool = False,
) -> torch.Tensor:
    if not texts:
        return torch.empty((0, emb_dim), dtype=torch.float32, device=device)

    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc="Embedding")
    for start in iterator:
        batch = texts[start : start + batch_size]
        batch_dict = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        batch_emb = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        batch_emb = F.normalize(batch_emb, p=2, dim=1)
        all_embeddings.append(batch_emb)
    return torch.cat(all_embeddings, dim=0)


def setup_distributed(device_arg: str) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device(device_arg)
    else:
        device = torch.device(device_arg)

    return rank, world_size, device


def split_range(total: int, world_size: int, rank: int) -> tuple[int, int]:
    base = total // world_size
    rem = total % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def gather_variable_embeddings(local_emb: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return local_emb

    device = local_emb.device
    d = local_emb.shape[1]
    n_local = torch.tensor([local_emb.shape[0]], device=device, dtype=torch.int64)
    sizes = [torch.zeros_like(n_local) for _ in range(world_size)]
    dist.all_gather(sizes, n_local)
    sizes = [int(x.item()) for x in sizes]
    max_n = max(sizes) if sizes else 0

    padded = torch.zeros((max_n, d), dtype=local_emb.dtype, device=device)
    if local_emb.shape[0] > 0:
        padded[: local_emb.shape[0]] = local_emb

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([gathered[i][: sizes[i]] for i in range(world_size)], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed one file and save embeddings")
    parser.add_argument("--input-file", type=Path, required=True, help="Input .txt/.jsonl/.csv")
    parser.add_argument("--text-field", default="conversations", help="Text field for .jsonl/.csv")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Model name/path")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=8192, help="Tokenizer max length")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cuda:0/cpu)",
    )
    parser.add_argument("--output-path", type=Path, required=True, help="Output .pt path")
    args = parser.parse_args()
    rank, world_size, device = setup_distributed(args.device)
    is_main = rank == 0

    texts = load_texts(args.input_file, text_field=args.text_field)
    if not texts:
        raise ValueError("No texts loaded from input file.")
    if is_main:
        print(f"Loaded {len(texts)} texts from {args.input_file}")

    AutoModel, AutoTokenizer = _import_transformers_auto()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    emb_dim = int(model.config.hidden_size)

    start, end = split_range(len(texts), world_size, rank)
    local_texts = texts[start:end]
    embeddings_local = batched_embed(
        model=model,
        tokenizer=tokenizer,
        texts=local_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        emb_dim=emb_dim,
        show_progress=is_main,
    )
    embeddings = gather_variable_embeddings(embeddings_local, world_size).detach().cpu()

    if not is_main:
        if dist.is_initialized():
            dist.barrier()
        return

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings}, args.output_path)
    print(f"Saved embeddings: {args.output_path}")
    print(f"Embedding shape: {tuple(embeddings.shape)}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
