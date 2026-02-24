#!/usr/bin/env python3
"""Batch embedding demo for Qwen3-Embedding with CSV exports."""

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


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


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
    desc: str = "Embedding",
) -> torch.Tensor:
    """Embed texts in batches and return a normalized [N, D] tensor on device."""
    if not texts:
        return torch.empty((0, emb_dim), dtype=torch.float32, device=device)

    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc=desc)

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
        batch_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(batch_embeddings)
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
        if torch.cuda.is_available():
            if world_size > 1:
                torch.cuda.set_device(local_rank)
                device = torch.device(f"cuda:{local_rank}")
            else:
                device = torch.device(device_arg)
        else:
            raise RuntimeError("CUDA device requested but CUDA is not available.")
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


def evaluate_diagonal_retrieval(
    scores: torch.Tensor, topk: int = 5
) -> tuple[float, float, list[int], list[int]]:
    n_queries, n_docs = scores.shape
    if n_docs < n_queries:
        raise ValueError(
            f"Diagonal GT requires num_docs >= num_queries, got {n_docs} docs and {n_queries} queries."
        )
    k = min(topk, n_docs)
    topk_idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices
    gt_idx = torch.arange(n_queries, device=scores.device).unsqueeze(1)

    hit1 = (topk_idx[:, :1] == gt_idx).squeeze(1).to(torch.int32)
    hitk = (topk_idx == gt_idx).any(dim=1).to(torch.int32)
    top1 = float(hit1.float().mean().item())
    topk_score = float(hitk.float().mean().item())
    return top1, topk_score, hit1.cpu().tolist(), hitk.cpu().tolist()


def write_summary_metrics_csv(
    path: Path,
    num_queries: int,
    num_documents: int,
    top1: float,
    top5: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_queries", "num_documents", "top1", "top5"])
        writer.writerow([num_queries, num_documents, f"{top1:.6f}", f"{top5:.6f}"])


def _model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].strip().lower().replace(" ", "_")


def _load_tensor_from_any_payload(path: Path, key: str | None = None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict):
        if key is not None and isinstance(payload.get(key), torch.Tensor):
            return payload[key]
        for candidate in ("embeddings", "query_embeddings", "doc_embeddings"):
            if isinstance(payload.get(candidate), torch.Tensor):
                return payload[candidate]
    raise ValueError(f"Could not find tensor in embeddings file: {path}")


def load_retrieval_embeddings(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(paths) == 1:
        payload = torch.load(paths[0], map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Single-path load requires dict payload with query/doc tensors: {paths[0]}"
            )
        q = payload.get("query_embeddings")
        d = payload.get("doc_embeddings")
        if isinstance(q, torch.Tensor) and isinstance(d, torch.Tensor):
            return q, d
        raise ValueError(
            f"Single-path load requires keys 'query_embeddings' and 'doc_embeddings': {paths[0]}"
        )
    if len(paths) == 2:
        q = _load_tensor_from_any_payload(paths[0], key="query_embeddings")
        d = _load_tensor_from_any_payload(paths[1], key="doc_embeddings")
        return q, d
    raise ValueError("--load-embeddings accepts either 1 path (combined) or 2 paths (query, doc).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-Embedding transformers batch demo")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Transformers embedding model name/path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for embedding calls",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store CSV result files",
    )
    parser.add_argument(
        "--task",
        default="Given a web search query, retrieve relevant passages that answer the query",
        help="Instruction text used to wrap queries for retrieval",
    )
    parser.add_argument("--queries-file", type=Path, default=None, help="Query file (.txt/.jsonl/.csv).")
    parser.add_argument("--documents-file", type=Path, default=None, help="Document file (.txt/.jsonl/.csv).")
    parser.add_argument(
        "--query-field",
        default="query",
        help="Field name for query text in .jsonl/.csv files",
    )
    parser.add_argument(
        "--document-field",
        default="document",
        help="Field name for document text in .jsonl/.csv files",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="If set, only use the first N queries after loading",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Tokenizer max_length",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference (e.g., cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--write-metrics-csv",
        action="store_true",
        help="Write summary metrics CSV (Top-1/Top-5 only).",
    )
    parser.add_argument(
        "--convo-part",
        default="full",
        choices=["full", "first", "last", "first_last"],
        help="Conversation slice tag used in output naming.",
    )
    parser.add_argument(
        "--load-embeddings",
        nargs="+",
        type=Path,
        default=None,
        help="Load embeddings instead of encoding. Use 1 combined path or 2 paths: <query_emb> <doc_emb>.",
    )
    args = parser.parse_args()

    rank, world_size, device = setup_distributed(args.device)
    is_main = rank == 0

    raw_queries: list[str] = []
    documents: list[str] = []
    queries: list[str] = []

    if args.load_embeddings is not None:
        if not is_main:
            if dist.is_initialized():
                dist.barrier()
            return
        qry_emb, doc_emb = load_retrieval_embeddings(args.load_embeddings)
        print(f"Loaded embeddings from: {', '.join(str(p) for p in args.load_embeddings)}")
    else:
        if args.queries_file is None or args.documents_file is None:
            raise ValueError(
                "When --load-embeddings is not provided, both --queries-file and --documents-file are required."
            )
        raw_queries = load_texts(args.queries_file, text_field=args.query_field)
        documents = load_texts(args.documents_file, text_field=args.document_field)
        if not raw_queries:
            raise ValueError("No queries loaded.")
        if not documents:
            raise ValueError("No documents loaded.")
        if args.max_queries is not None:
            if args.max_queries <= 0:
                raise ValueError("--max-queries must be > 0")
            raw_queries = raw_queries[: args.max_queries]
        queries = [get_detailed_instruct(args.task, q) for q in raw_queries]

        AutoModel, AutoTokenizer = _import_transformers_auto()
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        model = AutoModel.from_pretrained(args.model).to(device).eval()
        emb_dim = int(model.config.hidden_size)

        doc_start, doc_end = split_range(len(documents), world_size, rank)
        local_docs = documents[doc_start:doc_end]
        local_doc_emb = batched_embed(
            model=model,
            tokenizer=tokenizer,
            texts=local_docs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            emb_dim=emb_dim,
            show_progress=is_main,
            desc=f"Embedding docs (rank {rank}, {len(local_docs)} items)",
        )
        doc_emb = gather_variable_embeddings(local_doc_emb, world_size)

        qry_start, qry_end = split_range(len(queries), world_size, rank)
        local_queries = queries[qry_start:qry_end]
        local_qry_emb = batched_embed(
            model=model,
            tokenizer=tokenizer,
            texts=local_queries,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            emb_dim=emb_dim,
            show_progress=is_main,
            desc=f"Embedding queries (rank {rank}, {len(local_queries)} items)",
        )
        qry_emb = gather_variable_embeddings(local_qry_emb, world_size)

        if not is_main:
            if dist.is_initialized():
                dist.barrier()
            return

        qry_emb = qry_emb.detach().cpu()
        doc_emb = doc_emb.detach().cpu()

    num_queries = int(qry_emb.shape[0])
    num_documents = int(doc_emb.shape[0])

    scores_cpu = (qry_emb @ doc_emb.T).cpu()
    if num_queries <= 5 and num_documents <= 20:
        print(scores_cpu.tolist())
    else:
        print(f"Computed score matrix with shape: {tuple(scores_cpu.shape)}")
    print(f"Used {num_queries} queries and {num_documents} documents on {world_size} process(es).")

    top1, top5, _, _ = evaluate_diagonal_retrieval(scores_cpu, topk=5)
    print(f"Top-1: {top1:.4f}")
    print(f"Top-5: {top5:.4f}")

    if args.write_metrics_csv:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = args.output_dir / f"{_model_tag(args.model)}_retrieval_{args.convo_part}.csv"
        write_summary_metrics_csv(
            metrics_csv,
            num_queries=num_queries,
            num_documents=num_documents,
            top1=top1,
            top5=top5,
        )
        print(f"Wrote metrics CSV: {metrics_csv}")

    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
