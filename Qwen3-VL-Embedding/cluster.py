#!/usr/bin/env python3
"""Text clustering with Qwen3 embeddings and label-based GT evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
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


@torch.no_grad()
def batched_embed(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    all_embeddings = []
    iterator = range(0, len(texts), batch_size)
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
    if not all_embeddings:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    return torch.cat(all_embeddings, dim=0)


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_rows(path: Path) -> list[dict]:
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return load_jsonl_rows(path)
    if ext == ".csv":
        return load_csv_rows(path)
    raise ValueError(f"Unsupported input format: {path}. Use .jsonl or .csv")


def prepare_dataset(
    rows: list[dict], text_field: str, label_field: str, max_samples: int | None = None
) -> tuple[list[str], list[str]]:
    texts = []
    labels = []
    for row in rows:
        text = row.get(text_field)
        label = row.get(label_field)
        if text is None or label is None:
            continue
        text = str(text).strip()
        label = str(label).strip()
        if not text or not label:
            continue
        texts.append(text)
        labels.append(label)
        if max_samples is not None and len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError("No valid rows found after filtering text/label fields.")
    return texts, labels


@torch.no_grad()
def kmeans_cosine(
    embeddings: torch.Tensor,
    k: int,
    max_iters: int = 50,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = embeddings.shape
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot be > number of samples ({n})")

    g = torch.Generator(device=embeddings.device)
    g.manual_seed(seed)
    init_idx = torch.randperm(n, generator=g, device=embeddings.device)[:k]
    centroids = embeddings[init_idx].clone()
    centroids = F.normalize(centroids, p=2, dim=1)

    prev_assign = None
    for _ in tqdm(range(max_iters), desc="KMeans"):
        sim = embeddings @ centroids.T
        assign = torch.argmax(sim, dim=1)

        if prev_assign is not None and torch.equal(assign, prev_assign):
            break
        prev_assign = assign.clone()

        new_centroids = torch.zeros((k, d), dtype=embeddings.dtype, device=embeddings.device)
        counts = torch.bincount(assign, minlength=k)
        new_centroids.index_add_(0, assign, embeddings)

        empty = counts == 0
        if empty.any():
            refill_idx = torch.randperm(n, generator=g, device=embeddings.device)[: int(empty.sum().item())]
            new_centroids[empty] = embeddings[refill_idx]
            counts[empty] = 1

        new_centroids = new_centroids / counts.unsqueeze(1)
        centroids = F.normalize(new_centroids, p=2, dim=1)

    final_sim = embeddings @ centroids.T
    final_assign = torch.argmax(final_sim, dim=1)
    return final_assign, centroids


def purity_score(gt: list[int], pred: list[int]) -> float:
    by_cluster = defaultdict(list)
    for g, p in zip(gt, pred):
        by_cluster[p].append(g)
    correct = 0
    for members in by_cluster.values():
        correct += Counter(members).most_common(1)[0][1]
    return correct / max(1, len(gt))


def nmi_score(gt: list[int], pred: list[int]) -> float:
    n = len(gt)
    if n == 0:
        return 0.0

    gt_counts = Counter(gt)
    pred_counts = Counter(pred)
    joint = Counter(zip(gt, pred))

    mi = 0.0
    for (g, p), n_gp in joint.items():
        p_gp = n_gp / n
        p_g = gt_counts[g] / n
        p_p = pred_counts[p] / n
        mi += p_gp * math.log((p_gp / (p_g * p_p)) + 1e-12)

    h_g = -sum((c / n) * math.log((c / n) + 1e-12) for c in gt_counts.values())
    h_p = -sum((c / n) * math.log((c / n) + 1e-12) for c in pred_counts.values())
    denom = math.sqrt(max(h_g * h_p, 1e-12))
    return mi / denom


def write_assignments_csv(
    path: Path, texts: list[str], gt_labels: list[str], pred_cluster: list[int]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "text", "gt_label", "pred_cluster"])
        for i, (t, g, p) in enumerate(zip(texts, gt_labels, pred_cluster)):
            writer.writerow([i, t, g, p])


def write_cluster_metrics_csv(
    path: Path,
    num_samples: int,
    num_clusters: int,
    purity: float,
    nmi: float,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["num_samples", "num_clusters", "purity", "nmi"])
        writer.writerow([num_samples, num_clusters, f"{purity:.6f}", f"{nmi:.6f}"])


def _model_tag(model_name: str) -> str:
    return model_name.split("/")[-1].strip().lower().replace(" ", "_")


def load_cluster_embeddings(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), torch.Tensor):
        raise ValueError(f"Invalid embeddings file format: {path}")
    return payload["embeddings"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 embedding clustering")
    parser.add_argument("--input-file", type=Path, required=True, help="Input .jsonl or .csv")
    parser.add_argument("--text-field", default="conversations", help="Text field key")
    parser.add_argument("--label-field", required=True, help="Label field key for clustering GT")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B", help="Model name/path")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--max-length", type=int, default=8192, help="Tokenizer max length")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cuda:0/cpu)",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    parser.add_argument("--kmeans-iters", type=int, default=50, help="Max k-means iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument(
        "--write-assignments-csv",
        action="store_true",
        help="Write per-sample assignments CSV",
    )
    parser.add_argument(
        "--write-metrics-csv",
        action="store_true",
        help="Write summary clustering metrics CSV (purity, nmi).",
    )
    parser.add_argument(
        "--convo-part",
        default="full",
        choices=["full", "first", "last", "first_last"],
        help="Conversation slice tag used in output naming.",
    )
    parser.add_argument(
        "--load-embeddings",
        type=Path,
        default=None,
        help="Optional path to load embeddings (.pt) and skip encoding.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_file)
    texts, gt_labels = prepare_dataset(
        rows, text_field=args.text_field, label_field=args.label_field, max_samples=args.max_samples
    )

    unique_labels = sorted(set(gt_labels))
    k = len(unique_labels)
    label2id = {v: i for i, v in enumerate(unique_labels)}
    gt_ids = [label2id[x] for x in gt_labels]

    print(f"Loaded {len(texts)} samples from {args.input_file}")
    print(f"Label field: {args.label_field}")
    print(f"Unique GT labels (num clusters): {k}")

    if args.load_embeddings is not None:
        embeddings = load_cluster_embeddings(args.load_embeddings)
        if embeddings.shape[0] != len(texts):
            raise ValueError(
                f"Loaded embeddings ({embeddings.shape[0]}) != sample count ({len(texts)})."
            )
        print(f"Loaded embeddings from: {args.load_embeddings}")
    else:
        device = torch.device(args.device)
        AutoModel, AutoTokenizer = _import_transformers_auto()
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        model = AutoModel.from_pretrained(args.model).to(device).eval()

        embeddings = batched_embed(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        embeddings = embeddings.detach().cpu()

    pred_ids_t, _ = kmeans_cosine(
        embeddings=embeddings,
        k=k,
        max_iters=args.kmeans_iters,
        seed=args.seed,
    )
    pred_ids = pred_ids_t.detach().cpu().tolist()

    purity = purity_score(gt_ids, pred_ids)
    nmi = nmi_score(gt_ids, pred_ids)
    print(f"Purity: {purity:.4f}")
    print(f"NMI: {nmi:.4f}")

    if args.write_assignments_csv:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "cluster_assignments.csv"
        write_assignments_csv(out_path, texts, gt_labels, pred_ids)
        print(f"Wrote assignments CSV: {out_path}")

    if args.write_metrics_csv:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{_model_tag(args.model)}_cluster_{args.convo_part}.csv"
        write_cluster_metrics_csv(
            out_path,
            num_samples=len(texts),
            num_clusters=k,
            purity=purity,
            nmi=nmi,
        )
        print(f"Wrote metrics CSV: {out_path}")


if __name__ == "__main__":
    main()
