import os
import json
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist

from vllm import LLM

# =========================
# Configuration
# =========================
MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
MAX_LENGTH = 8192
BATCH_SIZE = 16
TOPK = 5

# =========================
# Distributed setup
# =========================
def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

# =========================
# vLLM prompt formatting (text-only)
# =========================
def format_input_to_conversation(
    text: str,
    instruction: str = "Represent the user's input.",
):
    # Qwen chat template expects list[{"role":..., "content":[...]}]
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "text", "text": text or ""}]},
    ]

def make_vllm_embed_inputs(
    texts,
    tokenizer,
    instruction: str = "Represent the user's input.",
):
    # vLLM embed() accepts list[{"prompt": str, "multi_modal_data": Optional[dict]}]
    out = []
    for t in texts:
        conv = format_input_to_conversation(t, instruction=instruction)
        prompt = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
        )
        out.append({"prompt": prompt, "multi_modal_data": None})
    return out

@torch.no_grad()
def vllm_embed_batched(
    llm: LLM,
    texts,
    batch_size=BATCH_SIZE,
    instruction: str = "Represent the user's input.",
    device="cuda",
):
    """
    Returns: torch.Tensor [N, D] on `device`, L2-normalized.
    """
    tokenizer = llm.get_tokenizer()

    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        vllm_inputs = make_vllm_embed_inputs(batch_texts, tokenizer, instruction=instruction)

        outputs = llm.embed(vllm_inputs)

        # vLLM returns python lists; move to torch on GPU
        batch_emb = torch.tensor(
            [o.outputs.embedding for o in outputs],
            dtype=torch.float32,  # normalize in fp32
            device=device,
        )
        batch_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
        all_emb.append(batch_emb)

    if not all_emb:
        return torch.empty((0, 0), device=device)
    return torch.cat(all_emb, dim=0)

# =========================
# Gather helper (variable sizes)
# =========================
def gather_haystack(local_emb: torch.Tensor, world_size: int):
    """
    Gathers [n_local, d] from all ranks into [n_total, d] on each rank.
    """
    device = local_emb.device
    d = local_emb.shape[1]
    n_local = torch.tensor([local_emb.shape[0]], device=device, dtype=torch.int64)

    sizes = [torch.zeros_like(n_local) for _ in range(world_size)]
    dist.all_gather(sizes, n_local)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes) if sizes else 0

    padded = torch.zeros((max_n, d), device=device, dtype=local_emb.dtype)
    if local_emb.shape[0] > 0:
        padded[: local_emb.shape[0]] = local_emb

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    return torch.cat([gathered[r][: sizes[r]] for r in range(world_size)], dim=0)

# =========================
# Main Retrieval Logic (vLLM)
# =========================
def run_retrieval_vllm(query_path, doc_path, n_queries=256, seed=42):
    rank, world_size = setup_distributed()

    # ---------- load data ----------
    with open(query_path) as f:
        all_queries = [json.loads(l) for l in f]
    with open(doc_path) as f:
        all_docs = [json.loads(l) for l in f]

    random.seed(seed)
    q_indices = random.sample(range(len(all_queries)), n_queries)
    selected_queries = [all_queries[i] for i in q_indices]

    # ---------- init vLLM on this rank's GPU ----------
    # Each torchrun rank uses exactly 1 GPU. (So set tensor_parallel_size=1.)
    llm = LLM(
        model=MODEL_NAME,
        runner="pooling",
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=MAX_LENGTH,
        gpu_memory_utilization=0.90,
    )

    # ---------- encode documents (sharded) ----------
    local_docs = all_docs[rank::world_size]
    local_doc_texts = [d["conversations"] for d in local_docs]

    if rank == 0:
        print(f"Encoding {len(all_docs)} documents across {world_size} GPUs...")

    emb_d_local = vllm_embed_batched(llm, local_doc_texts, batch_size=BATCH_SIZE, device="cuda")
    D_full = gather_haystack(emb_d_local, world_size)  # [num_docs, dim] on each rank

    # ---------- encode & score queries (sharded) ----------
    my_queries = selected_queries[rank::world_size]
    my_q_indices = q_indices[rank::world_size]  # NOTE: assumes gt index is the same id space as docs

    if rank == 0:
        print(f"Scoring {n_queries} queries...")

    local_metrics = []
    for i in range(0, len(my_queries), BATCH_SIZE):
        batch_q = [q["conversations"] for q in my_queries[i : i + BATCH_SIZE]]
        batch_gt = my_q_indices[i : i + BATCH_SIZE]

        emb_q = vllm_embed_batched(llm, batch_q, batch_size=BATCH_SIZE, device="cuda")  # [b, d]

        # cosine sim since both normalized
        sim = emb_q @ D_full.T  # [b, num_docs]
        topk_vals, topk_idx = torch.topk(sim, k=TOPK, dim=1)

        topk_idx_cpu = topk_idx.detach().cpu().tolist()
        for j, gt_idx in enumerate(batch_gt):
            preds = topk_idx_cpu[j]
            rank_pos = preds.index(gt_idx) + 1 if gt_idx in preds else 0
            local_metrics.append(
                {
                    "mrr": 1.0 / rank_pos if rank_pos > 0 else 0.0,
                    "top1": 1 if rank_pos == 1 else 0,
                    "top5": 1 if rank_pos > 0 else 0,
                }
            )

    # ---------- gather metrics ----------
    all_results = [None] * world_size
    dist.all_gather_object(all_results, local_metrics)

    if rank == 0:
        final_flat = [x for sub in all_results for x in sub]
        avg_top1 = sum(x["top1"] for x in final_flat) / max(1, len(final_flat))
        avg_top5 = sum(x["top5"] for x in final_flat) / max(1, len(final_flat))
        avg_mrr = sum(x["mrr"] for x in final_flat) / max(1, len(final_flat))

        print("\n" + "=" * 40)
        print(f"FINAL PERFORMANCE (1 vs {len(all_docs)})")
        print("-" * 40)
        print(f"Top-1 Accuracy:  {avg_top1:.4f}")
        print(f"Top-{TOPK} Recall:    {avg_top5:.4f}")
        print(f"MRR@{TOPK}:           {avg_mrr:.4f}")
        print("=" * 40)

def _destroy_dist():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    try:
        run_retrieval_vllm(
            "/home/hyang/BigML_sharding_dataset/Qwen3-VL-Embedding/data/consistent_chat/human_full.jsonl",
            "/home/hyang/BigML_sharding_dataset/Qwen3-VL-Embedding/data/consistent_chat/gpt_full.jsonl",
            n_queries=256,
        )
    finally:
        _destroy_dist()
