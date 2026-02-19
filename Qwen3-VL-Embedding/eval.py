import json
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from collections import Counter

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen3VLEmbedder(model_name_or_path=MODEL_NAME)

# =========================
# Utils
# =========================
def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def to_inputs(data):
    return [{"text": x["conversations"]} for x in data]

def get_labels(data, key="intent"):
    return [x[key] for x in data]

# =========================
# Retrieval Mode
# =========================
def retrieval_eval(query_path, doc_path, label_key="intent"):

    queries = load_jsonl(query_path)
    docs    = load_jsonl(doc_path)

    q_inputs = to_inputs(queries)
    d_inputs = to_inputs(docs)

    inputs = q_inputs + d_inputs
    emb = model.process(inputs)

    Q = emb[:len(q_inputs)]
    D = emb[len(q_inputs):]

    sim = Q @ D.T
    preds = sim.argmax(dim=1).cpu().tolist()

    gt = list(range(len(preds)))

    acc = accuracy_score(gt, preds)
    print(f"\nRetrieval accuracy (index match): {acc:.4f}")

    # Intent alignment
    q_intents = get_labels(queries, label_key)
    d_intents = get_labels(docs, label_key)

    intent_match = [
        q_intents[i] == d_intents[preds[i]]
        for i in range(len(preds))
    ]

    intent_acc = sum(intent_match) / len(intent_match)
    print(f"Intent alignment: {intent_acc:.4f}")

    return sim

# =========================
# Clustering Mode
# =========================
def clustering_eval(path, label_key="intent", k=20):

    data = load_jsonl(path)
    inputs = to_inputs(data)

    emb = model.process(inputs).cpu().numpy()

    kmeans = KMeans(n_clusters=k, random_state=0).fit(emb)
    clusters = kmeans.labels_

    labels = get_labels(data, label_key)

    cluster_purity = []

    for c in range(k):
        idx = [i for i,x in enumerate(clusters) if x==c]
        if len(idx)==0:
            continue
        majority = Counter([labels[i] for i in idx]).most_common(1)[0][1]
        purity = majority / len(idx)
        cluster_purity.append(purity)

    print(f"\nCluster purity ({label_key}): {sum(cluster_purity)/len(cluster_purity):.4f}")

    return clusters

# =========================
# Run Options
# =========================

# OPTION 1 — Retrieval
# Human as query, GPT as doc

retrieval_eval(
    "/home/hyang/BigML_sharding_dataset/Qwen3-VL-Embedding/data/consistent_chat/human_full.jsonl",
    "/home/hyang/BigML_sharding_dataset/Qwen3-VL-Embedding/data/consistent_chat/gpt_full.jsonl",
    label_key="intent"
)

# OPTION 2 — Clustering
# clustering_eval(
#     "/path/pair_full.jsonl",
#     label_key="intent",
#     k=20
# )
