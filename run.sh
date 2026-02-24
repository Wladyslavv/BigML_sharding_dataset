python Qwen3-VL-Embedding/embed.py \
  --input-file path/to/data.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --text-field conversations \
  --output-path Qwen3-VL-Embedding/embed/consistent_chat_6b_full.pt


python Qwen3-VL-Embedding/retrieve.py \
  --load-embeddings Qwen3-VL-Embedding/outputs/query_emb.pt Qwen3-VL-Embedding/outputs/doc_emb.pt \
  --write-metrics-csv \
  --convo-part full \
  --output-dir Qwen3-VL-Embedding/outputs



torchrun --nproc_per_node=8 Qwen3-VL-Embedding/retrieve.py \
  --queries-file ... \
  --documents-file ... \
  --query-field conversations \
  --document-field conversations \
  --max-queries 1000



python Qwen3-VL-Embedding/retrieve.py \
  --load-embeddings Qwen3-VL-Embedding/outputs/query_emb.pt Qwen3-VL-Embedding/outputs/doc_emb.pt \
  --write-metrics-csv \
  --convo-part full \
  --output-dir Qwen3-VL-Embedding/outputs



python Qwen3-VL-Embedding/cluster.py \
  --input-file ... \
  --text-field conversations \
  --label-field intent \
  --load-embeddings Qwen3-VL-Embedding/outputs/cluster_emb.pt \
  --write-metrics-csv \
  --convo-part full
