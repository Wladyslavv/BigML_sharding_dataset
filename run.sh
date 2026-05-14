#!/bin/bash
# MediQ benchmark — ScaleExpert (rationale_generation) + FactSelectPatient
# Model: google/medgemma-1.5-4b-it  |  Backend: vLLM
#
# GPUs: defaults to physical GPUs 2 and 3 with tensor_parallel_size=2 (one vLLM engine
# sharded across both). Override before running, e.g.:
#   export CUDA_VISIBLE_DEVICES=0,1
#
# Optional vLLM tuning via env (see src/helper.py):
#   MEDIQ_VLLM_GPU_MEMORY_UTILIZATION   fraction of total VRAM (0–1)
#   MEDIQ_VLLM_MAX_MODEL_LEN            KV cache context cap
#   MEDIQ_VLLM_MAX_NUM_SEQS             max concurrent sequences
#   MEDIQ_VLLM_ENFORCE_EAGER            set to 1 to disable CUDA graphs
#   MEDIQ_VLLM_SWAP_SPACE_GB            CPU swap (GiB) for paged attention
# Still OOM: lower --batch_size / --vllm_max_num_seqs or --vllm_max_model_len,
#   or a quantized checkpoint (vLLM --quantization; see vLLM docs).
#
# Speed (after vLLM is stable): keep max_model_len only as large as you need — huge
# values waste KV cache and slow startup. max_tokens caps worst-case decode length.
# Omit --vllm_enforce_eager so CUDA graphs stay on (faster). Raise gpu_memory_utilization
# if you have headroom: export MEDIQ_VLLM_GPU_MEMORY_UTILIZATION=0.88

PYTHON=/home/hyang/miniconda3/envs/scope/bin/python
SRC=/home/hyang/mediQ/src
DATA=/home/hyang/mediQ/data/med_data
FILE=all_dev_convo.jsonl
MODEL=google/medgemma-1.5-4b-it
LOGS=/home/hyang/mediQ/logs
RESULTS=/home/hyang/mediQ/results

# ── Models ───────────────────────────────────────────────────────────────────
# expert_class: ScaleExpert | BasicExpert | FixedExpert | BinaryExpert | NumericalExpert | HumanExpert
# expert_module: python file containing the expert class (default: expert)
# expert_model: HuggingFace model ID or local path; used for confidence + shadow answer calls
# expert_model_question_generator: separate model for follow-up questions; set to same as
#   MODEL so one vLLM engine is used (required when using tensor parallel across 2 GPUs).
# patient_class: FactSelectPatient | InstructPatient | DirectPatient | RandomPatient | FullContextPatient
# patient_module: python file containing the patient class (default: patient)
# patient_model: model for patient responses; also used as judge when options are hidden

# ── Data ───────────────────────────────────────────────────────────────────
# data_dir / dev_filename: location of the JSONL dev set
# max_examples: number of cases to run; -1 = all

# ── Output ──────────────────────────────────────────────────────────────────
# output_filename: per-case results JSONL; _hidden suffix may apply when using no-option / hide flow
# overwrite: clear output + log files before running; omit to resume from checkpoint

# ── Interaction control ─────────────────────────────────────────────────────
# max_questions: max doctor turns before forcing a final answer
# abstain_threshold: Likert score (1–5) below which the expert asks another question
# option_mode: yes-option (expert always sees A/B/C/D) | no-option (never sees options,
#   answers in \box{}, judged) | option-in-the-end (options only at final decision)
# rationale_generation: expert produces a REASON line before confidence / answer (uncomment flag below)
#   --rationale_generation
# self_consistency: repeat each LLM call N times and majority-vote (default: 1)
#   --self_consistency 1
# independent_modules: each cognitive module starts fresh with no prior conversation
#   --independent_modules

# ── Generation ─────────────────────────────────────────────────────────────
# max_tokens: max new tokens per LLM call
# temperature / top_p: sampling controls (defaults 0.6 / 0.9)
# top_logprobs: top logprobs per token; 0 = disabled
#   --top_logprobs 0

# ── Backend: vLLM ───────────────────────────────────────────────────────────
# use_vllm: use vLLM for inference; omit to fall back to HuggingFace generate
# vllm_max_model_len: KV cache context cap; lower saves VRAM
# batch_size: alias for vLLM max_num_seqs when vllm_max_num_seqs is unset in older flows
# vllm_max_num_seqs: max concurrent sequences; lower (8–32) saves VRAM
# vllm_enforce_eager: disable CUDA graph capture — less peak VRAM, slower throughput
# tensor_parallel_size: must match number of GPUs in CUDA_VISIBLE_DEVICES (here: 2)
# gpu_memory_utilization: fraction of GPU VRAM to target (0–1); helper auto-infers if omitted
#   --gpu_memory_utilization 0.5

# ── Backend: API (alternative to vLLM) ─────────────────────────────────────
# use_api: OpenAI-compatible API instead of vLLM; choices: openai
#   --use_api openai
# api_account: key name in keys.py
#   --api_account mediQ

# ── Logging ──────────────────────────────────────────────────────────────────
# convo_log_filename: full per-case log — confidence rationale, shadow answer, turn details
# doctor_log_filename: doctor-view JSONL — initial_info, Q&A pairs, final answer
# log_filename: general benchmark progress (accuracy after each case)
#   --log_filename "$LOGS/results.log"
# history_log_filename / detail_log_filename / message_log_filename: optional extra logs
#   --history_log_filename "$LOGS/history.log"
#   --detail_log_filename "$LOGS/detail.log"
#   --message_log_filename "$LOGS/messages.log"

cd "$SRC" && CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}" $PYTHON mediQ_benchmark.py \
  --expert_class ScaleExpert \
  --expert_module expert \
  --expert_model "$MODEL" \
  --expert_model_question_generator "$MODEL" \
  --patient_class FactSelectPatient \
  --patient_module patient \
  --patient_model "$MODEL" \
  --data_dir "$DATA" \
  --dev_filename "$FILE" \
  --max_examples 100 \
  --output_filename "$RESULTS/results.jsonl" \
  --max_questions 10 \
  --abstain_threshold 4.0 \
  --option_mode no-option \
  --max_tokens 2048 \
  --temperature 0.6 \
  --top_p 0.9 \
  --use_vllm \
  --vllm_max_model_len 32768 \
  --batch_size 1 \
  --vllm_max_num_seqs 32 \
  --tensor_parallel_size 2 \
  --convo_log_filename "$LOGS/scale_rg_medgemma4b_convo_no_options.jsonl" \
  --doctor_log_filename "$LOGS/scale_rg_medgemma4b_no_options_doctor_view.jsonl"
