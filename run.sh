#!/bin/bash
# MediQ benchmark — ScaleExpert (rationale_generation) + FactSelectPatient
# Model: google/medgemma-1.5-4b-it  |  Backend: vLLM

PYTHON=/home/hyang/miniconda3/envs/scope/bin/python
SRC=/home/hyang/mediQ/src
DATA=/home/hyang/mediQ/data/med_data
FILE=all_dev_convo.jsonl
MODEL=google/medgemma-1.5-4b-it

cd $SRC && CUDA_VISIBLE_DEVICES=0 $PYTHON mediQ_benchmark.py \
  --expert_class ScaleExpert \
  --expert_model $MODEL \
  --patient_class FactSelectPatient \
  --patient_model $MODEL \
  --use_vllm \
  --data_dir $DATA \
  --dev_filename $FILE \
  --max_questions 10 \
  --rationale_generation \
  --output_filename results/scale_rg_medgemma4b.jsonl \
  --detail_log_filename logs/scale_rg_medgemma4b_detail.log \
  --convo_log_filename logs/scale_rg_medgemma4b_convo.jsonl \
  2>&1 | tee logs/scale_rg_medgemma4b.log
