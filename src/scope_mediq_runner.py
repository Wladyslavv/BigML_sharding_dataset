"""
scope_mediq_runner.py

Integrates SCOPE's MCTS planner as the expert in mediQ's benchmark loop.

Two simulators (clearly separated):
  1. SCOPE's transition model  — inner simulator used inside MCTS for semantic-space planning
  2. mediQ's FactSelectPatient — outer simulator that gives real patient answers to SCOPE's chosen question

Usage:
  python scope_mediq_runner.py [--data_file PATH] [--output_filename PATH] [--max_questions N]
"""

import sys, os, json, re, glob, argparse, time
import torch
import numpy as np
import random

# ---------- paths ----------
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR  = os.path.dirname(SRC_DIR)
SCOPE_DIR = os.path.join(REPO_DIR, 'convo-plan-SCOPE')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, SCOPE_DIR)

# ---------- mediQ imports ----------
from expert import Expert
from patient import FactSelectPatient
import helper as mediq_helper        # we inject SCOPE's model into its cache below

# ---------- SCOPE imports ----------
from agent.Model import create_human_and_llm
from agent.Conversation import Conversation
from monte_carlo_tree_search.policy_agent import OnlineAgent
from monte_carlo_tree_search.qtable import DeepQSemanticFunction
from monte_carlo_tree_search.conversation_env import conversation_state
from transition_models.transition_model import TransitionModelMOE
from transition_models.embedding_model import embedding_model_llama
from reward.Llama_2_Guard_Reward import Llama_2_Guard_Reward


# ============================================================
#  CLI args
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str,
                        default=os.path.join(REPO_DIR, 'data/med_data/all_dev_convo.jsonl'))
    parser.add_argument('--output_filename', type=str,
                        default=os.path.join(SRC_DIR, 'results', 'scope_mediq.jsonl'))
    parser.add_argument('--max_questions', type=int, default=5)
    parser.add_argument('--max_examples', type=int, default=0,
                        help='Stop after this many examples (0 = all)')
    return parser.parse_args()


# ============================================================
#  Fix transition model paths
# ============================================================
def prepare_transition_models(cuda_q):
    det_dir = os.path.join(SCOPE_DIR, 'transition_models', 'deterministic')
    for val_pth in glob.glob(os.path.join(det_dir, '**', 'model_min_val.pth'), recursive=True):
        train_pth = val_pth.replace('model_min_val.pth', 'model_min_train.pth')
        if not os.path.exists(train_pth):
            os.symlink(val_pth, train_pth)
            print(f"[setup] symlinked model_min_val.pth -> model_min_train.pth")
    orig = os.getcwd()
    os.chdir(det_dir)
    model = TransitionModelMOE(noise=0.005, cuda=cuda_q, transition_model_dir='.')
    os.chdir(orig)
    return model


# ============================================================
#  GPU assignment
# ============================================================
CUDA_LLM = 0
CUDA_Q   = 1


# ============================================================
#  Initialize SCOPE components once
# ============================================================
config_path = os.path.join(SCOPE_DIR, 'agent', 'mediq_llm_config.yaml')

print("[init] Loading SCOPE LLM models…")
human_sim, human_eval, llm_agent = create_human_and_llm(config=config_path, cuda=CUDA_LLM)

print("[init] Loading Llama Guard reward + embedding model…")
reward_function = Llama_2_Guard_Reward(device_map=CUDA_Q)
embed_model     = embedding_model_llama(model=reward_function.model, cuda=torch.device(CUDA_Q))
dim             = embed_model.output_dim

print("[init] Loading transition model…")
# ── Simulator 1: SCOPE's transition model (inner MCTS planning) ──────────────
transition_model = prepare_transition_models(CUDA_Q)

print("[init] Building SCOPE OnlineAgent…")
semanticqfunction = DeepQSemanticFunction(
    dim=dim, alpha=0.0001, cuda=torch.device(CUDA_Q), steps_update=50
)
scope_agent = OnlineAgent(
    semanticqfunction,
    search_depth=8,
    mcts_time_limit=3,
    llm_agent=llm_agent,
    human_simulator=human_sim,          # SCOPE's inner human simulator
    reward_function_for_mcts=reward_function,
    search_space="semantic_space",
    transition_model=transition_model,  # SCOPE's learned transition model
    embedding_model=embed_model,
)

# ============================================================
#  Inject SCOPE's loaded HF model into mediQ's helper cache
#  so FactSelectPatient reuses it without loading a second copy.
#
#  ── Simulator 2: mediQ's FactSelectPatient (outer real patient) ────────────
# ============================================================
class _SCOPEModelAdapter:
    """Adapts SCOPE's Local_LLM to the interface expected by mediQ helper.py."""
    def __init__(self, local_llm):
        self._local_llm = local_llm
        self.use_vllm   = False
        self.use_api    = None
        self.args       = {"temperature": 0.7, "max_tokens": 256, "top_p": 0.9, "top_logprobs": 0}

    def generate(self, messages):
        tokenizer = self._local_llm.tokenizer
        hf_model  = self._local_llm.model
        max_new   = self.args.get("max_tokens", 256)

        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_attention_mask=True, return_dict=True,
        ).to(hf_model.device)

        with torch.no_grad():
            output = hf_model.generate(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new,
                do_sample=True,
                temperature=self.args["temperature"],
                top_p=self.args["top_p"],
                pad_token_id=tokenizer.eos_token_id,
            )
        generated     = output[:, tokens["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        usage = {
            "input_tokens":  int(tokens["input_ids"].shape[-1]),
            "output_tokens": int(generated.shape[-1]),
        }
        print(f"[LLM OUTPUT]: {response_text}\n")
        return response_text, None, usage


# Register in helper's model cache — FactSelectPatient will find it here
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
mediq_helper.models[MODEL_NAME] = _SCOPEModelAdapter(llm_agent.model)
print("[init] SCOPE model injected into mediQ helper cache — FactSelectPatient ready.\n")


# ============================================================
#  Expert: wraps SCOPE's OnlineAgent as a mediQ Expert
# ============================================================
class SCOPEExpert(Expert):

    def _build_conversation(self, patient_state):
        opts = ", ".join(f"{k}: {v}" for k, v in self.options.items())
        first_msg = (
            f"A patient presents with: {patient_state['initial_info']}\n\n"
            f"Clinical question: {self.inquiry}\n"
            f"Options: {opts}\n\n"
            "Either ask ONE focused follow-up question to the patient, "
            "or give your final answer as: FINAL ANSWER: [A/B/C/D]"
        )
        convo = Conversation(first_msg, start_with_human=True)
        for qa in patient_state['interaction_history']:
            convo = convo.add_llm_response(qa['question'], copy=False)
            convo = convo.add_human_response(qa['answer'],  copy=False)
        return convo

    def _parse_action(self, text):
        m = re.search(r'FINAL\s*ANSWER\s*[:\-]?\s*([ABCD])', text.upper())
        if m:
            return "choice", m.group(1)
        m = re.search(r'(?<![A-Z])([ABCD])(?![A-Z])\s*$', text.strip().upper())
        if m and len(text.strip()) < 30:
            return "choice", m.group(1)
        return "question", None

    def respond(self, patient_state):
        convo  = self._build_conversation(patient_state)
        last_r = convo.full_convo[-1] if convo.full_convo else ""
        state  = conversation_state(last_r, convo)
        state.depth = len(patient_state['interaction_history']) * 2 + 1

        scope_agent.qfunction.reset()
        results = {}
        action  = scope_agent.generate_action(state, results=results)
        print(f"\n  [SCOPE action]: {action!r}")

        action_type, letter = self._parse_action(action)
        if action_type == "choice":
            return {"type": "choice", "letter_choice": letter, "raw_action": action,
                    "confidence": 1.0, "usage": {"input_tokens": 0, "output_tokens": 0}}
        else:
            return {"type": "question", "question": action, "raw_action": action,
                    "letter_choice": list(self.options.keys())[0],
                    "confidence": 0.5, "usage": {"input_tokens": 0, "output_tokens": 0}}


# ============================================================
#  Interaction loop
# ============================================================
class RunArgs:
    use_vllm      = False   # FactSelectPatient will use the injected HF model
    use_api       = None
    temperature   = 0.7
    max_tokens    = 256
    top_p         = 0.9
    top_logprobs  = 0
    api_account   = "mediQ"
    patient_model = MODEL_NAME


def run_interaction(sample, max_questions):
    run_args = RunArgs()
    run_args.max_questions = max_questions

    expert  = SCOPEExpert(run_args, sample["question"], sample["options"])
    # ── Simulator 2: mediQ's FactSelectPatient ────────────────────────────────
    patient = FactSelectPatient(run_args, sample)

    temp_choices    = []
    temp_additional = []

    while len(patient.get_questions()) < max_questions:
        patient_state = patient.get_state()
        resp = expert.respond(patient_state)
        temp_additional.append({k: v for k, v in resp.items()
                                 if k not in ["type", "letter_choice", "question"]})

        if resp["type"] == "question":
            temp_choices.append(resp["letter_choice"])
            patient_answer = patient.respond(resp["question"])
            print(f"  [Patient]: {patient_answer[:200]}")

        elif resp["type"] == "choice":
            temp_choices.append(resp["letter_choice"])
            return resp["letter_choice"], patient.get_questions(), patient.get_answers(), \
                   temp_choices, temp_additional
        else:
            raise ValueError(f"Unknown response type: {resp['type']}")

    print(f"\n  [max questions ({max_questions}) reached — forcing final answer]")
    patient_state = patient.get_state()
    resp = expert.respond(patient_state)
    final = resp["letter_choice"]
    temp_choices.append(final)
    temp_additional.append({k: v for k, v in resp.items()
                             if k not in ["type", "letter_choice", "question"]})
    return final, patient.get_questions(), patient.get_answers(), temp_choices, temp_additional


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    args = get_args()

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    print(f"[run] data         : {args.data_file}")
    print(f"[run] output       : {args.output_filename}")
    print(f"[run] max_questions: {args.max_questions}")
    print(f"[run] inner sim    : SCOPE transition model (semantic space)")
    print(f"[run] outer sim    : mediQ FactSelectPatient\n")

    with open(args.data_file) as f:
        data = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)
    processed_ids = set()
    correct_history, timeout_history, turn_lengths = [], [], []
    if os.path.exists(args.output_filename):
        with open(args.output_filename) as f:
            for line in f:
                rec = json.loads(line)
                processed_ids.add(rec["id"])
                correct_history.append(rec["interactive_system"]["correct"])
                timeout_history.append(
                    len(rec["interactive_system"]["intermediate_choices"]) > args.max_questions)
                turn_lengths.append(rec["interactive_system"]["num_questions"])

    for sample in data:
        if args.max_examples and len(correct_history) >= args.max_examples:
            break
        pid = sample["id"]
        if pid in processed_ids:
            print(f"Skipping {pid} (already processed)")
            continue

        print(f"\n{'='*65}")
        print(f"Sample {pid} | GT: {sample['answer_idx']} ({sample['answer']})")
        print(f"{'='*65}")

        t0 = time.time()
        letter, questions, answers, choices, extra = run_interaction(sample, args.max_questions)
        elapsed = time.time() - t0

        correct = letter == sample["answer_idx"]
        correct_history.append(correct)
        timeout_history.append(len(choices) > args.max_questions)
        turn_lengths.append(len(questions))

        n = len(correct_history)
        accuracy     = sum(correct_history) / n
        timeout_rate = sum(timeout_history) / n
        avg_turns    = sum(turn_lengths)    / n
        print(f"\n  Predicted: {letter} | GT: {sample['answer_idx']} | Correct: {correct} | {elapsed:.1f}s")
        print(f"  Running — Acc: {accuracy:.3f}  Timeout: {timeout_rate:.3f}  "
              f"AvgTurns: {avg_turns:.1f}  ({n}/{len(data)})")

        output_dict = {
            "id": pid,
            "interactive_system": {
                "correct": correct,
                "letter_choice": letter,
                "questions": questions,
                "answers": answers,
                "num_questions": len(questions),
                "intermediate_choices": choices,
                "temp_additional_info": extra,
            },
            "info": {
                "initial_info": (sample["context"][0] if sample["context"] else "") if isinstance(sample["context"], list)
                                else sample["context"].split(". ")[0],
                "correct_answer": sample["answer"],
                "correct_answer_idx": sample["answer_idx"],
                "question": sample["question"],
                "options": sample["options"],
                "context": sample["context"],
                "facts": sample.get("atomic_facts"),
            }
        }
        with open(args.output_filename, "a") as f:
            f.write(json.dumps(output_dict) + "\n")

    n = len(correct_history)
    accuracy     = sum(correct_history) / n if n else 0
    timeout_rate = sum(timeout_history) / n if n else 0
    avg_turns    = sum(turn_lengths)    / n if n else 0
    print(f"\n{'='*65}")
    print(f"FINAL SUMMARY  ({len(data)} examples)")
    print(f"  Accuracy     : {sum(correct_history)} / {n} = {accuracy:.4f}")
    print(f"  Timeout Rate : {sum(timeout_history)} / {n} = {timeout_rate:.4f}")
    print(f"  Avg. Turns   : {avg_turns:.2f}")
    print(f"  Output       : {args.output_filename}")
    print(f"{'='*65}")
