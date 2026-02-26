import os
import argparse
import logging
import json
import torch
import hashlib
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from vllm import LLM

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("DatasetLoader")


class DatasetManager:
    """
    Manager class responsible for checking local cache, downloading, streaming, and saving datasets.
    First we try to read the dataset, if dataset do not exist, we'll download it.
    Try using -head X to read first X rows.
    """
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def get_local_path(self, dataset_name: str, subset: str = None, head_n: int = None, split: str = "train") -> str:
        """
        Generate the local storage path. To avoid conflicts, head_n is appended to the directory name if provided.
        """
        safe_name = dataset_name.replace("/", "_")
        parts = [safe_name]
        if subset:
            parts.append(subset)
        parts.append(split)
        if head_n is not None:
            parts.append(f"head_{head_n}")
        return os.path.join(self.data_dir, "_".join(parts))

    def load_or_download(self, dataset_name: str, subset: str = None, head_n: int = None, split: str = "train"):
        """
        Check if local cache exists, otherwise download or stream the data.
        """
        local_path = self.get_local_path(dataset_name, subset, head_n, split)

        if os.path.exists(local_path):
            logger.info(f"Local cache found. Loading from {local_path}...")
            dataset = load_from_disk(local_path)
            return dataset

        logger.info(f"Local cache not found. Fetching from Hugging Face: {dataset_name}" + (f" (subset: {subset})" if subset else ""))
        
        # Prepare kwargs for load_dataset
        load_kwargs = {"path": dataset_name, "split": split}
        if subset:
            load_kwargs["name"] = subset
            
        if head_n is not None:
            logger.info(f"Streaming mode enabled. Fetching the first {head_n} rows...")
            load_kwargs["streaming"] = True
            iterable_ds = load_dataset(**load_kwargs)
            
            head_data = list(iterable_ds.take(head_n))
            dataset = Dataset.from_list(head_data)
        else:
            logger.info("Downloading full dataset...")
            dataset = load_dataset(**load_kwargs)

        logger.info(f"Saving dataset to local disk: {local_path}")
        dataset.save_to_disk(local_path)
        
        return dataset


def get_detailed_instruct(task_description: str, query: str) -> str:
    """
    Helper function to format instruction and query for embedding models.
    """
    return f'Instruct: {task_description}\nQuery:{query}'


def process_preference_data(example):
    """
    Processes preference dataset row to create chosen and rejected dialogue lists.
    Extracts the 'value' field from the 'conversations', 'chosen', and 'rejected' columns.
    """
    # History of conversation
    history = [turn["value"] for turn in example.get("conversations", [])]
    
    # Concat chosen and rejected
    chosen_dialogue = history + [example["chosen"]["value"]]
    rejected_dialogue = history + [example["rejected"]["value"]]
    
    return {
        "chosen_dialogue": chosen_dialogue,
        "rejected_dialogue": rejected_dialogue
    }


def format_prompt(history_list, answer):
    """
    Formats the conversation history and answer into the specific embedding prompt.
    """
    prompt = "Instruct: Encode the conversation context and the final model response to capture its semantic meaning, dialogue coherence, and relationship with previous turns.\nHistory:\n"
    for i, text in enumerate(history_list):
        role = "human" if i % 2 == 0 else "model"
        prompt += f"{role}: {text}\n"
    prompt += f"Answer:\nmodel: {answer}"
    return prompt


def build_truncated_prompt(history_list, answer, tokenizer, max_length):
    """
    Iteratively truncates the earliest conversation pairs if the prompt exceeds max_length.
    Returns "exceed_length" if the final remaining pair is still too long.
    """
    current_history = history_list.copy()
    
    while True:
        prompt_text = format_prompt(current_history, answer)
        tokens = tokenizer.encode(prompt_text)
        
        if len(tokens) <= max_length:
            return prompt_text
        
        # If it exceeds max_length, attempt to drop the earliest pair (human + model = 2 items)
        if len(current_history) > 1:
            current_history = current_history[2:]
        else:
            # Even the most recent human query + answer exceeds max_length
            return "exceed_length"


def process_embeddings_in_batches(dataset, model: LLM, max_length: int, batch_size: int = 100):
    """
    Generates VLLM embeddings for the dataset in batches, saving incrementally.
    """
    tokenizer = model.get_tokenizer()
    os.makedirs("./results", exist_ok=True)
    results_file = "./results/embeddings.jsonl"
    
    current_batch_prompts = []
    current_batch_metadata = []
    
    logger.info(f"Starting embedding process. Saving to {results_file} incrementally every {batch_size} conversations...")
    
    for idx, row in enumerate(dataset):
        if "chosen_dialogue" not in row or "rejected_dialogue" not in row:
            continue
            
        chosen_dialogue = row["chosen_dialogue"]
        rejected_dialogue = row["rejected_dialogue"]
        
        # Hash
        hash_string = "".join(chosen_dialogue).encode('utf-8')
        hash_id = hashlib.md5(hash_string).hexdigest()
        
        num_turns = len(chosen_dialogue) // 2
        
        # Prompt formatting and chuncking
        for turn_idx in range(num_turns):
            if turn_idx < num_turns - 1:
                history = chosen_dialogue[:turn_idx*2 + 1]
                answer = chosen_dialogue[turn_idx*2 + 1]
                prompt = build_truncated_prompt(history, answer, tokenizer, max_length)
                
                current_batch_metadata.append({"hash_id": hash_id, "turn_index": turn_idx, "type": "history"})
                current_batch_prompts.append(prompt)
            else:
                history_chosen = chosen_dialogue[:-1]
                answer_chosen = chosen_dialogue[-1]
                prompt_chosen = build_truncated_prompt(history_chosen, answer_chosen, tokenizer, max_length)
                current_batch_metadata.append({"hash_id": hash_id, "turn_index": turn_idx, "type": "chosen"})
                current_batch_prompts.append(prompt_chosen)
                
                history_rejected = rejected_dialogue[:-1]
                answer_rejected = rejected_dialogue[-1]
                prompt_rejected = build_truncated_prompt(history_rejected, answer_rejected, tokenizer, max_length)
                current_batch_metadata.append({"hash_id": hash_id, "turn_index": turn_idx, "type": "rejected"})
                current_batch_prompts.append(prompt_rejected)

        # Write into file for every batch size
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(dataset):
            save_batch(model, current_batch_prompts, current_batch_metadata, results_file)
            current_batch_prompts = []
            current_batch_metadata = []
            
    logger.info("Processed and saved!")


def save_batch(model: LLM, prompts: list, metadata: list, results_file: str):
    """
    Executes VLLM embed logic for valid prompts and appends the results to a JSONL file.
    """
    if not prompts: return
    
    valid_indices = [i for i, p in enumerate(prompts) if p != "exceed_length"]
    valid_prompts = [prompts[i] for i in valid_indices]
    
    if valid_prompts:
        outputs = model.embed(valid_prompts)
        for valid_i, output in zip(valid_indices, outputs):
            metadata[valid_i]["embedding"] = output.outputs.embedding
            
    exceed_indices = [i for i, p in enumerate(prompts) if p == "exceed_length"]
    for exceed_i in exceed_indices:
        metadata[exceed_i]["embedding"] = "exceed_length"
        
    with open(results_file, 'a', encoding='utf-8') as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
            
    logger.info(f"Appended {len(metadata)} embedding records to {results_file}.")


def load_vllm_model(model_name_or_path: str, q_config_path: str = None, gpu_util: float = 0.9, max_length: int = 2048) -> LLM:
    """
    Validates the model path, loads quantization config if provided, 
    and initializes the VLLM model for embedding tasks with proper max_model_len.
    """
    vllm_kwargs = {}

    # Check if model is a local path or a Hugging Face Hub name
    if os.path.exists(model_name_or_path):
        logger.info(f"Local model path found: {model_name_or_path}")
    else:
        if model_name_or_path.startswith((".", "/", "\\")) or os.sep in model_name_or_path and not "/" in model_name_or_path.strip("/"):
             logger.error(f"Model path does not exist: {model_name_or_path}")
             raise FileNotFoundError(f"Model path does not exist: {model_name_or_path}")
        else:
             logger.info(f"Can't find model locally. Downloaded: '{model_name_or_path}'.")

    # Load JSON quantization config if provided
    if q_config_path:
        if not os.path.exists(q_config_path):
            raise FileNotFoundError(f"Quantization config file not found: {q_config_path}")
        with open(q_config_path, 'r', encoding='utf-8') as f:
            vllm_kwargs = json.load(f)
        logger.info(f"Loaded quantization config from {q_config_path}: {vllm_kwargs}")

    logger.info(f"Initializing VLLM with model: {model_name_or_path} (GPU Util: {gpu_util}, Max Model Len: {max_length})")
    # Unpack the quantization configuration and pass max_model_len explicitly to prevent VRAM explosion
    model = LLM(
        model=model_name_or_path, 
        task="embed", 
        gpu_memory_utilization=gpu_util, 
        max_model_len=max_length, 
        **vllm_kwargs
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Face Dataset & VLLM Model Loader (GitHub Ready)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    '''
    Dataset Arguments
    -hf: Name of huggingface dataset, e.g., "microsoft/WildFeedback"
    -head: Explore top N rows.
    --split: The specific split.
    '''
    parser.add_argument('-hf', dest='hf', type=str, required=True, 
                        help='Hugging Face dataset name (e.g., "microsoft/WildFeedback")')
    parser.add_argument('--subset', dest='subset', type=str, default=None, 
                        help='Dataset subset/config name (e.g., "wildfeedback")')
    parser.add_argument('-head', dest='head', type=int, default=None, 
                        help='Limit to the first N rows (uses streaming to avoid full download)')
    parser.add_argument('--split', type=str, default='train', 
                        help='Dataset split to load')

    '''
    Model Arguments
    -model: Name of the huggingface model or local path
    -q_config: Path to the JSON configuration file for quantization (optional)
    '''
    parser.add_argument('-model', dest='model', type=str, required=True,
                        help='Hugging Face model name or local path (e.g., "Qwen/Qwen3-Embedding-8B")')
    parser.add_argument('-q_config', dest='q_config', type=str, default=None,
                        help='Path to a JSON file containing VLLM quantization config')
    parser.add_argument('-max_length', dest='max_length', type=int, default=2048,
                        help='Maximum token length per prompt; also passed to VLLM as max_model_len to save KV cache.')
    parser.add_argument('-gpu_util', dest='gpu_util', type=float, default=0.9,
                        help='Fraction of GPU memory to use for VLLM (default: 0.9, e.g., use 0.8 if memory is tight)')

    args = parser.parse_args()

    # 1. Dataset Loading Process
    manager = DatasetManager(data_dir="./data")
    
    try:
        dataset = manager.load_or_download(
            dataset_name=args.hf, 
            subset=args.subset,
            head_n=args.head, 
            split=args.split
        )
        logger.info("Dataset loaded successfully!")
        
        logger.info("Processing data...")
        
        sample_item = dataset[list(dataset.keys())[0]][0] if isinstance(dataset, DatasetDict) else dataset[0]
        
        if all(col in sample_item for col in ["conversations", "chosen", "rejected"]):

            dataset = dataset.map(process_preference_data, desc="Formatting preference dialogues")
            
            sample_to_print = dataset[list(dataset.keys())[0]][0] if isinstance(dataset, DatasetDict) else dataset[0]
            logger.info(f"Sample Chosen Dialogue List:\n{sample_to_print['chosen_dialogue']}")
            logger.info(f"Sample Rejected Dialogue List:\n{sample_to_print['rejected_dialogue']}")
            
            if isinstance(dataset, DatasetDict):
                dataset = dataset[args.split] if args.split in dataset else dataset[list(dataset.keys())[0]]
        else:
            logger.warning("Dataset does not contain 'conversations', 'chosen', and 'rejected' columns. Skipping preference processing.")
            
    except Exception as e:
        logger.error(f"Error occurred while loading dataset: {str(e)}")

    # 2. Model Loading & Embedding
    try:
        model = load_vllm_model(
            model_name_or_path=args.model,
            q_config_path=args.q_config,
            gpu_util=args.gpu_util,
            max_length=args.max_length # Explicitly pass it here
        )
        
        # 3. Process the dataset through embedding logic incrementally
        if "chosen_dialogue" in dataset.features:
            process_embeddings_in_batches(
                dataset=dataset, 
                model=model, 
                max_length=args.max_length, 
                batch_size=100
            )
        else:
            logger.warning("No preference dialogue data found to process. Exiting embedding stage.")

    except Exception as e:
        logger.error(f"Error occurred while loading model or computing embeddings: {str(e)}")


if __name__ == "__main__":
    main()