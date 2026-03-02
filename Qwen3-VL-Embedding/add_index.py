import json
import hashlib
import argparse
import os
import logging
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IndexMapper")

def build_hash_to_index_mapping(dataset):
    """
    遍历原数据集，按照与之前完全相同的逻辑计算 hash_id，
    并构建 {hash_id: original_index} 的映射字典。
    """
    mapping = {}
    logger.info("正在遍历数据集，构建 Hash 到 Index 的映射字典...")
    
    for idx, row in enumerate(tqdm(dataset, desc="构建映射字典")):
        if "conversations" not in row or "chosen" not in row or "rejected" not in row:
            continue
            
        history = [turn["value"] for turn in row.get("conversations", [])]
        chosen_dialogue = history + [row["chosen"]["value"]]
        
        # 与 VLLM_embedding.py 中完全一致的哈希计算逻辑
        hash_string = "".join(chosen_dialogue).encode('utf-8')
        hash_id = hashlib.md5(hash_string).hexdigest()
        
        # 使用 setdefault 确保如果遇到一模一样的重复对话，保留第一次出现的 index
        mapping.setdefault(hash_id, idx)
        
    logger.info(f"映射字典构建完成！共收集到 {len(mapping)} 个唯一的对话 Hash。")
    return mapping

def main():
    parser = argparse.ArgumentParser(description="为旧版 embeddings.jsonl 补齐 original_index")
    parser.add_argument('-hf', type=str, required=True, help='Hugging Face 数据集名称 (e.g., "microsoft/WildFeedback")')
    parser.add_argument('--subset', type=str, default=None, help='数据集 subset/config 名称 (e.g., "wildfeedback")')
    parser.add_argument('--split', type=str, default='train', help='数据集 split (默认: train)')
    parser.add_argument('--input', type=str, default='./results/embeddings.jsonl', help='原始输入的 jsonl 文件路径')
    parser.add_argument('--output', type=str, default='./results/embeddings_with_index.jsonl', help='输出的完整 jsonl 文件路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        logger.error(f"找不到输入文件: {args.input}")
        return

    # 1. 加载数据集
    logger.info(f"正在加载数据集: {args.hf} (subset: {args.subset}, split: {args.split})...")
    load_kwargs = {"path": args.hf, "split": args.split}
    if args.subset:
        load_kwargs["name"] = args.subset
        
    dataset = load_dataset(**load_kwargs)
    
    # 2. 构建映射
    hash_to_idx = build_hash_to_index_mapping(dataset)
    
    # 3. 读取旧文件并写入新文件
    logger.info(f"正在读取 {args.input} 并补齐 index...")
    
    success_count = 0
    missing_count = 0
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="处理 Embeddings"):
            if not line.strip():
                continue
                
            data = json.loads(line)
            hash_id = data.get("hash_id")
            
            # 如果原数据已经包含 original_index（比如断点重续后跑的新数据），就不覆盖
            if "original_index" not in data:
                if hash_id in hash_to_idx:
                    # 按照顺序，把 original_index 插入到字典的最前面（仅为美观）
                    new_data = {"original_index": hash_to_idx[hash_id]}
                    new_data.update(data)
                    data = new_data
                    success_count += 1
                else:
                    missing_count += 1
                    logger.debug(f"警告: 找不到哈希值 {hash_id} 对应的原始 index！")
            
            # 写入新文件
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            
    logger.info(f"处理完毕！成功补齐了 {success_count} 条记录的 index。")
    if missing_count > 0:
        logger.warning(f"有 {missing_count} 条记录未能找到匹配的 index。")
    logger.info(f"完整数据已保存至: {args.output}")

if __name__ == "__main__":
    main()