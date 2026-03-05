import json
import os
import torch
import torch.nn as nn
import argparse
from collections import defaultdict
from mamba_ssm import Mamba2

class Mamba2RewardModel(nn.Module):
    def __init__(self, input_dim, d_model=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mixer': Mamba2(d_model=d_model),
                'norm': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, src, lengths):
        x = self.input_proj(src)
        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x = layer['mixer'](x)
            x = x + residual
            
        batch_size = x.size(0)
        last_indices = lengths - 1 
        last_token_output = x[torch.arange(batch_size), last_indices, :]
        
        score = self.score_head(last_token_output)
        return score

def load_and_process_test_data(test_path):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试文件: {test_path}")
    
    
    groups = defaultdict(list)
    invalid_ids = set()
    
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
                    
    for item in data:
        idx = item['id']
        turn_id = item['TurnId']
        emb = item['embedding']
        
        if isinstance(emb, str):
            invalid_ids.add(idx)
            continue
            
        groups[idx].append({'turn_id': turn_id, 'embedding': emb})
        
    test_samples = []
    discarded_count = 0
    
    for idx, turns in groups.items():
        if idx in invalid_ids:
            discarded_count += 1
            continue
            
        turns.sort(key=lambda x: x['turn_id'])
        
        current_sequence = []
        for t in turns:
            current_sequence.append(t['embedding'])
            
            test_samples.append({
                'id': idx,
                'TurnId': t['turn_id'],
                'sequence': torch.tensor(current_sequence, dtype=torch.float32)
            })
            
    print(f"数据处理完成: 剔除了 {discarded_count} 个包含异常 Embedding 的 ID。")
    print(f"生成 {len(test_samples)} 条测试序列样本。")
    return test_samples

def run_inference(test_samples, model_path, output_dir, embed_dim=2560, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Mamba2RewardModel(input_dim=embed_dim, d_model=512, num_layers=2, dropout=0.0)
    print(f"Loading weights: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "inference_results.jsonl")
    
    print(f"Inferencing...")
    results = []
    
    with torch.no_grad():
        for i in range(0, len(test_samples), batch_size):
            batch = test_samples[i:i + batch_size]
            
            seqs = [item['sequence'] for item in batch]
            lengths = torch.tensor([len(s) for s in seqs])
            padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0).to(device)
            lengths = lengths.to(device)
            
            scores = model(padded_seqs, lengths)
            scores = scores.squeeze(-1).cpu().tolist()
            
            if isinstance(scores, float):
                scores = [scores]
            
            for item, score in zip(batch, scores):
                results.append({
                    "id": item['id'],
                    "TurnId": item['TurnId'],
                    "score": score
                })
                
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba-2 Reward Model Inference Script")
    
    parser.add_argument("--test_path", type=str, required=True, help="测试数据文件路径 (例如 ./data/embeddings.json)")
    parser.add_argument("--model", type=str, required=True, help="训练好的 Mamba-2 模型权重路径 (.pth)")
    parser.add_argument("--output_dir", type=str, required=True, help="推理结果的保存目录")
    
    parser.add_argument("--embed_dim", type=int, default=2560, help="Embedding 的维度 (默认 2560)")
    parser.add_argument("--batch_size", type=int, default=32, help="推理时的 Batch Size (默认 32)")
    
    args = parser.parse_args()
    
    samples = load_and_process_test_data(args.test_path)
    
    run_inference(
        test_samples=samples,
        model_path=args.model,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size
    )