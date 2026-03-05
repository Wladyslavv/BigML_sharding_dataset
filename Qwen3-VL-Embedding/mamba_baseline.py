import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict
import math
import os
import argparse
from mamba_ssm import Mamba2

class DialoguePreferenceDataset(Dataset):
    def __init__(self, file_path):
        self.data = self._process_data(file_path)
        
    def _process_data(self, file_path):
        groups = defaultdict(lambda: {'history': [], 'chosen': None, 'rejected': None})
        invalid_indices = set() 
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                idx = item['original_index']
                emb = item['embedding']
                
                if isinstance(emb, str):
                    invalid_indices.add(idx)
                    continue 
                
                if item['type'] == 'history':
                    groups[idx]['history'].append((item['turn_index'], emb))
                elif item['type'] == 'chosen':
                    groups[idx]['chosen'] = emb
                elif item['type'] == 'rejected':
                    groups[idx]['rejected'] = emb
                    
        processed_data = []
        total_groups = len(groups)
        discarded_string_count = 0
        discarded_missing_count = 0
        neutral_count = 0
        
        for idx, group in groups.items():
            if idx in invalid_indices:
                discarded_string_count += 1
                continue
                
            if group['chosen'] is None or group['rejected'] is None:
                discarded_missing_count += 1
                continue
                
            sorted_history = sorted(group['history'], key=lambda x: x[0])
            history_embs = [emb for _, emb in sorted_history]
            
            seq_chosen = torch.tensor(history_embs + [group['chosen']], dtype=torch.float32)
            seq_rejected = torch.tensor(history_embs + [group['rejected']], dtype=torch.float32)
            
            # 【修改1】：挑选 history 长度至少为 3 的 embeddings 作为 neutral 序列
            if len(history_embs) >= 3:
                seq_neutral = torch.tensor(history_embs, dtype=torch.float32)
                neutral_count += 1
            else:
                seq_neutral = None
            
            processed_data.append({
                'chosen_seq': seq_chosen,
                'rejected_seq': seq_rejected,
                'neutral_seq': seq_neutral
            })
            
        valid_groups = len(processed_data)
        
        print("=" * 40)
        print("Statistics:")
        print(f"Total original_index: {total_groups}")
        print(f"discarded_string_count: {discarded_string_count}")
        print(f"discarded_missing_count: {discarded_missing_count}")
        print(f"valid_groups: {valid_groups}")
        print(f"neutral_sequences_extracted: {neutral_count}")
        print("=" * 40)
        
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    chosen_seqs = [item['chosen_seq'] for item in batch]
    rejected_seqs = [item['rejected_seq'] for item in batch]
    neutral_seqs = [item['neutral_seq'] for item in batch if item['neutral_seq'] is not None]
    
    chosen_lengths = torch.tensor([len(seq) for seq in chosen_seqs])
    rejected_lengths = torch.tensor([len(seq) for seq in rejected_seqs])
    
    chosen_padded = torch.nn.utils.rnn.pad_sequence(chosen_seqs, batch_first=True, padding_value=0.0)
    rejected_padded = torch.nn.utils.rnn.pad_sequence(rejected_seqs, batch_first=True, padding_value=0.0)
    
    if len(neutral_seqs) > 0:
        neutral_lengths = torch.tensor([len(seq) for seq in neutral_seqs])
        neutral_padded = torch.nn.utils.rnn.pad_sequence(neutral_seqs, batch_first=True, padding_value=0.0)
    else:
        neutral_lengths = None
        neutral_padded = None
    
    return chosen_padded, chosen_lengths, rejected_padded, rejected_lengths, neutral_padded, neutral_lengths


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


def train_and_evaluate_mamba(file_path, embed_dim, epochs=15, batch_size=32, lr=1e-4, beta=0.1, output_dir='./output'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | Beta (MSE Weight): {beta} | Output Dir: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    full_dataset = DialoguePreferenceDataset(file_path)
    total_size = len(full_dataset)
    train_size = int(0.9 * total_size)
    test_size = total_size - train_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = Mamba2RewardModel(input_dim=embed_dim, d_model=512, num_layers=2, dropout=0.2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_test_loss = float('inf') 
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch in train_loader:
            batch_to_device = [b.to(device) if b is not None else None for b in batch]
            chosen_padded, chosen_lengths, rejected_padded, rejected_lengths, neutral_padded, neutral_lengths = batch_to_device
            
            optimizer.zero_grad()
            
            score_chosen = model(chosen_padded, chosen_lengths)
            score_rejected = model(rejected_padded, rejected_lengths)
            pairwise_loss = -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()
            
            pointwise_loss = torch.tensor(0.0, device=device)
            
            if neutral_padded is not None:
                num_neutral = neutral_padded.size(0)
                num_select = max(1, int(num_neutral * 0.5)) 
                
                rand_indices = torch.randperm(num_neutral, device=device)[:num_select]
                
                selected_neutral_padded = neutral_padded[rand_indices]
                selected_neutral_lengths = neutral_lengths[rand_indices]
                
                score_neutral = model(selected_neutral_padded, selected_neutral_lengths)
                
                target_val = torch.rand(1, device=device).expand_as(score_neutral)
                
                pointwise_loss = F.mse_loss(score_neutral, target_val)
            
            loss = pairwise_loss + beta * pointwise_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(chosen_padded)
            train_correct += (score_chosen > score_rejected).sum().item()
            
        avg_train_loss = train_loss / train_size
        train_acc = train_correct / train_size
        
        model.eval()
        test_loss = 0.0
        test_correct = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_to_device = [b.to(device) if b is not None else None for b in batch]
                chosen_padded, chosen_lengths, rejected_padded, rejected_lengths, neutral_padded, neutral_lengths = batch_to_device
                
                score_chosen = model(chosen_padded, chosen_lengths)
                score_rejected = model(rejected_padded, rejected_lengths)
                
                pairwise_loss = -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()
                
                pointwise_loss = torch.tensor(0.0, device=device)
                if neutral_padded is not None:
                    num_neutral = neutral_padded.size(0)
                    num_select = max(1, int(num_neutral * 0.5))
                    rand_indices = torch.randperm(num_neutral, device=device)[:num_select]
                    
                    selected_neutral_padded = neutral_padded[rand_indices]
                    selected_neutral_lengths = neutral_lengths[rand_indices]
                    score_neutral = model(selected_neutral_padded, selected_neutral_lengths)
                    
                    target_val = torch.rand(1, device=device).expand_as(score_neutral)
                    pointwise_loss = F.mse_loss(score_neutral, target_val)
                    
                loss = pairwise_loss + beta * pointwise_loss
                
                test_loss += loss.item() * len(chosen_padded)
                test_correct += (score_chosen > score_rejected).sum().item()
                
        avg_test_loss = test_loss / test_size
        test_acc = test_correct / test_size
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            save_path = os.path.join(output_dir, "best_mamba2_reward_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Updated {save_path}！")
            
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mamba-2 Reward Model with Hybrid Loss")
    
    parser.add_argument("--data_file", type=str, default="./results/embeddings_with_index.jsonl", help="训练数据 JSONL 文件的路径")
    parser.add_argument("--embed_dim", type=int, default=2560, help="Embedding 的维度")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size 大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--beta", type=float, default=0.1, help="MSE Loss (Pointwise) 的权重超参数 β")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="模型保存的输出目录路径")
    
    args = parser.parse_args()
    
    print("Training")
    trained_model = train_and_evaluate_mamba(
        file_path=args.data_file, 
        embed_dim=args.embed_dim, 
        epochs=args.epochs,         
        batch_size=args.batch_size,     
        lr=args.lr,
        beta=args.beta, 
        output_dir=args.output_dir 
    )