#!/usr/bin/env python3
"""v65 — KoELECTRA embeddings 교체
ko-sbert (유사도용) → KoELECTRA-base-discriminator (분류용) 임베딩
KoELECTRA는 discriminative 학습으로 classification에 더 유리한 표현 가질 수 있음.

주의: 토크나이저도 KoELECTRA용으로 교체 필요
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, random, copy, re
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_CLASSES, HEAD_NAMES, HEAD_L2I, HEAD_I2L, HEAD_NC
from transformers import AutoModel, AutoTokenizer


class CNNMultiHeadKE(nn.Module):
    """Same CNN 5-head but with KoELECTRA embeddings (768d)"""
    def __init__(self, pretrained_emb_weights, d_model=256, max_len=32, dropout=0.15):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb_weights.shape
        # KoELECTRA vocab ≈ 32200 tokens, same 768d as ko-sbert
        self.token_emb = nn.Embedding.from_pretrained(pretrained_emb_weights, freeze=True, padding_idx=0)
        self.proj = nn.Linear(emb_dim, d_model)
        self.drop_in = nn.Dropout(dropout)

        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 5, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 7, padding=3),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv4 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, nc)
            )
            for name, nc in HEAD_NC.items()
        })
        self.max_len = max_len

    def forward(self, token_ids):
        seq_len = min(token_ids.shape[1], self.max_len)
        x = self.proj(self.token_emb(token_ids[:, :seq_len].long()))
        x = self.drop_in(x)
        x = x.permute(0, 2, 1)
        x = x + self.conv1(x)
        x = x + self.conv2(x)
        x = x + self.conv3(x)
        x = x + self.conv4(x)
        x = x.mean(dim=2)
        return {name: head(x) for name, head in self.heads.items()}


class MixupDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32, mixup_prob=0.3):
        self.data = data; self.tokenizer = tokenizer
        self.max_len = max_len; self.mixup_prob = mixup_prob
        self.fn_groups = {}
        for i, d in enumerate(data):
            self.fn_groups.setdefault(d['labels']['fn'], []).append(i)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        tk = self.tokenizer(d['utterance'], padding='max_length', truncation=True,
                           max_length=self.max_len, return_tensors='pt')
        input_ids = tk['input_ids'].squeeze(0)
        labels = {h: HEAD_L2I[h].get(d['labels'].get(h, 'none'), 0) for h in HEAD_NAMES}
        if random.random() < self.mixup_prob:
            fn = d['labels']['fn']; cands = self.fn_groups.get(fn, [])
            if len(cands) > 1:
                j = random.choice(cands)
                while j == idx and len(cands) > 1: j = random.choice(cands)
                tk2 = self.tokenizer(self.data[j]['utterance'], padding='max_length',
                                    truncation=True, max_length=self.max_len, return_tensors='pt')
                input_ids = tk2['input_ids'].squeeze(0)
        return input_ids, labels

def collate_fn(batch):
    return (torch.stack([b[0] for b in batch]),
            {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES})

def eval_test_suite(model, tok, device):
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad(): l = model(tk['input_ids'].to(device))
        p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1
    n = len(suite)
    return fn_ok/n*100, exec_ok/n*100, dir_ok/n*100, all_ok/n*100

def eval_koelectra(model, tok, device):
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    fn_ok = 0
    for d in ke_val:
        t = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad(): l = model(t['input_ids'].to(device))
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']: fn_ok += 1
    return fn_ok / len(ke_val) * 100


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('data/train_final_v43.json') as f: train_data = json.load(f)
    with open('data/val_final_v43.json') as f: val_data = json.load(f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load KoELECTRA tokenizer & embeddings
    print("Loading KoELECTRA...")
    ke_model = AutoModel.from_pretrained('monologg/koelectra-base-discriminator')
    ke_tok = AutoTokenizer.from_pretrained('monologg/koelectra-base-discriminator')
    pw = ke_model.embeddings.word_embeddings.weight.detach()
    del ke_model
    print(f"KoELECTRA emb: {pw.shape}")

    model = CNNMultiHeadKE(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    train_ds = MixupDataset(train_data, ke_tok, mixup_prob=0.3)
    val_ds = MixupDataset(val_data, ke_tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    best_balanced = 0
    best_state = None

    for epoch in range(35):
        model.train()
        train_loss = 0; train_n = 0
        for input_ids, labels in train_dl:
            input_ids = input_ids.to(device)
            labels = {h: v.to(device) for h, v in labels.items()}
            logits = model(input_ids)
            loss = 0
            loss += 2.0 * F.cross_entropy(logits['fn'], labels['fn'], weight=fn_weights)
            loss += 2.0 * F.cross_entropy(logits['exec_type'], labels['exec_type'])
            loss += 1.5 * F.cross_entropy(logits['param_direction'], labels['param_direction'])
            loss += 1.0 * F.cross_entropy(logits['param_type'], labels['param_type'])
            loss += 1.5 * F.cross_entropy(logits['judge'], labels['judge'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)
        scheduler.step()

        model.eval()
        head_correct = {h: 0 for h in HEAD_NAMES}
        all_correct = 0; total = 0
        with torch.no_grad():
            for input_ids, labels in val_dl:
                input_ids = input_ids.to(device)
                labels = {h: v.to(device) for h, v in labels.items()}
                logits = model(input_ids)
                B = input_ids.size(0)
                all_match = torch.ones(B, dtype=torch.bool, device=device)
                for h in HEAD_NAMES:
                    pred = logits[h].argmax(1)
                    correct = (pred == labels[h])
                    head_correct[h] += correct.sum().item()
                    all_match &= correct
                all_correct += all_match.sum().item()
                total += B

        fn_acc = head_correct['fn'] / total * 100
        combo_acc = all_correct / total * 100

        ts_str = ""
        if (epoch + 1) % 5 == 0 or epoch >= 30:
            ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, ke_tok, device)
            ke_fn = eval_koelectra(model, ke_tok, device)
            balanced = (ts_combo * ke_fn) ** 0.5
            ts_str = f" | TS={ts_combo:.1f}% KE={ke_fn:.1f}% bal={balanced:.1f}"
            if balanced > best_balanced:
                best_balanced = balanced
                best_state = copy.deepcopy(model.state_dict())
                torch.save({'epoch': epoch+1, 'state': model.state_dict(),
                           'ts_combo': ts_combo, 'ke_fn': ke_fn, 'balanced': balanced,
                           'emb_source': 'koelectra-base-discriminator'},
                          'checkpoints/cnn_multihead_v65.pt')
                ts_str += " ★"

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} combo={combo_acc:.1f}%{ts_str}")

    print(f"\n=== v65 KoELECTRA emb Final: balanced {best_balanced:.1f} ===")


if __name__ == '__main__':
    random.seed(42); torch.manual_seed(42)
    train()
