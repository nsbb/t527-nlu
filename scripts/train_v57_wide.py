#!/usr/bin/env python3
"""v57 — Wider model (d=384) + sample-weighted GT emphasis + mixup
Hypothesis: more capacity helps learn both GT patterns and general patterns.
GT samples weighted 2x vs KoELECTRA for exec/dir accuracy.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, random, copy
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


class WiderCNNMultiHead(nn.Module):
    """Same architecture as CNNMultiHead but with configurable d_model"""
    def __init__(self, pretrained_emb_weights, d_model=384, max_len=32, dropout=0.1):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb_weights.shape
        self.token_emb = nn.Embedding.from_pretrained(pretrained_emb_weights, freeze=True, padding_idx=0)
        self.proj = nn.Linear(emb_dim, d_model)
        self.drop_in = nn.Dropout(dropout)

        self.conv1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))
        self.conv4 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout))

        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
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


class WeightedMixupDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32, mixup_prob=0.3):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mixup_prob = mixup_prob
        self.fn_groups = {}
        for i, d in enumerate(data):
            fn = d['labels']['fn']
            self.fn_groups.setdefault(fn, []).append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        tk = self.tokenizer(d['utterance'], padding='max_length', truncation=True,
                           max_length=self.max_len, return_tensors='pt')
        input_ids = tk['input_ids'].squeeze(0)
        labels = {h: HEAD_L2I[h].get(d['labels'].get(h, 'none'), 0) for h in HEAD_NAMES}

        # Sample weight: GT data gets higher weight
        source = d.get('source', '')
        weight = 2.0 if source.startswith('gt_') or source.startswith('fix_') else 1.0

        if random.random() < self.mixup_prob:
            fn = d['labels']['fn']
            candidates = self.fn_groups.get(fn, [])
            if len(candidates) > 1:
                j = random.choice(candidates)
                while j == idx and len(candidates) > 1:
                    j = random.choice(candidates)
                d2 = self.data[j]
                tk2 = self.tokenizer(d2['utterance'], padding='max_length', truncation=True,
                                    max_length=self.max_len, return_tensors='pt')
                input_ids = tk2['input_ids'].squeeze(0)

        return input_ids, labels, weight


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    weights = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return input_ids, labels, weights


def eval_model(model, val_dl, device):
    model.eval()
    head_correct = {h: 0 for h in HEAD_NAMES}
    all_correct = 0; total = 0
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch[0].to(device)
            labels = {h: v.to(device) for h, v in batch[1].items()}
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
    return {h: head_correct[h] / total * 100 for h in HEAD_NAMES}, all_correct / total * 100


def eval_test_suite(model, tok, device):
    """Run test suite evaluation"""
    import re
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            l = model(tk['input_ids'].to(device))
        p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
        if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
        if p['judge'] != 'none': p['param_type'] = 'none'
        if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'

        f = p['fn'] == t['fn']
        e = p['exec_type'] == t['exec']
        d = p['param_direction'] == t['dir']
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
        with torch.no_grad():
            l = model(t['input_ids'].to(device))
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']:
            fn_ok += 1
    return fn_ok / len(ke_val) * 100


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('data/train_final_v43.json') as f: train_data = json.load(f)
    with open('data/val_final_v43.json') as f: val_data = json.load(f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    # Wider model
    model = WiderCNNMultiHead(pw, d_model=384, max_len=32, dropout=0.15).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable (d=384)")

    train_ds = WeightedMixupDataset(train_data, tok, mixup_prob=0.3)
    val_ds = WeightedMixupDataset(val_data, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_combo = 0; best_fn = 0

    for epoch in range(40):
        model.train()
        train_loss = 0; train_n = 0

        for input_ids, labels, weights in train_dl:
            input_ids = input_ids.to(device)
            labels = {h: v.to(device) for h, v in labels.items()}
            weights = weights.to(device)

            logits = model(input_ids)

            # Sample-weighted loss
            loss = 0
            for h in HEAD_NAMES:
                hw = {'fn': 2.0, 'exec_type': 2.0, 'param_direction': 1.5,
                      'param_type': 1.0, 'judge': 1.5}[h]
                if h == 'fn':
                    per_sample = F.cross_entropy(logits[h], labels[h], weight=fn_weights, reduction='none')
                else:
                    per_sample = F.cross_entropy(logits[h], labels[h], reduction='none')
                loss += hw * (per_sample * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)

        scheduler.step()

        head_accs, combo_acc = eval_model(model, val_dl, device)
        fn_acc = head_accs['fn']

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} "
              f"exec={head_accs['exec_type']:.1f} dir={head_accs['param_direction']:.1f} | combo={combo_acc:.1f}%")

        if combo_acc > best_combo or (combo_acc == best_combo and fn_acc > best_fn):
            best_combo = combo_acc; best_fn = fn_acc
            torch.save({
                'epoch': epoch + 1, 'state': model.state_dict(),
                'combo': combo_acc, 'fn': fn_acc,
                'exec': head_accs['exec_type'], 'dir': head_accs['param_direction'],
                'd_model': 384,
            }, 'checkpoints/cnn_multihead_v57.pt')
            print(f"  ★ Best: combo={combo_acc:.1f}%, fn={fn_acc:.1f}%")

    # Final eval on TS + KE
    ckpt = torch.load('checkpoints/cnn_multihead_v57.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state']); model.eval()

    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)

    print(f"\n=== v57 (d=384) Final ===")
    print(f"  Val combo: {best_combo:.1f}%, fn: {best_fn:.1f}%")
    print(f"  Test Suite: fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}%")
    print(f"  KoELECTRA fn: {ke_fn:.1f}%")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train()
