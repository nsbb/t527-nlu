#!/usr/bin/env python3
"""v63 — Conformer backbone with CNN's 5-head structure
Fair architecture comparison: same data, same heads, different backbone.
Conformer 2L (d=256, 4-head attention + depthwise conv) vs CNN 4L.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, random, copy, re
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_CLASSES, HEAD_NAMES, HEAD_L2I, HEAD_I2L, HEAD_NC
from transformers import AutoModel, AutoTokenizer


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, ff_dim=512, kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2, groups=d_model),
            nn.ReLU(), nn.Conv1d(d_model, d_model, 1), nn.Dropout(dropout))
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, ff_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        attn_in = self.attn_norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in)
        x = x + self.attn_drop(attn_out)
        conv_in = self.conv_norm(x).permute(0, 2, 1)
        x = x + self.conv(conv_in).permute(0, 2, 1)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerMultiHead(nn.Module):
    """Conformer backbone + same 5-head classifier as CNNMultiHead"""
    def __init__(self, pretrained_emb_weights, d_model=256, num_layers=2,
                 num_heads=4, ff_dim=512, kernel_size=15, max_len=32, dropout=0.15):
        super().__init__()
        vocab_size, emb_dim = pretrained_emb_weights.shape

        self.token_emb = nn.Embedding.from_pretrained(pretrained_emb_weights, freeze=True, padding_idx=0)
        self.proj = nn.Linear(emb_dim, d_model)
        self.drop_in = nn.Dropout(dropout)

        # Position embedding
        self.pos_emb = nn.Embedding(max_len + 1, d_model)  # +1 for CLS

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_dim, kernel_size, dropout)
            for _ in range(num_layers)
        ])

        # Same 5 classification heads
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model // 2, nc)
            )
            for name, nc in HEAD_NC.items()
        })

        self.max_len = max_len

    def forward(self, token_ids):
        B = token_ids.shape[0]
        seq_len = min(token_ids.shape[1], self.max_len)

        x = self.proj(self.token_emb(token_ids[:, :seq_len].long()))  # [B, T, d]
        x = self.drop_in(x)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+T, d]

        # Position embedding
        positions = torch.arange(x.shape[1], device=token_ids.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # Conformer encoding
        for block in self.blocks:
            x = block(x)

        # CLS output
        cls_out = x[:, 0, :]

        return {name: head(cls_out) for name, head in self.heads.items()}


# ============================================================
# Dataset (same as v46)
# ============================================================
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
        if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
        if p['judge'] != 'none': p['param_type'] = 'none'
        if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'
        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1;
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

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    model = ConformerMultiHead(pw, d_model=256, num_layers=2, num_heads=4,
                                ff_dim=512, kernel_size=15, max_len=32, dropout=0.15).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Conformer 2L: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    train_ds = MixupDataset(train_data, tok, mixup_prob=0.3)
    val_ds = MixupDataset(val_data, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_combo = 0

    for epoch in range(40):
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

        # Val eval
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
        exec_acc = head_correct['exec_type'] / total * 100
        dir_acc = head_correct['param_direction'] / total * 100
        combo_acc = all_correct / total * 100

        ts_str = ""
        if (epoch + 1) % 10 == 0:
            ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
            ke_fn = eval_koelectra(model, tok, device)
            ts_str = f" | TS={ts_combo:.1f}% KE={ke_fn:.1f}%"

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} exec={exec_acc:.1f} "
              f"dir={dir_acc:.1f} | combo={combo_acc:.1f}%{ts_str}")

        if combo_acc > best_combo:
            best_combo = combo_acc
            torch.save({'epoch': epoch+1, 'state': model.state_dict(),
                       'combo': combo_acc, 'fn': fn_acc,
                       'arch': 'conformer_2L'}, 'checkpoints/conformer_multihead_v63.pt')
            print(f"  ★ Best: combo={combo_acc:.1f}%")

    # Final eval
    ckpt = torch.load('checkpoints/conformer_multihead_v63.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state']); model.eval()
    ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
    ke_fn = eval_koelectra(model, tok, device)

    print(f"\n{'='*60}")
    print(f"v63 Conformer 2L Results:")
    print(f"  Test Suite: fn={ts_fn:.1f} exec={ts_exec:.1f} dir={ts_dir:.1f} combo={ts_combo:.1f}%")
    print(f"  KoELECTRA fn: {ke_fn:.1f}%")
    print(f"  Balanced: {(ts_combo * ke_fn)**0.5:.1f}")
    print(f"  vs CNN v46: TS=93.3% KE=97.8% bal=95.5")
    print(f"{'='*60}")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train()
