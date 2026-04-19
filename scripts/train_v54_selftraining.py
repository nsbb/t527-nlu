#!/usr/bin/env python3
"""v54 — Self-training Round 2: re-pseudo-label KoELECTRA with v46
v28 pseudo labels → v46 pseudo labels (better exec/dir predictions)
Then train with mixup augmentation (same as v46).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import json, os, sys, time, random, copy
from torch.utils.data import Dataset, DataLoader
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer

# ============================================================
# Step 1: Re-pseudo-label KoELECTRA with v46
# ============================================================
def repseudo_label_koelectra():
    """Use v46 to re-predict exec/dir on KoELECTRA train data"""
    print("=== Re-pseudo-labeling KoELECTRA with v46 ===")

    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    # Load v46
    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt46 = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
    m46.load_state_dict(ckpt46['state']); m46.eval()
    print(f"v46 loaded (epoch {ckpt46['epoch']}, combo {ckpt46['combo']:.1f}%)")

    # Load KoELECTRA original train data (with original fn labels)
    with open('data/koelectra_converted_train.json') as f:
        ke_data = json.load(f)
    print(f"KoELECTRA train: {len(ke_data)} samples")

    # Re-predict exec/dir/param/judge with v46, keep original fn
    relabeled = []
    for d in ke_data:
        text = d['utterance']
        t = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            logits = m46(t['input_ids'])

        v46_preds = {h: HEAD_I2L[h][logits[h].argmax(1).item()] for h in HEAD_NAMES}
        conf = F.softmax(logits['fn'], dim=1).max().item()

        # Keep original fn (KoELECTRA's fn labels are good)
        # Use v46's exec/dir/param/judge predictions
        original_fn = d['labels']['fn']

        # Skip if v46 confidence is very low
        if conf < 0.3:
            continue

        new_labels = {
            'fn': original_fn,
            'exec_type': v46_preds['exec_type'],
            'param_direction': v46_preds['param_direction'],
            'param_type': v46_preds['param_type'],
            'judge': v46_preds['judge'],
        }

        # param_type rule correction
        if new_labels['param_direction'] in ('open', 'close', 'stop'):
            new_labels['param_type'] = 'none'
        if new_labels['judge'] != 'none':
            new_labels['param_type'] = 'none'
        if new_labels['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
            new_labels['param_type'] = 'none'

        relabeled.append({
            'utterance': text,
            'labels': new_labels,
            'source': 'koelectra_pseudo_v46',
        })

    print(f"Re-labeled: {len(relabeled)} samples (filtered low-conf)")

    # Save
    with open('data/koelectra_pseudo_v46.json', 'w', encoding='utf-8') as f:
        json.dump(relabeled, f, ensure_ascii=False, indent=2)

    return relabeled, pw, tok


# ============================================================
# Step 2: Build v54 training data
# ============================================================
def build_v54_data(relabeled_ke):
    """Replace old KoELECTRA pseudo labels in v34 data with v46 re-labels"""
    with open('data/train_final_v34.json') as f:
        v34_data = json.load(f)

    # Remove old KoELECTRA pseudo-labeled data
    non_ke = [d for d in v34_data if d.get('source') != 'koelectra_pseudo']
    print(f"v34 non-KoELECTRA: {len(non_ke)} samples")
    print(f"v46 re-labeled KoELECTRA: {len(relabeled_ke)} samples")

    v54_data = non_ke + relabeled_ke
    random.shuffle(v54_data)

    with open('data/train_final_v54.json', 'w', encoding='utf-8') as f:
        json.dump(v54_data, f, ensure_ascii=False, indent=2)

    print(f"v54 total: {len(v54_data)} samples")
    return v54_data


# ============================================================
# Dataset with Mixup
# ============================================================
class MixupDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Group by fn for mixup
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

        labels = {}
        for h in HEAD_NAMES:
            val = d['labels'].get(h, 'none')
            labels[h] = HEAD_L2I[h].get(val, 0)

        # Mixup: 30% chance to swap with same-fn utterance
        if random.random() < 0.3:
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
                # Keep original labels (same fn class)

        return input_ids, labels


def collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    return input_ids, labels


# ============================================================
# Training with SWA
# ============================================================
def train_v54(train_data, pw, tok):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Train: {len(train_data)} samples")

    # Val data
    with open('data/val_final.json') as f:
        val_data = json.load(f)
    print(f"Val: {len(val_data)} samples")

    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    train_ds = MixupDataset(train_data, tok)
    val_ds = MixupDataset(val_data, tok)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    # Class weights
    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_combo = 0
    best_fn = 0
    best_state = None

    # SWA: collect weights from epoch 25+
    swa_states = []
    SWA_START = 25

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

        # Collect SWA weights
        if epoch >= SWA_START:
            swa_states.append(copy.deepcopy(model.state_dict()))

        # Eval
        model.eval()
        head_correct = {h: 0 for h in HEAD_NAMES}
        head_total = {h: 0 for h in HEAD_NAMES}
        all_correct = 0; all_total = 0

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
                    head_total[h] += B
                    all_match &= correct
                all_correct += all_match.sum().item()
                all_total += B

        fn_acc = head_correct['fn'] / head_total['fn'] * 100
        exec_acc = head_correct['exec_type'] / head_total['exec_type'] * 100
        dir_acc = head_correct['param_direction'] / head_total['param_direction'] * 100
        combo_acc = all_correct / all_total * 100

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} | fn={fn_acc:.1f} exec={exec_acc:.1f} dir={dir_acc:.1f} | combo={combo_acc:.1f}%"
              + (" [SWA]" if epoch >= SWA_START else ""))

        if combo_acc > best_combo or (combo_acc == best_combo and fn_acc > best_fn):
            best_combo = combo_acc
            best_fn = fn_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch + 1, 'state': model.state_dict(),
                'combo': combo_acc, 'fn': fn_acc, 'exec': exec_acc, 'dir': dir_acc,
            }, 'checkpoints/cnn_multihead_v54.pt')
            print(f"  ★ Best: combo={combo_acc:.1f}%, fn={fn_acc:.1f}%")

    # SWA averaging
    if swa_states:
        print(f"\n=== SWA: averaging {len(swa_states)} checkpoints (epochs {SWA_START+1}-40) ===")
        avg_state = {}
        for key in swa_states[0]:
            avg_state[key] = torch.stack([s[key].float() for s in swa_states]).mean(0)
            if swa_states[0][key].dtype != torch.float32:
                avg_state[key] = avg_state[key].to(swa_states[0][key].dtype)

        model.load_state_dict(avg_state)
        model.eval()

        # Eval SWA model
        head_correct = {h: 0 for h in HEAD_NAMES}
        head_total = {h: 0 for h in HEAD_NAMES}
        all_correct = 0; all_total = 0

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
                    head_total[h] += B
                    all_match &= correct
                all_correct += all_match.sum().item()
                all_total += B

        swa_fn = head_correct['fn'] / head_total['fn'] * 100
        swa_combo = all_correct / all_total * 100
        print(f"SWA result: fn={swa_fn:.1f}%, combo={swa_combo:.1f}%")

        torch.save({
            'epoch': 40, 'state': avg_state,
            'combo': swa_combo, 'fn': swa_fn,
            'swa': True, 'swa_n': len(swa_states),
        }, 'checkpoints/cnn_multihead_v54_swa.pt')
        print(f"SWA model saved")

    print(f"\n=== v54 best: combo={best_combo:.1f}%, fn={best_fn:.1f}% ===")
    return best_combo, best_fn


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)

    # Step 1: re-pseudo-label
    relabeled, pw, tok = repseudo_label_koelectra()

    # Step 2: build data
    train_data = build_v54_data(relabeled)

    # Step 3: train
    train_v54(train_data, pw, tok)
