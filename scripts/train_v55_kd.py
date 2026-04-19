#!/usr/bin/env python3
"""v55 — Knowledge Distillation from Ensemble (v28+v46)
Single model trained to match ensemble predictions via soft labels.
Ensemble strategy B: fn=v46, exec/dir=v28

Goal: single model with ensemble-level performance (TS 94%+ KE 97%+)
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
# Step 1: Generate soft labels from ensemble
# ============================================================
def generate_soft_labels(train_data, pw, tok, device='cpu'):
    """Generate soft label logits from v28+v46 ensemble for each training sample"""
    print("=== Generating soft labels from ensemble (v28+v46) ===")

    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    m28.load_state_dict(ckpt28['state']); m28.eval(); m28.to(device)
    print(f"v28 loaded (epoch {ckpt28['epoch']}, combo {ckpt28['combo']:.1f}%)")

    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt46 = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
    m46.load_state_dict(ckpt46['state']); m46.eval(); m46.to(device)
    print(f"v46 loaded (epoch {ckpt46['epoch']}, combo {ckpt46['combo']:.1f}%)")

    soft_labels = []
    batch_size = 128
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        texts = [d['utterance'] for d in batch]
        tks = tok(texts, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        input_ids = tks['input_ids'].to(device)

        with torch.no_grad():
            l28 = m28(input_ids)
            l46 = m46(input_ids)

        # Ensemble Strategy B: fn=v46, exec/dir=v28
        for j in range(len(batch)):
            soft = {
                'fn': l46['fn'][j].cpu(),           # fn from v46 (better generalization)
                'exec_type': l28['exec_type'][j].cpu(),  # exec from v28 (better accuracy)
                'param_direction': l28['param_direction'][j].cpu(),  # dir from v28
                'param_type': l28['param_type'][j].cpu(),
                'judge': l46['judge'][j].cpu(),      # judge from v46
            }
            soft_labels.append(soft)

        if (i // batch_size) % 50 == 0:
            print(f"  {i}/{len(train_data)} processed...")

    del m28, m46
    torch.cuda.empty_cache() if device != 'cpu' else None
    print(f"Generated soft labels for {len(soft_labels)} samples")
    return soft_labels


# ============================================================
# Dataset with soft labels + mixup
# ============================================================
class KDDataset(Dataset):
    def __init__(self, data, soft_labels, tokenizer, max_len=32, mixup_prob=0.3):
        self.data = data
        self.soft_labels = soft_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mixup_prob = mixup_prob

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

        # Hard labels
        hard_labels = {}
        for h in HEAD_NAMES:
            val = d['labels'].get(h, 'none')
            hard_labels[h] = HEAD_L2I[h].get(val, 0)

        # Soft labels (logits from ensemble)
        soft = self.soft_labels[idx]

        # Mixup: swap utterance with same-fn sample
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
                soft = self.soft_labels[j]

        return input_ids, hard_labels, soft


def kd_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    hard_labels = {h: torch.tensor([b[1][h] for b in batch]) for h in HEAD_NAMES}
    soft_labels = {h: torch.stack([b[2][h] for b in batch]) for h in HEAD_NAMES}
    return input_ids, hard_labels, soft_labels


# ============================================================
# KD Loss: alpha * KD_loss + (1-alpha) * CE_loss
# ============================================================
def kd_loss(student_logits, hard_labels, teacher_logits, alpha=0.7, temperature=3.0,
            fn_weights=None, head_weights=None):
    """
    Knowledge distillation loss
    - Soft loss: KL divergence between student and teacher soft predictions
    - Hard loss: Cross entropy with original labels
    """
    if head_weights is None:
        head_weights = {'fn': 2.0, 'exec_type': 2.0, 'param_direction': 1.5,
                       'param_type': 1.0, 'judge': 1.5}

    total_loss = 0

    for h in HEAD_NAMES:
        w = head_weights.get(h, 1.0)

        # Soft loss: KL(student || teacher) at temperature T
        student_soft = F.log_softmax(student_logits[h] / temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits[h] / temperature, dim=1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

        # Hard loss: CE with original labels
        if h == 'fn' and fn_weights is not None:
            hard_loss = F.cross_entropy(student_logits[h], hard_labels[h], weight=fn_weights)
        else:
            hard_loss = F.cross_entropy(student_logits[h], hard_labels[h])

        total_loss += w * (alpha * soft_loss + (1 - alpha) * hard_loss)

    return total_loss


# ============================================================
# Training
# ============================================================
def train_v55():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Use v43 train/val (what v46 was trained with)
    train_file = 'data/train_final_v43.json'
    val_file = 'data/val_final_v43.json'

    # Fallback
    if not os.path.exists(train_file):
        train_file = 'data/train_final_v34.json'
        val_file = 'data/val_final_v34.json'

    with open(train_file) as f:
        train_data = json.load(f)
    with open(val_file) as f:
        val_data = json.load(f)
    print(f"Train: {len(train_data)} ({train_file})")
    print(f"Val: {len(val_data)} ({val_file})")

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert

    # Generate soft labels
    soft_labels_train = generate_soft_labels(train_data, pw, tok, device)
    soft_labels_val = generate_soft_labels(val_data, pw, tok, device)

    # Model (student)
    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Student: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")

    train_ds = KDDataset(train_data, soft_labels_train, tok, mixup_prob=0.3)
    val_ds = KDDataset(val_data, soft_labels_val, tok, mixup_prob=0.0)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=kd_collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=kd_collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    fn_counts = Counter(d['labels']['fn'] for d in train_data)
    fn_total = sum(fn_counts.values())
    fn_weights = torch.tensor([fn_total / (len(fn_counts) * fn_counts.get(c, 1))
                               for c in HEAD_CLASSES['fn']], dtype=torch.float32).to(device)
    fn_weights = fn_weights.clamp(max=5.0)

    best_combo = 0
    best_fn = 0

    # SWA collection
    swa_states = []
    SWA_START = 25

    for epoch in range(40):
        model.train()
        train_loss = 0; train_n = 0

        # Anneal alpha: start at 0.7, decrease to 0.3 over training
        alpha = max(0.3, 0.7 - 0.4 * epoch / 40)

        for input_ids, hard_labels, soft_labels in train_dl:
            input_ids = input_ids.to(device)
            hard_labels = {h: v.to(device) for h, v in hard_labels.items()}
            soft_labels = {h: v.to(device) for h, v in soft_labels.items()}

            logits = model(input_ids)
            loss = kd_loss(logits, hard_labels, soft_labels, alpha=alpha,
                          temperature=3.0, fn_weights=fn_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * input_ids.size(0)
            train_n += input_ids.size(0)

        scheduler.step()

        if epoch >= SWA_START:
            swa_states.append(copy.deepcopy(model.state_dict()))

        # Eval (hard labels only)
        model.eval()
        head_correct = {h: 0 for h in HEAD_NAMES}
        head_total = {h: 0 for h in HEAD_NAMES}
        all_correct = 0; all_total = 0

        with torch.no_grad():
            for input_ids, hard_labels, _ in val_dl:
                input_ids = input_ids.to(device)
                hard_labels = {h: v.to(device) for h, v in hard_labels.items()}
                logits = model(input_ids)
                B = input_ids.size(0)
                all_match = torch.ones(B, dtype=torch.bool, device=device)
                for h in HEAD_NAMES:
                    pred = logits[h].argmax(1)
                    correct = (pred == hard_labels[h])
                    head_correct[h] += correct.sum().item()
                    head_total[h] += B
                    all_match &= correct
                all_correct += all_match.sum().item()
                all_total += B

        fn_acc = head_correct['fn'] / head_total['fn'] * 100
        exec_acc = head_correct['exec_type'] / head_total['exec_type'] * 100
        dir_acc = head_correct['param_direction'] / head_total['param_direction'] * 100
        combo_acc = all_correct / all_total * 100

        print(f"[{epoch+1:2d}] loss={train_loss/train_n:.3f} α={alpha:.2f} | "
              f"fn={fn_acc:.1f} exec={exec_acc:.1f} dir={dir_acc:.1f} | combo={combo_acc:.1f}%"
              + (" [SWA]" if epoch >= SWA_START else ""))

        if combo_acc > best_combo or (combo_acc == best_combo and fn_acc > best_fn):
            best_combo = combo_acc
            best_fn = fn_acc
            torch.save({
                'epoch': epoch + 1, 'state': model.state_dict(),
                'combo': combo_acc, 'fn': fn_acc, 'exec': exec_acc, 'dir': dir_acc,
                'kd': True, 'alpha': alpha,
            }, 'checkpoints/cnn_multihead_v55.pt')
            print(f"  ★ Best: combo={combo_acc:.1f}%, fn={fn_acc:.1f}%")

    # SWA
    if swa_states:
        print(f"\n=== SWA: averaging {len(swa_states)} checkpoints ===")
        avg_state = {}
        for key in swa_states[0]:
            avg_state[key] = torch.stack([s[key].float() for s in swa_states]).mean(0)
            if swa_states[0][key].dtype != torch.float32:
                avg_state[key] = avg_state[key].to(swa_states[0][key].dtype)

        model.load_state_dict(avg_state); model.eval()

        head_correct = {h: 0 for h in HEAD_NAMES}
        head_total = {h: 0 for h in HEAD_NAMES}
        all_correct = 0; all_total = 0
        with torch.no_grad():
            for input_ids, hard_labels, _ in val_dl:
                input_ids = input_ids.to(device)
                hard_labels = {h: v.to(device) for h, v in hard_labels.items()}
                logits = model(input_ids)
                B = input_ids.size(0)
                all_match = torch.ones(B, dtype=torch.bool, device=device)
                for h in HEAD_NAMES:
                    pred = logits[h].argmax(1)
                    correct = (pred == hard_labels[h])
                    head_correct[h] += correct.sum().item()
                    head_total[h] += B
                    all_match &= correct
                all_correct += all_match.sum().item()
                all_total += B

        swa_fn = head_correct['fn'] / head_total['fn'] * 100
        swa_exec = head_correct['exec_type'] / head_total['exec_type'] * 100
        swa_dir = head_correct['param_direction'] / head_total['param_direction'] * 100
        swa_combo = all_correct / all_total * 100
        print(f"SWA: fn={swa_fn:.1f} exec={swa_exec:.1f} dir={swa_dir:.1f} combo={swa_combo:.1f}%")

        torch.save({
            'epoch': 40, 'state': avg_state,
            'combo': swa_combo, 'fn': swa_fn, 'exec': swa_exec, 'dir': swa_dir,
            'swa': True, 'kd': True,
        }, 'checkpoints/cnn_multihead_v55_swa.pt')

    print(f"\n=== v55 Final: best combo={best_combo:.1f}%, fn={best_fn:.1f}% ===")


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    train_v55()
