#!/usr/bin/env python3
"""Model Soup — Weight interpolation between v28 and v46
merged_weights = alpha * v28 + (1-alpha) * v46

Grid search alpha from 0.0 to 1.0 and evaluate on TS + KE.
"""
import torch, torch.nn.functional as F, json, os, sys, re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


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
        if HEAD_I2L['fn'][l['fn'].argmax(1).item()] == d['labels']['fn']:
            fn_ok += 1
    return fn_ok / len(ke_val) * 100


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load embeddings
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    # Load both checkpoints
    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    ckpt46 = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
    state28 = ckpt28['state']
    state46 = ckpt46['state']

    print(f"v28: epoch {ckpt28['epoch']}, combo {ckpt28['combo']:.1f}%")
    print(f"v46: epoch {ckpt46['epoch']}, combo {ckpt46['combo']:.1f}%")

    # Grid search alpha
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = []
    best_balanced = 0
    best_alpha = 0

    print(f"\n{'alpha':>6} | {'TS fn':>6} {'TS exec':>7} {'TS dir':>6} {'TS combo':>8} | {'KE fn':>6} | {'balanced':>8}")
    print("-" * 70)

    for alpha in alphas:
        # Interpolate weights
        merged_state = {}
        for key in state28:
            merged_state[key] = alpha * state28[key].float() + (1 - alpha) * state46[key].float()
            if state28[key].dtype != torch.float32:
                merged_state[key] = merged_state[key].to(state28[key].dtype)

        model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
        model.load_state_dict(merged_state)
        model.eval()

        ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
        ke_fn = eval_koelectra(model, tok, device)

        # Balanced score: geometric mean of TS combo and KE fn
        balanced = (ts_combo * ke_fn) ** 0.5

        results.append({
            'alpha': alpha, 'ts_fn': ts_fn, 'ts_exec': ts_exec,
            'ts_dir': ts_dir, 'ts_combo': ts_combo, 'ke_fn': ke_fn,
            'balanced': balanced
        })

        marker = " ★" if balanced > best_balanced else ""
        if balanced > best_balanced:
            best_balanced = balanced
            best_alpha = alpha

        print(f"{alpha:>6.1f} | {ts_fn:>6.1f} {ts_exec:>7.1f} {ts_dir:>6.1f} {ts_combo:>8.1f} | {ke_fn:>6.1f} | {balanced:>8.1f}{marker}")

        del model

    # Save best model
    print(f"\n=== Best alpha: {best_alpha:.1f} (balanced: {best_balanced:.1f}) ===")

    merged_state = {}
    for key in state28:
        merged_state[key] = best_alpha * state28[key].float() + (1 - best_alpha) * state46[key].float()
        if state28[key].dtype != torch.float32:
            merged_state[key] = merged_state[key].to(state28[key].dtype)

    best_r = [r for r in results if r['alpha'] == best_alpha][0]
    torch.save({
        'epoch': 0, 'state': merged_state,
        'combo': best_r['ts_combo'], 'fn': best_r['ts_fn'],
        'alpha': best_alpha,
        'ts_combo': best_r['ts_combo'], 'ke_fn': best_r['ke_fn'],
        'model_soup': True,
    }, 'checkpoints/cnn_multihead_v60.pt')
    print(f"Saved: checkpoints/cnn_multihead_v60.pt")

    # Also try head-specific alpha: different alpha per head
    print(f"\n=== Head-specific interpolation ===")
    print("Strategy: fn head from v46 (alpha=0), exec/dir heads from v28 (alpha=1)")

    # Separate head keys from backbone keys
    backbone_keys = [k for k in state28 if not k.startswith('heads.')]
    head_keys = {h: [k for k in state28 if k.startswith(f'heads.{h}.')] for h in HEAD_NAMES}

    # Try different backbone alphas with head-specific alphas
    head_alphas = {
        'fn': 0.0,       # v46 fn (better generalization)
        'exec_type': 1.0, # v28 exec (better accuracy)
        'param_direction': 1.0,  # v28 dir (better accuracy)
        'param_type': 0.5,  # mixed
        'judge': 0.0,    # v46 judge (better with external data)
    }

    for backbone_alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        merged = {}
        for key in backbone_keys:
            merged[key] = backbone_alpha * state28[key].float() + (1 - backbone_alpha) * state46[key].float()
            if state28[key].dtype != torch.float32:
                merged[key] = merged[key].to(state28[key].dtype)

        for h, h_alpha in head_alphas.items():
            for key in head_keys[h]:
                merged[key] = h_alpha * state28[key].float() + (1 - h_alpha) * state46[key].float()
                if state28[key].dtype != torch.float32:
                    merged[key] = merged[key].to(state28[key].dtype)

        model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15).to(device)
        model.load_state_dict(merged); model.eval()

        ts_fn, ts_exec, ts_dir, ts_combo = eval_test_suite(model, tok, device)
        ke_fn = eval_koelectra(model, tok, device)
        balanced = (ts_combo * ke_fn) ** 0.5

        marker = " ★★" if balanced > best_balanced else ""
        if balanced > best_balanced:
            best_balanced = balanced
            torch.save({
                'epoch': 0, 'state': merged,
                'combo': ts_combo, 'fn': ts_fn,
                'backbone_alpha': backbone_alpha, 'head_alphas': head_alphas,
                'ts_combo': ts_combo, 'ke_fn': ke_fn,
                'model_soup': True, 'head_specific': True,
            }, 'checkpoints/cnn_multihead_v60.pt')

        print(f"  backbone={backbone_alpha:.1f} heads=[fn=0.0,exec=1.0,dir=1.0] | "
              f"TS={ts_combo:.1f}% KE={ke_fn:.1f}% bal={balanced:.1f}{marker}")

        del model


if __name__ == '__main__':
    main()
