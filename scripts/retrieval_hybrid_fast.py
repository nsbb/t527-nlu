#!/usr/bin/env python3
"""Retrieval Hybrid — Fast version (배치 임베딩 + threshold sweep 재사용)
기존 retrieval_hybrid.py의 느림 문제 해결:
- 테스트 발화 전체를 한 번만 배치 임베딩 → threshold마다 재사용
- 모델 fallback도 한 번만 모든 발화 예측 → threshold에 따라 선택

실행 시간: 이전 >1시간 → 예상 5분 이내
"""
import os, sys, json, re, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from transformers import AutoModel, AutoTokenizer
from model_cnn_multihead import HEAD_I2L, HEAD_NAMES, CNNMultiHead
from preprocess import preprocess


def sentence_encode(model, tok, texts, device, batch_size=64):
    """L2-normalized mean-pooled embeddings"""
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**enc)
        mask = enc['attention_mask'].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = F.normalize(emb, p=2, dim=1)
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


def model_predict_all(model, tok_cnn, texts, device, batch_size=64):
    """v46 모델로 모든 발화 예측 (배치)"""
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tk = tok_cnn(batch, padding='max_length', truncation=True,
                     max_length=32, return_tensors='pt').to(device)
        with torch.no_grad():
            l = model(tk['input_ids'])
        for j in range(len(batch)):
            p = {h: HEAD_I2L[h][l[h][j].argmax().item()] for h in HEAD_NAMES}
            if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
            if p['judge'] != 'none': p['param_type'] = 'none'
            if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
                p['param_type'] = 'none'
            preds.append(p)
    return preds


def apply_retrieval(query_embs, gt_embs, gt_labels, high_th, low_th, model_preds):
    """주어진 threshold로 hybrid 결과 생성"""
    # Cosine sim (L2 normalized이므로 dot)
    sims = query_embs @ gt_embs.T  # [N_q, N_gt]
    top_sims = sims.max(axis=1)    # [N_q]
    top_idx = sims.argmax(axis=1)  # [N_q]

    preds = []
    modes = []
    for i, (sim, gi) in enumerate(zip(top_sims, top_idx)):
        if sim >= high_th:
            preds.append(gt_labels[gi])
            modes.append('retrieval')
        elif sim >= low_th:
            preds.append(model_preds[i])
            modes.append('model')
        else:
            preds.append({'fn': 'unknown', 'exec_type': 'direct_respond',
                          'param_direction': 'none', 'param_type': 'none', 'judge': 'none'})
            modes.append('unknown')
    return preds, modes


def score(preds, targets, fn_only=False):
    """Test Suite / GT scenarios / KE 공통 scoring"""
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for p, t in zip(preds, targets):
        if 'labels' in t:
            exp = t['labels']
            exp_fn = exp.get('fn')
            exp_exec = exp.get('exec_type')
            exp_dir = exp.get('param_direction')
        else:
            exp_fn = t.get('fn')
            exp_exec = t.get('exec')
            exp_dir = t.get('dir')

        if p['fn'] == exp_fn: fn_ok += 1
        if exp_exec is not None and p['exec_type'] == exp_exec: exec_ok += 1
        if exp_dir is not None and p['param_direction'] == exp_dir: dir_ok += 1
        if p['fn'] == exp_fn and (exp_exec is None or p['exec_type'] == exp_exec) and \
           (exp_dir is None or p['param_direction'] == exp_dir):
            all_ok += 1

    n = len(targets)
    r = {'fn': fn_ok / n * 100, 'total': n}
    if not fn_only:
        r['exec'] = exec_ok / n * 100
        r['dir'] = dir_ok / n * 100
        r['combo'] = all_ok / n * 100
    return r


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    print("=== 모델 로드 ===")
    t0 = time.time()
    sbert_tok = AutoTokenizer.from_pretrained('jhgan/ko-sbert-sts')
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts').to(device).eval()
    print(f"  ko-sbert ({time.time()-t0:.1f}s)")

    t0 = time.time()
    tok_cnn = AutoTokenizer.from_pretrained('tokenizer/')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    model_v46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    model_v46.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt',
                                           map_location='cpu', weights_only=False)['state'])
    model_v46 = model_v46.to(device).eval()
    print(f"  v46 ({time.time()-t0:.1f}s)")

    # GT pool
    gt_known = json.load(open('data/gt_known_scenarios.json'))
    gt_unknown = json.load(open('data/gt_unknown_scenarios.json'))
    gt_all = gt_known + gt_unknown
    gt_utts = [preprocess(d['utterance']) for d in gt_all]
    gt_labels = [d['labels'] for d in gt_all]

    print(f"\n=== GT 임베딩 ({len(gt_all)}개) ===")
    t0 = time.time()
    gt_embs = sentence_encode(sbert, sbert_tok, gt_utts, device)
    print(f"  {gt_embs.shape} ({time.time()-t0:.1f}s)")

    # Test Suite 임베딩 + 모델 예측 (한 번만)
    test_suite = json.load(open('data/test_suite.json'))
    ts_utts = [preprocess(t['utterance']) for t in test_suite]
    print(f"\n=== Test Suite 임베딩 ({len(test_suite)}개) ===")
    t0 = time.time()
    ts_embs = sentence_encode(sbert, sbert_tok, ts_utts, device)
    print(f"  {ts_embs.shape} ({time.time()-t0:.1f}s)")

    print(f"\n=== Test Suite v46 예측 (fallback) ===")
    t0 = time.time()
    ts_model_preds = model_predict_all(model_v46, tok_cnn, ts_utts, device)
    print(f"  {len(ts_model_preds)}개 ({time.time()-t0:.1f}s)")

    # KoELECTRA 임베딩 + 모델 예측
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    ke_utts = [preprocess(d['utterance']) for d in ke_val]
    print(f"\n=== KoELECTRA 임베딩 ({len(ke_val)}개) ===")
    t0 = time.time()
    ke_embs = sentence_encode(sbert, sbert_tok, ke_utts, device)
    print(f"  {ke_embs.shape} ({time.time()-t0:.1f}s)")

    print(f"\n=== KoELECTRA v46 예측 (fallback) ===")
    t0 = time.time()
    ke_model_preds = model_predict_all(model_v46, tok_cnn, ke_utts, device)
    print(f"  {len(ke_model_preds)}개 ({time.time()-t0:.1f}s)")

    # Baseline: v46 단독
    ts_v46 = score(ts_model_preds, test_suite)
    ke_v46 = score(ke_model_preds, ke_val, fn_only=True)

    print(f"\n{'='*80}")
    print(f"  Threshold Sweep 결과")
    print('='*80)
    print(f"\n[Baseline]")
    print(f"  v46 단독 — TS combo: {ts_v46['combo']:.2f}%  KE fn: {ke_v46['fn']:.2f}%")

    print(f"\n[Hybrid Sweep]")
    print(f"{'HIGH':>6} {'TS combo':>10} {'TS fn':>7} {'TS exec':>8} {'TS dir':>8} "
          f"{'KE fn':>7} {'TS retr%':>9} {'TS model%':>10} {'TS unk%':>9} {'KE retr%':>9}")
    print('-' * 90)

    for ht in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        ts_preds, ts_modes = apply_retrieval(ts_embs, gt_embs, gt_labels, ht, 0.5, ts_model_preds)
        ke_preds, ke_modes = apply_retrieval(ke_embs, gt_embs, gt_labels, ht, 0.5, ke_model_preds)

        ts_r = score(ts_preds, test_suite)
        ke_r = score(ke_preds, ke_val, fn_only=True)

        ts_retr_pct = sum(1 for m in ts_modes if m == 'retrieval') / len(ts_modes) * 100
        ts_model_pct = sum(1 for m in ts_modes if m == 'model') / len(ts_modes) * 100
        ts_unk_pct = sum(1 for m in ts_modes if m == 'unknown') / len(ts_modes) * 100
        ke_retr_pct = sum(1 for m in ke_modes if m == 'retrieval') / len(ke_modes) * 100

        print(f"{ht:>6.2f} {ts_r['combo']:>10.2f} {ts_r['fn']:>7.2f} {ts_r['exec']:>8.2f} "
              f"{ts_r['dir']:>8.2f} {ke_r['fn']:>7.2f} "
              f"{ts_retr_pct:>9.1f} {ts_model_pct:>10.1f} {ts_unk_pct:>9.1f} {ke_retr_pct:>9.1f}")

    # GT leave-one-out self-test
    print(f"\n=== GT 219개 Leave-One-Out Retrieval 자기테스트 ===")
    lo_sims = gt_embs @ gt_embs.T
    np.fill_diagonal(lo_sims, -1)
    lo_top_idx = lo_sims.argmax(axis=1)
    lo_top_sim = lo_sims[np.arange(len(gt_all)), lo_top_idx]

    fn_ok = exec_ok = dir_ok = all_ok = 0
    for i, gi in enumerate(lo_top_idx):
        exp = gt_labels[i]; got = gt_labels[gi]
        f = exp['fn'] == got['fn']
        e = exp.get('exec_type') == got.get('exec_type')
        d = exp.get('param_direction') == got.get('param_direction')
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1

    n = len(gt_all)
    print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.1f}%")
    print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.1f}%")
    print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.1f}%")
    print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.1f}%")
    print(f"  top-1 sim 분포: mean={lo_top_sim.mean():.3f} min={lo_top_sim.min():.3f} max={lo_top_sim.max():.3f}")
    print(f"  sim>0.85: {(lo_top_sim > 0.85).sum()}개  >0.9: {(lo_top_sim > 0.9).sum()}개")


if __name__ == '__main__':
    main()
