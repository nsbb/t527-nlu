#!/usr/bin/env python3
"""Retrieval Hybrid vs Ensemble 비교
Sample by sample 분석 — 어디서 이기고 어디서 지는지
"""
import os, sys, json, re, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from retrieval_hybrid import SentenceEncoder, GTRetriever, HybridPredictor
from preprocess import preprocess
from model_cnn_multihead import HEAD_I2L, HEAD_NAMES, CNNMultiHead
from transformers import AutoModel, AutoTokenizer
import onnxruntime as ort


def apply_post(p):
    if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
    if p['judge'] != 'none': p['param_type'] = 'none'
    if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
        p['param_type'] = 'none'
    return p


def predict_ensemble(sess, tok, text):
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='np')
    outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
    return apply_post({
        'fn': HEAD_I2L['fn'][outs[0][0].argmax()],
        'exec_type': HEAD_I2L['exec_type'][outs[1][0].argmax()],
        'param_direction': HEAD_I2L['param_direction'][outs[2][0].argmax()],
        'param_type': HEAD_I2L['param_type'][outs[3][0].argmax()],
        'judge': HEAD_I2L['judge'][outs[4][0].argmax()],
    })


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Retrieval hybrid setup
    encoder = SentenceEncoder(device)
    gt_known = json.load(open('data/gt_known_scenarios.json'))
    gt_unknown = json.load(open('data/gt_unknown_scenarios.json'))
    gt_all = gt_known + gt_unknown
    retriever = GTRetriever(encoder, gt_all)

    # v46 fallback
    sbert_full = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert_full.embeddings.word_embeddings.weight.detach()
    del sbert_full
    tok_v46 = AutoTokenizer.from_pretrained('tokenizer/')
    model_v46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    model_v46.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt',
                                           map_location='cpu', weights_only=False)['state'])
    model_v46.eval()

    # Ensemble ONNX
    ens = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                providers=['CPUExecutionProvider'])

    hybrid = HybridPredictor(encoder, retriever, model_v46, tok_v46,
                             high_threshold=0.85, low_threshold=0.5)

    test_suite = json.load(open('data/test_suite.json'))
    ke_val = json.load(open('data/koelectra_converted_val.json'))

    print("=" * 70)
    print("  Retrieval Hybrid (v46 fallback) vs Ensemble v28+v46")
    print("=" * 70)

    # Test Suite
    both_right = hyb_right_ens_wrong = ens_right_hyb_wrong = both_wrong = 0
    examples_hyb_better = []
    examples_ens_better = []

    for t in test_suite:
        h_res = hybrid.predict(t['utterance'], return_detail=True)
        h_p = h_res['labels']
        e_p = predict_ensemble(ens, tok_v46, preprocess(t['utterance']))

        h_ok = (h_p['fn'] == t['fn'] and h_p['exec_type'] == t['exec']
                and h_p['param_direction'] == t['dir'])
        e_ok = (e_p['fn'] == t['fn'] and e_p['exec_type'] == t['exec']
                and e_p['param_direction'] == t['dir'])

        if h_ok and e_ok: both_right += 1
        elif h_ok and not e_ok:
            hyb_right_ens_wrong += 1
            if len(examples_hyb_better) < 10:
                examples_hyb_better.append({
                    't': t, 'h': h_p, 'e': e_p, 'mode': h_res['mode'],
                    'top_sim': h_res['top_sim'], 'top_gt': h_res['top_gt'],
                })
        elif e_ok and not h_ok:
            ens_right_hyb_wrong += 1
            if len(examples_ens_better) < 10:
                examples_ens_better.append({
                    't': t, 'h': h_p, 'e': e_p, 'mode': h_res['mode'],
                    'top_sim': h_res['top_sim'], 'top_gt': h_res['top_gt'],
                })
        else: both_wrong += 1

    n = len(test_suite)
    print(f"\n[Test Suite {n}개 combo 비교]")
    print(f"  둘 다 맞음:            {both_right} ({both_right/n*100:.1f}%)")
    print(f"  Hybrid만 맞음:         {hyb_right_ens_wrong} ({hyb_right_ens_wrong/n*100:.1f}%)")
    print(f"  Ensemble만 맞음:       {ens_right_hyb_wrong} ({ens_right_hyb_wrong/n*100:.1f}%)")
    print(f"  둘 다 틀림:            {both_wrong} ({both_wrong/n*100:.1f}%)")
    print(f"  Hybrid 총:             {both_right + hyb_right_ens_wrong}/{n} = {(both_right + hyb_right_ens_wrong)/n*100:.2f}%")
    print(f"  Ensemble 총:           {both_right + ens_right_hyb_wrong}/{n} = {(both_right + ens_right_hyb_wrong)/n*100:.2f}%")

    print(f"\n=== Hybrid가 맞춘 것 (Ensemble 틀림) — 최대 10개 ===")
    for ex in examples_hyb_better:
        t = ex['t']
        print(f'  "{t["utterance"]}" → exp(fn={t["fn"]}, exec={t["exec"]}, dir={t["dir"]})')
        print(f"    Hybrid [{ex['mode']} sim={ex['top_sim']:.2f}]: fn={ex['h']['fn']} exec={ex['h']['exec_type']} dir={ex['h']['param_direction']}")
        print(f"    Ensemble: fn={ex['e']['fn']} exec={ex['e']['exec_type']} dir={ex['e']['param_direction']}")
        print(f"    (top GT: \"{ex['top_gt']}\")")
        print()

    print(f"\n=== Ensemble이 맞춘 것 (Hybrid 틀림) — 최대 10개 ===")
    for ex in examples_ens_better:
        t = ex['t']
        print(f'  "{t["utterance"]}" → exp(fn={t["fn"]}, exec={t["exec"]}, dir={t["dir"]})')
        print(f"    Hybrid [{ex['mode']} sim={ex['top_sim']:.2f}]: fn={ex['h']['fn']} exec={ex['h']['exec_type']} dir={ex['h']['param_direction']}")
        print(f"    Ensemble: fn={ex['e']['fn']} exec={ex['e']['exec_type']} dir={ex['e']['param_direction']}")
        print(f"    (top GT: \"{ex['top_gt']}\")")
        print()

    # KoELECTRA fn 비교
    h_ok = e_ok = 0
    for d in ke_val:
        h_p = hybrid.predict(d['utterance'])
        e_p = predict_ensemble(ens, tok_v46, preprocess(d['utterance']))
        if h_p['fn'] == d['labels']['fn']: h_ok += 1
        if e_p['fn'] == d['labels']['fn']: e_ok += 1
    n = len(ke_val)
    print(f"\n[KoELECTRA fn {n}개]")
    print(f"  Hybrid:   {h_ok}/{n} = {h_ok/n*100:.2f}%")
    print(f"  Ensemble: {e_ok}/{n} = {e_ok/n*100:.2f}%")


if __name__ == '__main__':
    main()
