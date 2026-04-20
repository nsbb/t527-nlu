#!/usr/bin/env python3
"""3-Model Majority-Vote Ensemble ONNX Export
v28 + v34 + v46: 각 head별 3개 예측 중 majority argmax

ONNX에서 argmax + majority voting 구현은 복잡하므로,
여기서는 **softmax 확률 평균**으로 구현 (majority vote 근사)
softmax average ≈ majority vote in most cases
"""
import os, sys, time, json, re
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


class TripleEnsembleModel(nn.Module):
    """v28 + v34 + v46 softmax-average ensemble"""
    def __init__(self, pw):
        super().__init__()
        self.m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)
        self.m34 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)
        self.m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)

    def forward(self, input_ids):
        l28 = self.m28(input_ids)
        l34 = self.m34(input_ids)
        l46 = self.m46(input_ids)

        # Softmax probabilities averaged (근사 majority vote)
        out = []
        for h in ['fn', 'exec_type', 'param_direction', 'param_type', 'judge']:
            p28 = torch.softmax(l28[h], dim=-1)
            p34 = torch.softmax(l34[h], dim=-1)
            p46 = torch.softmax(l46[h], dim=-1)
            avg = (p28 + p34 + p46) / 3.0
            out.append(avg)

        # Returns: fn, exec, dir, param, judge
        return tuple(out)


def main():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert

    model = TripleEnsembleModel(pw)

    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    ckpt34 = torch.load('checkpoints/cnn_multihead_v34.pt', map_location='cpu', weights_only=False)
    ckpt46 = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)

    model.m28.load_state_dict(ckpt28['state'])
    model.m34.load_state_dict(ckpt34['state'])
    model.m46.load_state_dict(ckpt46['state'])
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Triple ensemble: {total/1e6:.1f}M params")

    dummy = torch.zeros(1, 32, dtype=torch.long)
    with torch.no_grad():
        outs = model(dummy)
    for i, name in enumerate(['fn', 'exec', 'dir', 'param', 'judge']):
        print(f"  {name}: {outs[i].shape}")

    onnx_path = 'checkpoints/nlu_v28_v34_v46_triple.onnx'
    output_names = ['fn_probs', 'exec_probs', 'dir_probs', 'param_probs', 'judge_probs']

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input_ids'], output_names=output_names,
        dynamic_axes={'input_ids': {0: 'batch'}, **{n: {0: 'batch'} for n in output_names}},
        opset_version=14, do_constant_folding=True,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX saved: {onnx_path} ({size_mb:.1f}MB)")

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    # Test Suite
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in t['utterance'])).strip()
        tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='np')
        outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        fn = HEAD_I2L['fn'][outs[0][0].argmax()]
        exec_t = HEAD_I2L['exec_type'][outs[1][0].argmax()]
        dir_t = HEAD_I2L['param_direction'][outs[2][0].argmax()]
        f = fn == t['fn']; e = exec_t == t['exec']; d = dir_t == t['dir']
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1
    n = len(suite)
    print(f"\n=== Test Suite ({n}) ===")
    print(f"  fn={fn_ok/n*100:.2f}% exec={exec_ok/n*100:.2f}% dir={dir_ok/n*100:.2f}% combo={all_ok/n*100:.2f}%")

    # KE
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    ke_ok = 0
    for d in ke_val:
        tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
        outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        if HEAD_I2L['fn'][outs[0][0].argmax()] == d['labels']['fn']:
            ke_ok += 1
    print(f"KE fn: {ke_ok/len(ke_val)*100:.2f}%")

    # Latency
    tk = tok("거실 에어컨 켜줘", padding='max_length', truncation=True, max_length=32, return_tensors='np')
    input_ids = tk['input_ids'].astype(np.int64)
    for _ in range(10): sess.run(None, {'input_ids': input_ids})
    N = 100
    start = time.time()
    for _ in range(N): sess.run(None, {'input_ids': input_ids})
    latency = (time.time() - start) * 1000 / N
    print(f"\nLatency: {latency:.2f}ms")

    print(f"\n=== 요약 ===")
    print(f"  크기: {size_mb:.1f}MB (vs 2-model 104.9MB)")
    print(f"  Test Suite combo: {all_ok/n*100:.2f}%")
    print(f"  KE fn: {ke_ok/len(ke_val)*100:.2f}%")
    print(f"  Latency: {latency:.2f}ms")


if __name__ == '__main__':
    main()
