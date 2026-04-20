#!/usr/bin/env python3
"""Verify ensemble ONNX matches PyTorch ensemble performance"""
import os, sys, json, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from transformers import AutoTokenizer

import onnxruntime as ort


def main():
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                providers=['CPUExecutionProvider'])

    # Test Suite eval
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
    print(f"=== ONNX Ensemble Test Suite ({n}개) ===")
    print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.1f}%")
    print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.1f}%")
    print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.1f}%")
    print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.1f}%")

    # KoELECTRA eval
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    ke_ok = 0
    for d in ke_val:
        tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
        outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        if HEAD_I2L['fn'][outs[0][0].argmax()] == d['labels']['fn']:
            ke_ok += 1

    print(f"\n=== ONNX Ensemble KoELECTRA ({len(ke_val)}개) ===")
    print(f"  fn: {ke_ok}/{len(ke_val)} = {ke_ok/len(ke_val)*100:.1f}%")

    # Latency benchmark
    import time
    # Warmup
    for _ in range(10):
        tk = tok("거실 에어컨 켜줘", padding='max_length', truncation=True, max_length=32, return_tensors='np')
        sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})

    # Measure
    N = 100
    start = time.time()
    for _ in range(N):
        tk = tok("거실 에어컨 23도로 맞춰줘", padding='max_length', truncation=True, max_length=32, return_tensors='np')
        sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
    total = (time.time() - start) * 1000

    print(f"\n=== Latency (CPU, 100회 avg) ===")
    print(f"  토큰화 + 추론: {total/N:.2f}ms")

    # Just inference
    tk = tok("거실 에어컨 23도로 맞춰줘", padding='max_length', truncation=True, max_length=32, return_tensors='np')
    input_ids = tk['input_ids'].astype(np.int64)

    start = time.time()
    for _ in range(N):
        sess.run(None, {'input_ids': input_ids})
    infer_total = (time.time() - start) * 1000
    print(f"  추론만:       {infer_total/N:.2f}ms")


if __name__ == '__main__':
    main()
