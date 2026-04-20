#!/usr/bin/env python3
"""확장된 Test Suite (3,109개)로 성능 재평가"""
import os, sys, json, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from preprocess import preprocess
from transformers import AutoTokenizer

import onnxruntime as ort

def predict(text, sess, tok):
    clean = preprocess(text)
    tk = tok(clean, padding='max_length', truncation=True, max_length=32, return_tensors='np')
    outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
    p = {
        'fn': HEAD_I2L['fn'][outs[0][0].argmax()],
        'exec_type': HEAD_I2L['exec_type'][outs[1][0].argmax()],
        'param_direction': HEAD_I2L['param_direction'][outs[2][0].argmax()],
        'param_type': HEAD_I2L['param_type'][outs[3][0].argmax()],
        'judge': HEAD_I2L['judge'][outs[4][0].argmax()],
    }
    if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
    if p['judge'] != 'none': p['param_type'] = 'none'
    if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'
    return p

def main():
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                providers=['CPUExecutionProvider'])

    suite = json.load(open('data/test_suite_v67.json'))
    print(f"Total: {len(suite)}")

    # Overall + by source
    sources = {}
    fn_ok = exec_ok = dir_ok = all_ok = 0
    errs = []
    for t in suite:
        p = predict(t['utterance'], sess, tok)
        src = t.get('source', 'original')
        if src not in sources:
            sources[src] = {'n': 0, 'fn': 0, 'exec': 0, 'dir': 0, 'all': 0, 'err': []}
        sources[src]['n'] += 1

        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1; sources[src]['fn'] += 1
        if e: exec_ok += 1; sources[src]['exec'] += 1
        if d: dir_ok += 1; sources[src]['dir'] += 1
        if f and e and d: all_ok += 1; sources[src]['all'] += 1
        elif len(sources[src]['err']) < 5:
            det = []
            if not f: det.append(f"fn:{t['fn']}→{p['fn']}")
            if not e: det.append(f"exec:{t['exec']}→{p['exec_type']}")
            if not d: det.append(f"dir:{t['dir']}→{p['param_direction']}")
            sources[src]['err'].append(f"  \"{t['utterance']}\" {' '.join(det)}")

    n = len(suite)
    print(f"\n=== 확장 Test Suite 전체 (Ensemble ONNX + preprocess) ===")
    print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.2f}%")
    print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.2f}%")
    print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.2f}%")
    print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.2f}%")

    print(f"\n=== Source별 ===")
    for src in sorted(sources):
        s = sources[src]
        print(f"  {src} ({s['n']}): fn={s['fn']/s['n']*100:.1f}% combo={s['all']/s['n']*100:.1f}%")
        if s['err']:
            for e in s['err'][:3]:
                print(f"     오류: {e.strip()}")

if __name__ == '__main__':
    main()
