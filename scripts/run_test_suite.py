#!/usr/bin/env python3
"""테스트 스위트 실행 — data/test_suite.json 기준
사용법: python3 scripts/run_test_suite.py [v21|v23|checkpoint_path]
"""
import torch, torch.nn.functional as F, json, os, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')
from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer

# 모델 선택
ckpt_path = 'checkpoints/cnn_multihead_v25.pt'
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if os.path.exists(arg):
        ckpt_path = arg
    elif os.path.exists(f'checkpoints/cnn_multihead_{arg}.pt'):
        ckpt_path = f'checkpoints/cnn_multihead_{arg}.pt'

print(f"모델: {ckpt_path}")
sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert
model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state']); model.eval()
tok = AutoTokenizer.from_pretrained('tokenizer/')

def pred(text):
    text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in text)).strip()
    t = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad(): l = model(t['input_ids'])
    p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
    # param_type 규칙 보정
    if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
    if p['judge'] != 'none': p['param_type'] = 'none'
    if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): p['param_type'] = 'none'
    return p

# 테스트 스위트 로드
with open('data/test_suite.json') as f:
    suite = json.load(f)

fn_ok = exec_ok = dir_ok = all_ok = 0
errs = []

for t in suite:
    p = pred(t['utterance'])
    f = p['fn'] == t['fn']
    e = p['exec_type'] == t['exec']
    d = p['param_direction'] == t['dir']
    if f: fn_ok += 1
    if e: exec_ok += 1
    if d: dir_ok += 1
    if f and e and d: all_ok += 1
    if not (f and e and d):
        det = []
        if not f: det.append(f"fn:{t['fn']}→{p['fn']}")
        if not e: det.append(f"exec:{t['exec']}→{p['exec_type']}")
        if not d: det.append(f"dir:{t['dir']}→{p['param_direction']}")
        errs.append(f"  \"{t['utterance']}\" {' '.join(det)}")

n = len(suite)
print(f"\n=== Test Suite ({n}개) ===")
print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.1f}%")
print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.1f}%")
print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.1f}%")
print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.1f}%")

if errs:
    print(f"\n오류 {len(errs)}개:")
    for e in errs: print(e)
else:
    print("\n  오류 없음!")
