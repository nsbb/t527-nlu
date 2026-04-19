#!/usr/bin/env python3
"""앙상블 추론 — v28(기존) + v34(일반화) 결합
전략 B: fn=v34, exec/dir=v28
사용법:
    python3 scripts/ensemble_inference.py "거실 에어컨 23도로 맞춰줘"
    python3 scripts/ensemble_inference.py  # 대화형
"""
import torch, torch.nn.functional as F, json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from preprocess import preprocess
from transformers import AutoModel, AutoTokenizer

class EnsemblePipeline:
    def __init__(self):
        print("앙상블 로딩 중 (v28 + v34)...")
        sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        pw = sbert.embeddings.word_embeddings.weight.detach()
        del sbert

        self.m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
        self.m28.load_state_dict(torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)['state'])
        self.m28.eval()

        self.m34 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
        self.m34.load_state_dict(torch.load('checkpoints/cnn_multihead_v34.pt', map_location='cpu', weights_only=False)['state'])
        self.m34.eval()

        self.tok = AutoTokenizer.from_pretrained('tokenizer/')
        print("로딩 완료 — 앙상블 전략 B (fn=v34, exec/dir=v28)")

    def predict(self, text):
        text = preprocess(text)
        if not text:
            return {h: 'none' for h in HEAD_NAMES}, 0

        tk = self.tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')

        with torch.no_grad():
            l28 = self.m28(tk['input_ids'])
            l34 = self.m34(tk['input_ids'])

        p28 = {h: HEAD_I2L[h][l28[h].argmax(1).item()] for h in HEAD_NAMES}
        p34 = {h: HEAD_I2L[h][l34[h].argmax(1).item()] for h in HEAD_NAMES}
        c34 = F.softmax(l34['fn'], dim=1).max().item()

        # 전략 B: fn=v34, exec/dir=v28
        result = {
            'fn': p34['fn'],
            'exec_type': p28['exec_type'],
            'param_direction': p28['param_direction'],
            'param_type': p28['param_type'],
            'judge': p34['judge'],
        }

        # param_type 규칙 보정
        if result['param_direction'] in ('open', 'close', 'stop'):
            result['param_type'] = 'none'
        if result['judge'] != 'none':
            result['param_type'] = 'none'
        if result['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
            result['param_type'] = 'none'

        return result, c34


if __name__ == '__main__':
    ens = EnsemblePipeline()

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        p, conf = ens.predict(text)
        print(f"\n입력: {text}")
        for h in HEAD_NAMES:
            print(f"  {h}: {p[h]}")
        print(f"  conf: {conf:.2f}")
    else:
        print("\n=== 앙상블 대화형 테스트 ===\n")
        while True:
            try:
                text = input("사용자: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ('q', 'quit', 'exit', '종료'):
                break
            if not text:
                continue
            p, conf = ens.predict(text)
            parts = [f"fn={p['fn']}"]
            if p['exec_type'] != 'none': parts.append(f"exec={p['exec_type']}")
            if p['param_direction'] != 'none': parts.append(f"dir={p['param_direction']}")
            if p['judge'] != 'none': parts.append(f"judge={p['judge']}")
            parts.append(f"conf={conf:.2f}")
            print(f"  → {', '.join(parts)}")
            print()
