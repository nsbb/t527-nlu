#!/usr/bin/env python3
"""Dir-aware Ensemble — dir을 내용에 따라 v28/v46 선택

발견: v28은 밝게→down, 냉방모드→on 등 잘못 학습 (옛 라벨)
      v46은 같은 케이스에서 정확 (pseudo-labeled data로 교정)
→ 해당 패턴 감지되면 dir을 v46에서 가져오기
"""
import os, sys, json, re, torch, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


# v46의 dir이 v28보다 나은 패턴들 (실증으로 발견)
V46_DIR_PATTERNS = [
    r'밝게',               # 밝게→up
    r'어둡게',             # 어둡게→down
    r'모드로',             # N모드로→set
    r'제습',               # 제습 모드→set
    r'냉방',               # 냉방 모드→set
    r'자동.*모드',         # 자동 모드→set
    r'풍량',               # 풍량 조절
    r'타이머',             # 타이머 관련
]

def should_use_v46_dir(text):
    for pat in V46_DIR_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def main():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach(); del sbert
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m28.load_state_dict(torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)['state'])
    m28.eval()
    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m46.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)['state'])
    m46.eval()

    suite = json.load(open('data/test_suite.json'))
    ke = json.load(open('data/koelectra_converted_val.json'))

    strategies = {
        'B (baseline)': {'fn': 'v46', 'exec': 'v28', 'dir_default': 'v28', 'dir_override': None},
        'B + dir-aware': {'fn': 'v46', 'exec': 'v28', 'dir_default': 'v28', 'dir_override': 'v46_patterns'},
        'B2 (dir=v46)': {'fn': 'v46', 'exec': 'v28', 'dir_default': 'v46', 'dir_override': None},
    }

    # 예측 한 번만
    l28_all, l46_all = [], []
    for t in suite:
        tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
        with torch.no_grad():
            o28 = m28(tk['input_ids'])
            o46 = m46(tk['input_ids'])
        l28_all.append({h: o28[h] for h in HEAD_NAMES})
        l46_all.append({h: o46[h] for h in HEAD_NAMES})

    def get(l, h):
        return HEAD_I2L[h][l[h].argmax(1).item()]

    print(f"\n{'Strategy':<25} {'fn':>8} {'exec':>8} {'dir':>8} {'combo':>8}")
    print('-'*60)

    for name, strat in strategies.items():
        fn_ok = exec_ok = dir_ok = all_ok = 0
        override_count = 0
        for i, t in enumerate(suite):
            l28, l46 = l28_all[i], l46_all[i]
            fn = get(l46 if strat['fn'] == 'v46' else l28, 'fn')
            ex = get(l28 if strat['exec'] == 'v28' else l46, 'exec_type')

            if strat['dir_override'] == 'v46_patterns' and should_use_v46_dir(t['utterance']):
                d = get(l46, 'param_direction')
                override_count += 1
            else:
                d = get(l28 if strat['dir_default'] == 'v28' else l46, 'param_direction')

            if fn == t['fn']: fn_ok += 1
            if ex == t['exec']: exec_ok += 1
            if d == t['dir']: dir_ok += 1
            if fn == t['fn'] and ex == t['exec'] and d == t['dir']: all_ok += 1

        n = len(suite)
        print(f"{name:<25} {fn_ok/n*100:>7.2f}% {exec_ok/n*100:>7.2f}% {dir_ok/n*100:>7.2f}% {all_ok/n*100:>7.2f}%"
              + (f"  (override {override_count})" if override_count else ""))

    # KE
    print(f"\nKoELECTRA fn:")
    for name, strat in strategies.items():
        ok = 0
        for d in ke:
            tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='pt')
            with torch.no_grad():
                o46 = m46(tk['input_ids'])
            fn = HEAD_I2L['fn'][o46['fn'].argmax(1).item()]
            if fn == d['labels']['fn']: ok += 1
        print(f"  {name}: {ok/len(ke)*100:.2f}%")


if __name__ == '__main__':
    main()
