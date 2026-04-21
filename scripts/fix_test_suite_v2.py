#!/usr/bin/env python3
"""test_suite.json 라벨 오류 수정 v2
v2 탐지 결과 + 수동 검토 기반 확실한 것만 수정
"""
import json, os, shutil, sys, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

# Category A — 매우 확실한 오류 (모델 + 규칙 일치)
FIXES_CATEGORY_A = [
    ('남방 올려쥬', {'dir': 'up'}),
    ('날방 올려', {'dir': 'up'}),
    ('남방 올려', {'dir': 'up'}),
    ('도어록 열어쥬', {'dir': 'open'}),
    ('안방 남방 올려줘', {'dir': 'up'}),
    ('안방 난반 올려', {'dir': 'up'}),
]

# Category B — 규칙 기반 확실 오류
FIXES_CATEGORY_B = [
    # 밝게 → up (down으로 잘못 라벨됨)
    ('거실 등 좀 밝게', {'dir': 'up'}),
    ('안방 등 밝게', {'dir': 'up'}),
    ('주방 불 좀 밝게', {'dir': 'up'}),
    ('작은방 좀 밝게', {'dir': 'up'}),
    # 블라인드/커튼 올려 → up (stop은 '멈춰'일 때만)
    ('블라인드 올려', {'dir': 'up'}),
    ('커튼 올려', {'dir': 'up'}),
    ('커튼 올려줘', {'dir': 'up'}),
    ('블라인드 올려줘', {'dir': 'up'}),
    # 커튼 내려 → down
    ('커튼 내려줘', {'dir': 'down'}),
    # 수치 질의 → exec=query
    ('몇 시야?', {'exec': 'query_then_respond'}),
    ('환율 얼마야', {'exec': 'query_then_respond'}),
    ('택시비 얼마야', {'exec': 'query_then_respond'}),
    ('바깥 온도 몇 도', {'exec': 'query_then_respond'}),
    ('기온이 몇 도야', {'exec': 'query_then_respond'}),
    ('난방 온도 몇 도야', {'exec': 'query_then_respond'}),
    # 엘베/승강기 내려와 → dir=down (위에서 내려오라는 뜻은 아닐 수도 있지만 모델이 none 예측하니 확인)
    # → 이건 Cat C이고 모델은 none, 라벨은 on → 실제로 "층 제어 없이 호출"이라 on도 애매. skip.
]

# Category C — 모델만 의심 (검토 후 확실한 것)
FIXES_CATEGORY_C = [
    # 꺼/끄 패턴 중 on으로 잘못 라벨된 것
    ('남방 꺼쥬', {'dir': 'off'}),
    ('에어컨꺼', {'dir': 'off'}),
    ('히팅 켜줘', {'dir': 'on'}),  # up → on
]


def main():
    path = 'data/test_suite.json'
    backup = 'data/test_suite_before_fix_v2.json'

    shutil.copy(path, backup)
    print(f"백업: {backup}")

    suite = json.load(open(path))
    n_fixed = 0

    all_fixes = [('A', f) for f in FIXES_CATEGORY_A] + \
                [('B', f) for f in FIXES_CATEGORY_B] + \
                [('C', f) for f in FIXES_CATEGORY_C]

    for cat, (utt, fix) in all_fixes:
        for t in suite:
            if t['utterance'] == utt:
                changes = []
                for k, v in fix.items():
                    if t[k] != v:
                        changes.append(f"{k}: {t[k]} → {v}")
                        t[k] = v
                if changes:
                    n_fixed += 1
                    print(f"  [{cat}] \"{utt}\" {', '.join(changes)}")
                break

    print(f"\n총 수정: {n_fixed}건")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)

    # 재평가 — Ensemble
    import numpy as np, onnxruntime as ort
    from model_cnn_multihead import HEAD_I2L
    from transformers import AutoTokenizer

    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    suite = json.load(open(path))
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
    print(f"\n=== Ensemble 재평가 ===")
    print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.2f}% (이전 98.0%)")
    print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.2f}% (이전 98.2%)")
    print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.2f}% (이전 97.8%)")
    print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.2f}% (이전 94.3%)")


if __name__ == '__main__':
    main()
