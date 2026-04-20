#!/usr/bin/env python3
"""test_suite.json의 라벨 오류 11건 수정
백업 후 수정, 수정 전/후 성능 비교
"""
import json, os, sys, shutil, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

# 수정 목록 (확실한 오류만)
FIXES = [
    ('커턴 닫아', {'dir': 'close'}),  # was open
    ('난방꺼줘', {'dir': 'off'}),  # was on
    ('환기꺼줘', {'dir': 'off'}),  # was on
    ('주방 남방 꺼줘', {'dir': 'off'}),  # was on
    ('주차 등록해줘', {'dir': 'set'}),  # was none
    ('방문 주차 등록해줘', {'dir': 'set'}),  # was none
    ('승강기 불러줘', {'exec': 'control_then_confirm'}),  # was query
    ('리프트 호출', {'exec': 'control_then_confirm'}),  # was query
    ('승강기 호출해줘', {'exec': 'control_then_confirm'}),  # was query
    ('승강기불러', {'exec': 'control_then_confirm'}),  # was query
    ('승강기 호출해줘봐', {'exec': 'control_then_confirm'}),  # was query
]

def main():
    path = 'data/test_suite.json'
    backup = 'data/test_suite_before_fix.json'

    # Backup
    shutil.copy(path, backup)
    print(f"백업 생성: {backup}")

    suite = json.load(open(path))
    n_fixed = 0
    for utt, fix in FIXES:
        for t in suite:
            if t['utterance'] == utt:
                changes = []
                for k, v in fix.items():
                    if t[k] != v:
                        changes.append(f"{k}: {t[k]} → {v}")
                        t[k] = v
                if changes:
                    n_fixed += 1
                    print(f"  수정: \"{utt}\" ({', '.join(changes)})")
                break

    print(f"\n총 수정: {n_fixed}건")

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)
    print(f"저장: {path}")

    # 성능 재측정
    print(f"\n=== 수정된 test_suite로 앙상블 재평가 ===")

    import onnxruntime as ort
    from transformers import AutoTokenizer
    from model_cnn_multihead import HEAD_I2L

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
    print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.2f}% (이전 98.0%)")
    print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.2f}% (이전 98.2%)")
    print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.2f}% (이전 97.8%)")
    print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.2f}% (이전 94.3%)")


if __name__ == '__main__':
    main()
