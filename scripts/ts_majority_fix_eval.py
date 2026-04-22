#!/usr/bin/env python3
"""TS Label 일관성 수정 후 성능 재측정.

data/ts_label_fix_candidates.json 의 minority-labeled 케이스를 majority label로 바꾸고,
Ensemble + rules 성능을 원본 TS vs 수정 TS로 비교.

목적: 얼마나 많은 '오류'가 라벨 문제였는지 정량화.
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import onnxruntime as ort, numpy as np
from transformers import AutoTokenizer
from model_cnn_multihead import HEAD_I2L
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess


def main():
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx', providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    test_orig = json.load(open('data/test_suite.json'))
    fixes = json.load(open('data/ts_label_fix_candidates.json'))

    # Build utterance → majority label map
    fix_map = {f['utterance']: f['majority'] for f in fixes}
    print(f"Fix candidates: {len(fix_map)}")

    # Apply fixes
    test_fixed = []
    for x in test_orig:
        x2 = dict(x)
        if x['utterance'] in fix_map:
            fix = fix_map[x['utterance']]
            x2['fn'] = fix['fn']
            x2['exec'] = fix['exec']
            x2['dir'] = fix['dir']
        test_fixed.append(x2)

    # Evaluate both
    def eval_ts(test_list, name):
        fn_ok = exec_ok = dir_ok = all_ok = 0
        for item in test_list:
            text = preprocess(item['utterance'])
            p = predict_with_rules(text, sess, tok)
            f = p['fn'] == item['fn']
            e = p['exec_type'] == item['exec']
            d = p['param_direction'] == item['dir']
            if f: fn_ok += 1
            if e: exec_ok += 1
            if d: dir_ok += 1
            if f and e and d: all_ok += 1
        n = len(test_list)
        print(f"\n{name}:")
        print(f"  fn:    {fn_ok/n*100:.2f}%")
        print(f"  exec:  {exec_ok/n*100:.2f}%")
        print(f"  dir:   {dir_ok/n*100:.2f}%")
        print(f"  combo: {all_ok/n*100:.2f}% ({all_ok}/{n})")
        return all_ok

    orig = eval_ts(test_orig, "원본 TS")
    fixed = eval_ts(test_fixed, "수정 TS (majority vote)")

    print(f"\n=== 결론 ===")
    print(f"원본 TS: {orig}/3043 정답 ({orig/3043*100:.2f}%)")
    print(f"수정 TS: {fixed}/3043 정답 ({fixed/3043*100:.2f}%)")
    print(f"라벨 일관성 수정만으로 {(fixed-orig)/3043*100:+.2f}%p")


if __name__ == '__main__':
    main()
