#!/usr/bin/env python3
"""v67 — 통합 검증: Ensemble ONNX + 확장 STT 전처리(120개) + 후처리 규칙
실제 end-to-end 성능 측정
"""
import os, sys, json, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from preprocess import preprocess
from transformers import AutoTokenizer

import onnxruntime as ort


def predict_full_pipeline(text, sess, tok):
    """End-to-end: preprocess → tokenize → ONNX → 후처리"""
    # 1. STT 전처리
    clean = preprocess(text)

    # 2. 토큰화
    tk = tok(clean, padding='max_length', truncation=True, max_length=32, return_tensors='np')

    # 3. 앙상블 추론
    outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})

    # 4. 디코딩
    p = {
        'fn': HEAD_I2L['fn'][outs[0][0].argmax()],
        'exec_type': HEAD_I2L['exec_type'][outs[1][0].argmax()],
        'param_direction': HEAD_I2L['param_direction'][outs[2][0].argmax()],
        'param_type': HEAD_I2L['param_type'][outs[3][0].argmax()],
        'judge': HEAD_I2L['judge'][outs[4][0].argmax()],
    }

    # 5. 후처리 규칙
    if p['param_direction'] in ('open', 'close', 'stop'):
        p['param_type'] = 'none'
    if p['judge'] != 'none':
        p['param_type'] = 'none'
    if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
        p['param_type'] = 'none'

    # 6. Confidence fallback (from fn logits)
    import scipy.special
    fn_probs = scipy.special.softmax(outs[0][0])
    conf = float(fn_probs.max())
    if conf < 0.5 and p['fn'] != 'unknown':
        p['fn'] = 'unknown'
        p['exec_type'] = 'direct_respond'

    return p, clean, conf


def main():
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                providers=['CPUExecutionProvider'])

    # Test Suite
    suite = json.load(open('data/test_suite.json'))
    fn_ok = exec_ok = dir_ok = all_ok = 0
    preprocessed_diff = 0
    for t in suite:
        p, clean, conf = predict_full_pipeline(t['utterance'], sess, tok)
        if clean != t['utterance']:
            preprocessed_diff += 1
        f = p['fn'] == t['fn']; e = p['exec_type'] == t['exec']; d = p['param_direction'] == t['dir']
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1

    n = len(suite)
    print(f"=== v67 통합 파이프라인 Test Suite ===")
    print(f"  전처리 변경됨: {preprocessed_diff}/{n} ({preprocessed_diff/n*100:.1f}%)")
    print(f"  fn:    {fn_ok}/{n} = {fn_ok/n*100:.1f}%")
    print(f"  exec:  {exec_ok}/{n} = {exec_ok/n*100:.1f}%")
    print(f"  dir:   {dir_ok}/{n} = {dir_ok/n*100:.1f}%")
    print(f"  combo: {all_ok}/{n} = {all_ok/n*100:.1f}%")

    # KoELECTRA
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    ke_ok = 0
    for d in ke_val:
        p, clean, conf = predict_full_pipeline(d['utterance'], sess, tok)
        if p['fn'] == d['labels']['fn']:
            ke_ok += 1

    print(f"\n=== v67 통합 파이프라인 KoELECTRA ===")
    print(f"  fn: {ke_ok}/{len(ke_val)} = {ke_ok/len(ke_val)*100:.1f}%")

    # STT 오류 내성 테스트
    stt_tests = [
        ("에어콘 켜줘", "ac_control"),
        ("뉴슈 틀어줘", "news_query"),
        ("거실에어컨 꺼", "ac_control"),
        ("씨원하게 해줘", "ac_control"),
        ("몇시야", "home_info"),
        ("도어렉 열어", "door_control"),
        ("미세문지 어때", "weather_query"),
        ("남방 켜쥬", "heat_control"),
        ("에어컨꺼", "ac_control"),
        ("오늘날씨어때", "weather_query"),
    ]

    print(f"\n=== STT 오류 내성 테스트 ===")
    stt_ok = 0
    for text, exp_fn in stt_tests:
        p, clean, conf = predict_full_pipeline(text, sess, tok)
        ok = "✓" if p['fn'] == exp_fn else "✗"
        if p['fn'] == exp_fn: stt_ok += 1
        print(f"  {ok} \"{text}\" → \"{clean}\" → fn={p['fn']} (exp={exp_fn}) conf={conf:.2f}")
    print(f"  STT 내성: {stt_ok}/{len(stt_tests)} = {stt_ok/len(stt_tests)*100:.1f}%")


if __name__ == '__main__':
    main()
