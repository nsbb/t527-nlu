#!/usr/bin/env python3
"""Ensemble ONNX INT8 Dynamic Quantization
- 104.9MB FP32 → ~27MB INT8
- CPU 추론 속도 2-4x 향상
- 정확도 손실 최소화 (dynamic quantization은 weight만 양자화)
"""
import os, sys, time, json, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from transformers import AutoTokenizer


def main():
    fp32_path = 'checkpoints/nlu_v28_v46_ensemble.onnx'
    int8_path = 'checkpoints/nlu_v28_v46_ensemble_int8.onnx'

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    print(f"FP32 size: {fp32_size:.1f}MB")

    # Dynamic Quantization (INT8, weight only)
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Quantizing to INT8...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Gemm'],
    )

    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"INT8 size: {int8_size:.1f}MB (reduction: {(1 - int8_size/fp32_size)*100:.1f}%)")

    # Verify accuracy
    import onnxruntime as ort
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    sess_fp32 = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])

    # Test Suite eval (both)
    suite = json.load(open('data/test_suite.json'))

    def eval_sess(sess, name):
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
        print(f"  {name}: fn={fn_ok/n*100:.1f}% exec={exec_ok/n*100:.1f}% dir={dir_ok/n*100:.1f}% combo={all_ok/n*100:.1f}%")
        return all_ok/n*100

    print(f"\n=== Test Suite 정확도 비교 ===")
    fp32_combo = eval_sess(sess_fp32, "FP32")
    int8_combo = eval_sess(sess_int8, "INT8")
    drop = fp32_combo - int8_combo
    print(f"  combo 차이: {drop:+.2f}%p")

    # KoELECTRA
    ke_val = json.load(open('data/koelectra_converted_val.json'))
    def eval_ke(sess, name):
        fn_ok = 0
        for d in ke_val:
            tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
            outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
            if HEAD_I2L['fn'][outs[0][0].argmax()] == d['labels']['fn']:
                fn_ok += 1
        acc = fn_ok / len(ke_val) * 100
        print(f"  {name}: fn={acc:.1f}%")
        return acc

    print(f"\n=== KoELECTRA 정확도 비교 ===")
    fp32_ke = eval_ke(sess_fp32, "FP32")
    int8_ke = eval_ke(sess_int8, "INT8")
    print(f"  fn 차이: {fp32_ke - int8_ke:+.2f}%p")

    # Latency
    print(f"\n=== Latency 비교 (CPU, 100회 avg) ===")
    tk = tok("거실 에어컨 23도로 맞춰줘", padding='max_length', truncation=True, max_length=32, return_tensors='np')
    input_ids = tk['input_ids'].astype(np.int64)

    # Warmup
    for _ in range(10):
        sess_fp32.run(None, {'input_ids': input_ids})
        sess_int8.run(None, {'input_ids': input_ids})

    N = 100
    start = time.time()
    for _ in range(N): sess_fp32.run(None, {'input_ids': input_ids})
    fp32_ms = (time.time() - start) * 1000 / N

    start = time.time()
    for _ in range(N): sess_int8.run(None, {'input_ids': input_ids})
    int8_ms = (time.time() - start) * 1000 / N

    print(f"  FP32: {fp32_ms:.2f}ms")
    print(f"  INT8: {int8_ms:.2f}ms (speedup: {fp32_ms/int8_ms:.2f}x)")

    print(f"\n=== 최종 요약 ===")
    print(f"  크기: {fp32_size:.1f}MB → {int8_size:.1f}MB ({(1-int8_size/fp32_size)*100:.1f}% 감소)")
    print(f"  속도: {fp32_ms:.2f}ms → {int8_ms:.2f}ms ({fp32_ms/int8_ms:.2f}x)")
    print(f"  TS combo: {fp32_combo:.1f}% → {int8_combo:.1f}% ({drop:+.2f}%p)")
    print(f"  KE fn:    {fp32_ke:.1f}% → {int8_ke:.1f}% ({fp32_ke-int8_ke:+.2f}%p)")


if __name__ == '__main__':
    main()
