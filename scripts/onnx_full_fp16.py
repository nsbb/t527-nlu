#!/usr/bin/env python3
"""전체 fp16 변환 (I/O는 fp32 유지)
onnxconverter_common.float16.convert_float_to_float16 사용
"""
import onnx
import os, sys, time, json
import numpy as np
import onnxruntime as ort
from onnxconverter_common import float16

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')


def main():
    from transformers import AutoTokenizer
    from model_cnn_multihead import HEAD_I2L

    orig = 'checkpoints/nlu_v28_v46_ensemble.onnx'
    out = 'checkpoints/nlu_v28_v46_ensemble_fp16.onnx'

    print(f"원본: {orig} ({os.path.getsize(orig)/1048576:.1f}MB)")

    model = onnx.load(orig)

    # Convert to fp16, keep I/O as fp32
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )

    onnx.save(model_fp16, out)
    size = os.path.getsize(out) / 1048576
    print(f"FP16: {out} ({size:.1f}MB, {(1 - size / (os.path.getsize(orig)/1048576))*100:.1f}% 감소)")

    # Verify
    sess_fp32 = ort.InferenceSession(orig, providers=['CPUExecutionProvider'])
    sess_fp16 = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    # Test Suite 전체 평가
    suite = json.load(open('data/test_suite.json'))
    match_all = 0
    fn_diff = exec_diff = dir_diff = 0

    for t in suite:
        tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
        input_ids = tk['input_ids'].astype(np.int64)

        o32 = sess_fp32.run(None, {'input_ids': input_ids})
        o16 = sess_fp16.run(None, {'input_ids': input_ids})

        fn32 = HEAD_I2L['fn'][o32[0][0].argmax()]
        fn16 = HEAD_I2L['fn'][o16[0][0].argmax()]
        exec32 = HEAD_I2L['exec_type'][o32[1][0].argmax()]
        exec16 = HEAD_I2L['exec_type'][o16[1][0].argmax()]
        dir32 = HEAD_I2L['param_direction'][o32[2][0].argmax()]
        dir16 = HEAD_I2L['param_direction'][o16[2][0].argmax()]

        if fn32 == fn16 and exec32 == exec16 and dir32 == dir16:
            match_all += 1
        if fn32 != fn16: fn_diff += 1
        if exec32 != exec16: exec_diff += 1
        if dir32 != dir16: dir_diff += 1

    n = len(suite)
    print(f"\n=== Test Suite equivalence ({n}) ===")
    print(f"  All heads match: {match_all}/{n} = {match_all/n*100:.2f}%")
    print(f"  fn diff:   {fn_diff}")
    print(f"  exec diff: {exec_diff}")
    print(f"  dir diff:  {dir_diff}")

    # FP16 재평가 (수정된 라벨 기준)
    fn_ok = exec_ok = dir_ok = all_ok = 0
    for t in suite:
        tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
        outs = sess_fp16.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        fn = HEAD_I2L['fn'][outs[0][0].argmax()]
        exec_t = HEAD_I2L['exec_type'][outs[1][0].argmax()]
        dir_t = HEAD_I2L['param_direction'][outs[2][0].argmax()]
        f = fn == t['fn']; e = exec_t == t['exec']; d = dir_t == t['dir']
        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d: all_ok += 1
    print(f"\n=== FP16 Ensemble Test Suite ===")
    print(f"  fn:    {fn_ok/n*100:.2f}%")
    print(f"  exec:  {exec_ok/n*100:.2f}%")
    print(f"  dir:   {dir_ok/n*100:.2f}%")
    print(f"  combo: {all_ok/n*100:.2f}%")

    # KE
    ke = json.load(open('data/koelectra_converted_val.json'))
    ke_ok = 0
    for d in ke:
        tk = tok(d['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
        outs = sess_fp16.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        if HEAD_I2L['fn'][outs[0][0].argmax()] == d['labels']['fn']:
            ke_ok += 1
    print(f"\n=== FP16 Ensemble KoELECTRA ===")
    print(f"  fn: {ke_ok/len(ke)*100:.2f}%")

    # Latency
    tk = tok("거실 에어컨 23도", padding='max_length', truncation=True, max_length=32, return_tensors='np')
    input_ids = tk['input_ids'].astype(np.int64)
    for _ in range(10): sess_fp16.run(None, {'input_ids': input_ids})

    N = 100
    t0 = time.time()
    for _ in range(N): sess_fp16.run(None, {'input_ids': input_ids})
    lat_fp16 = (time.time() - t0) * 1000 / N

    t0 = time.time()
    for _ in range(N): sess_fp32.run(None, {'input_ids': input_ids})
    lat_fp32 = (time.time() - t0) * 1000 / N

    print(f"\n=== Latency (CPU, 100 avg) ===")
    print(f"  FP32: {lat_fp32:.2f}ms")
    print(f"  FP16: {lat_fp16:.2f}ms")
    print(f"  speedup: {lat_fp32/lat_fp16:.2f}x")


if __name__ == '__main__':
    main()
