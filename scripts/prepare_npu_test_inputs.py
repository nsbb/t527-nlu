#!/usr/bin/env python3
"""실제 문장으로 NPU 검증용 입력+레퍼런스 출력 생성.

플로우:
  text → tokenizer → token_ids[1,32]
       ↓ (full v46 ONNX 실행) → 5-head logits_cpu (정답)
       ↓ (token_emb subgraph) → embedded[1,32,768] float32
       ↓ uint8 quantize (scale=0.003370, zp=155)
       → embed_uint8.bin (24576 bytes) — NPU 입력
       → ref_logits.json — NPU 출력과 비교용

사용:
  python3 prepare_npu_test_inputs.py "거실 불 켜줘"
"""
import sys, os, json
import numpy as np
import onnx
import onnxruntime as ort

CHK = '/home/nsbb/travail/claude/T527/t527-nlu/checkpoints'
FULL_ONNX = f'{CHK}/nlu_v46_generalization.onnx'
BODY_ONNX = f'{CHK}/cnn_body_v46.onnx'
OUT_DIR   = f'{CHK}/npu_test_inputs'

# CNN body 입력 양자화 파라미터 (nbg_meta.json에서)
INPUT_SCALE = 0.003369783
INPUT_ZP    = 155

def tokenize(text):
    """KoELECTRA tokenize → token_ids [1,32]."""
    sys.path.insert(0, '/home/nsbb/travail/claude/T527/t527-nlu/scripts')
    from preprocess import preprocess
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/home/nsbb/travail/claude/T527/t527-nlu/tokenizer/')
    pp = preprocess(text)
    enc = tok(pp, padding='max_length', max_length=32, truncation=True, return_tensors='np')
    return enc['input_ids'].astype(np.int64), pp

def extract_embedded(token_ids):
    """Full ONNX subgraph 0~10번 노드 (token_emb Gather까지) 만 실행."""
    m = onnx.load(FULL_ONNX)
    # Add output for node[10] /m/token_emb/Gather_output_0
    target = '/m/token_emb/Gather_output_0'
    # Build a sub-model that outputs target
    from onnx.utils import Extractor
    ex = Extractor(m)
    sub = ex.extract_model(['token_ids'], [target])
    sess = ort.InferenceSession(sub.SerializeToString(), providers=['CPUExecutionProvider'])
    embedded = sess.run([target], {'token_ids': token_ids})[0]
    return embedded

def run_full(token_ids):
    """전체 v46 추론 → 5-head logits."""
    sess = ort.InferenceSession(FULL_ONNX, providers=['CPUExecutionProvider'])
    outs = sess.run(None, {'token_ids': token_ids})
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outs))

def run_body(embedded):
    """cnn_body_v46.onnx → 5-head logits (참고용)."""
    sess = ort.InferenceSession(BODY_ONNX, providers=['CPUExecutionProvider'])
    outs = sess.run(None, {'embedded': embedded.astype(np.float32)})
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outs))

def quantize_uint8(x, scale, zp):
    q = np.round(x / scale + zp).clip(0, 255).astype(np.uint8)
    return q

def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "거실 불 켜줘"
    os.makedirs(OUT_DIR, exist_ok=True)

    token_ids, pp = tokenize(text)
    print(f"text: {text!r} → preprocessed: {pp!r}")
    print(f"token_ids[:8]: {token_ids[0,:8].tolist()}")

    # 1. Full v46 → 정답 logits
    full_logits = run_full(token_ids)
    print(f"\nFull v46 outputs:")
    for k, v in full_logits.items():
        amax = int(v.argmax(axis=-1).item())
        print(f"  {k}: shape={v.shape}, argmax={amax}, max={v.max():.3f}")

    # 2. Embedded 추출
    embedded = extract_embedded(token_ids)
    print(f"\nembedded: shape={embedded.shape}, dtype={embedded.dtype}")
    print(f"  range: [{embedded.min():.4f}, {embedded.max():.4f}]")
    print(f"  mean/std: {embedded.mean():.4f} / {embedded.std():.4f}")

    # 3. cnn_body로 검증 (full과 같아야 함)
    body_logits = run_body(embedded)
    print(f"\ncnn_body outputs (sanity check):")
    for k, v in body_logits.items():
        amax = int(v.argmax(axis=-1).item())
        print(f"  {k}: argmax={amax}")

    # 4. uint8 quantize
    embed_q = quantize_uint8(embedded, INPUT_SCALE, INPUT_ZP)
    print(f"\nuint8 quantized: shape={embed_q.shape}")
    print(f"  range: [{embed_q.min()}, {embed_q.max()}]")

    # 5. 저장
    safe = ''.join(c if c.isalnum() else '_' for c in text)[:20]
    bin_path  = f'{OUT_DIR}/{safe}_embed_uint8.bin'
    ref_path  = f'{OUT_DIR}/{safe}_ref.json'
    embed_q.tofile(bin_path)

    ref = {
        'text': text,
        'preprocessed': pp,
        'token_ids': token_ids[0].tolist(),
        'input_scale': INPUT_SCALE,
        'input_zp': INPUT_ZP,
        'full_argmax': {k: int(v.argmax(axis=-1).item()) for k, v in full_logits.items()},
        'full_logits': {k: v[0].tolist() for k, v in full_logits.items()},
        'body_argmax': {k: int(v.argmax(axis=-1).item()) for k, v in body_logits.items()},
    }
    with open(ref_path, 'w') as f:
        json.dump(ref, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료:")
    print(f"  {bin_path} ({os.path.getsize(bin_path)} bytes)")
    print(f"  {ref_path}")

if __name__ == '__main__':
    main()
