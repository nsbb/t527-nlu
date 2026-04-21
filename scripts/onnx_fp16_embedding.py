#!/usr/bin/env python3
"""ONNX Ensemble Embedding FP16 변환
Ensemble ONNX (105MB) 중 93.8MB가 embedding (FP32)
→ embedding만 FP16으로 변환 (46.9MB → ~58MB total)
"""
import onnx
import numpy as np
import os, sys, time
import onnxruntime as ort
from onnx import numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')


def convert_embedding_to_fp16(model_path, output_path):
    model = onnx.load(model_path)

    # Find embedding initializers
    emb_names = []
    converted_count = 0
    size_before = 0
    size_after = 0

    new_initializers = []
    for init in model.graph.initializer:
        # Large 2D embedding tensors (vocab × dim)
        if len(init.dims) == 2 and init.dims[0] == 32000 and init.dims[1] == 768:
            # Convert to fp16
            tensor = numpy_helper.to_array(init)
            size_before += tensor.nbytes
            tensor_fp16 = tensor.astype(np.float16)
            size_after += tensor_fp16.nbytes
            # 원본 이름 유지, dtype만 변경
            new_init = numpy_helper.from_array(tensor_fp16, init.name)
            new_initializers.append(new_init)
            emb_names.append(init.name)
            converted_count += 1
            print(f"  변환: {init.name} [{init.dims[0]}x{init.dims[1]}] fp32 → fp16 "
                  f"({tensor.nbytes/1048576:.1f}MB → {tensor_fp16.nbytes/1048576:.1f}MB)")
        else:
            new_initializers.append(init)

    print(f"\n총 {converted_count}개 embedding 변환")
    print(f"  크기: {size_before/1048576:.1f}MB → {size_after/1048576:.1f}MB ({(1-size_after/size_before)*100:.0f}% 감소)")

    # Graph에서 embedding 뒤에 Cast(fp16→fp32) 삽입 필요
    # — 일반 Gather op가 fp16 테이블 사용 시 자동 처리되는지 확인
    # onnxruntime이 자동 처리 못하면 명시적 Cast 필요

    # 간단한 방법: 그대로 시도 후 실패 시 Cast 추가
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    # 검증
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    fsize = os.path.getsize(output_path) / 1048576
    print(f"\n저장: {output_path} ({fsize:.1f}MB)")

    return emb_names


def verify_equivalence(original_path, fp16_path, n_samples=50):
    """ensemble ONNX 변환 전/후 출력 비교"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    try:
        sess_orig = ort.InferenceSession(original_path, providers=['CPUExecutionProvider'])
        sess_fp16 = ort.InferenceSession(fp16_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"FP16 모델 로드 실패: {e}")
        return False

    import json
    suite = json.load(open('data/test_suite.json'))[:n_samples]

    match = 0
    max_diff = 0
    for t in suite:
        tk = tok(t['utterance'], padding='max_length', truncation=True, max_length=32, return_tensors='np')
        input_ids = tk['input_ids'].astype(np.int64)
        outs_orig = sess_orig.run(None, {'input_ids': input_ids})
        outs_fp16 = sess_fp16.run(None, {'input_ids': input_ids})

        # Compare argmax
        orig_preds = [o[0].argmax() for o in outs_orig]
        fp16_preds = [o[0].argmax() for o in outs_fp16]
        if orig_preds == fp16_preds:
            match += 1

        # Max diff
        for oo, of in zip(outs_orig, outs_fp16):
            diff = np.abs(oo - of).max()
            max_diff = max(max_diff, diff)

    print(f"\n=== Equivalence check ({n_samples} samples) ===")
    print(f"  argmax 일치: {match}/{n_samples} ({match/n_samples*100:.0f}%)")
    print(f"  max output diff: {max_diff:.6f}")
    return match == n_samples


def main():
    orig = 'checkpoints/nlu_v28_v46_ensemble.onnx'
    out = 'checkpoints/nlu_v28_v46_ensemble_fp16emb.onnx'

    print(f"원본: {orig} ({os.path.getsize(orig)/1048576:.1f}MB)")
    print()

    emb_names = convert_embedding_to_fp16(orig, out)

    # Verify
    print()
    ok = verify_equivalence(orig, out)

    if ok:
        print(f"\n✅ 변환 성공 — 정확도 유지")
    else:
        print(f"\n⚠️ 정확도 차이 있음 — Cast 노드 추가 필요")


if __name__ == '__main__':
    main()
