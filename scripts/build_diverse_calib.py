#!/usr/bin/env python3
"""다양한 발화 → embedding npy 생성 (NPU calib용).

소스: 학습 데이터 + 골든셋 + 르엘 시나리오 → unique 발화 추출 → embedding npy 저장.
"""
import sys, os, json, random
import numpy as np
import onnx, onnxruntime as ort
from onnx.utils import Extractor

CHK = '/home/nsbb/travail/claude/T527/t527-nlu/checkpoints'
TOK_PATH = '/home/nsbb/travail/claude/T527/t527-nlu/tokenizer/'
DATA = '/home/nsbb/travail/claude/T527/t527-nlu/data'
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else f'{CHK}/calib_diverse'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300

def collect_utterances():
    utts = set()
    # Golden test 100/500
    for path in [f'{DATA}/golden_test_100.json', f'{DATA}/golden_test_500.json',
                 f'{DATA}/golden/golden_indirect_55.json', f'{DATA}/golden/gt_known_scenarios_v2.json']:
        if not os.path.exists(path): continue
        d = json.load(open(path))
        if isinstance(d, list):
            for it in d:
                u = it.get('utterance') or it.get('text') or it.get('utt')
                if u: utts.add(u.strip())
    # Training-augmented (if exists)
    for path in [f'{DATA}/train_gt_augmented.json', f'{DATA}/train_gt_augmented_v2.json']:
        if not os.path.exists(path): continue
        try:
            d = json.load(open(path))
            if isinstance(d, list):
                for it in d:
                    if isinstance(it, dict):
                        u = it.get('utterance') or it.get('text')
                        if u: utts.add(u.strip())
        except Exception as e:
            print(f'  skip {path}: {e}')
    print(f'collected {len(utts)} unique utterances')
    return list(utts)

def main():
    sys.path.insert(0, '/home/nsbb/travail/claude/T527/t527-nlu/scripts')
    from preprocess import preprocess
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_PATH)

    full = onnx.load(f'{CHK}/nlu_v46_generalization.onnx')
    sub = Extractor(full).extract_model(['token_ids'], ['/m/token_emb/Gather_output_0'])
    sess = ort.InferenceSession(sub.SerializeToString(), providers=['CPUExecutionProvider'])

    utts = collect_utterances()
    random.seed(42)
    random.shuffle(utts)
    sel = utts[:N]
    print(f'selected {len(sel)} for calib')

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, u in enumerate(sel):
        enc = tok(preprocess(u), padding='max_length', max_length=32, truncation=True, return_tensors='np')
        emb = sess.run(None, {'token_ids': enc['input_ids'].astype(np.int64)})[0].astype(np.float32)
        np.save(f'{OUT_DIR}/calib_{i:04d}.npy', emb)
    # dataset list
    with open(f'{OUT_DIR}/dataset.txt', 'w') as f:
        for i in range(len(sel)):
            f.write(f'./calib_{i:04d}.npy\n')
    # store text mapping for traceability
    with open(f'{OUT_DIR}/utterances.txt', 'w') as f:
        for i, u in enumerate(sel):
            f.write(f'{i:04d}\t{u}\n')
    sample = np.load(f'{OUT_DIR}/calib_0000.npy')
    print(f'saved {len(sel)} npy in {OUT_DIR}, sample shape={sample.shape}')

if __name__ == '__main__':
    main()
