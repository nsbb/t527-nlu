#!/usr/bin/env python3
"""Error Severity Analysis — 모든 error가 동등하지 않다.

기존 metric은 binary: correct vs incorrect.
실제 user 영향 관점에서는:
- 🔴 Catastrophic: 의도와 반대 행동 (on→off, heat→ac)
- 🟡 Annoying: 잘못 이해해 다시 말해야 함 (control→query)
- 🟢 Graceful: unknown 거부 (옳은 rejection)

Analysis: 현재 129 errors 중 severity 분포.
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort, numpy as np, json, re
from transformers import AutoTokenizer
from model_cnn_multihead import HEAD_I2L
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess


def severity(gt, pred):
    """Error severity 판정."""
    # 1. Catastrophic: 반대 direction (on↔off, open↔close, up↔down)
    opposite_dir = [('on','off'), ('off','on'), ('open','close'), ('close','open'),
                     ('up','down'), ('down','up')]
    if (gt['dir'], pred['param_direction']) in opposite_dir:
        return 'catastrophic'

    # 2. Catastrophic: 다른 device 조작 (heat↔ac, light↔curtain 같은 제어)
    catastrophic_fn_swaps = [
        ('heat_control', 'ac_control'), ('ac_control', 'heat_control'),
        ('light_control', 'curtain_control'), ('curtain_control', 'light_control'),
        ('gas_control', 'door_control'),  # 보안 관련
    ]
    if (gt['fn'], pred['fn']) in catastrophic_fn_swaps:
        return 'catastrophic'

    # 3. Graceful: unknown fn 맞춰서 거부
    if gt['fn'] == 'unknown' and pred['fn'] == 'unknown':
        return 'graceful'

    # 4. Annoying: control ↔ query 오분류
    if gt['exec'] in ('control_then_confirm',) and pred['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify'):
        return 'annoying_query_as_control'
    if gt['exec'] in ('query_then_respond',) and pred['exec_type'] == 'control_then_confirm':
        return 'annoying_control_as_query'

    # 5. Minor: 같은 fn + 같은 exec + dir만 약간 다름 (none ↔ 구체적 direction)
    if gt['fn'] == pred['fn'] and gt['exec'] == pred['exec_type']:
        if gt['dir'] == 'none' or pred['param_direction'] == 'none':
            return 'minor_dir_only'

    return 'other'


def main():
    print("Loading...")
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx', providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    test = json.load(open('data/test_suite.json'))
    print(f"Loaded {len(test)} TS cases.\n")

    from collections import Counter
    severity_counts = Counter()
    severity_examples = {}

    for item in test:
        text = preprocess(item['utterance'])
        p = predict_with_rules(text, sess, tok)
        f = p['fn'] == item['fn']; e = p['exec_type'] == item['exec']; d = p['param_direction'] == item['dir']
        if not (f and e and d):
            sev = severity(item, p)
            severity_counts[sev] += 1
            if sev not in severity_examples:
                severity_examples[sev] = []
            if len(severity_examples[sev]) < 3:
                severity_examples[sev].append((item['utterance'], item, p))

    total = sum(severity_counts.values())
    print(f"=== Error Severity Distribution ===\n")
    print(f"Total errors: {total} / {len(test)} ({total/len(test)*100:.2f}%)\n")

    order = ['catastrophic', 'annoying_query_as_control', 'annoying_control_as_query', 'minor_dir_only', 'graceful', 'other']
    for sev in order:
        cnt = severity_counts.get(sev, 0)
        pct = cnt / total * 100 if total else 0
        emoji = {'catastrophic': '🔴', 'annoying_query_as_control': '🟡', 'annoying_control_as_query': '🟡',
                 'minor_dir_only': '🟢', 'graceful': '🟢', 'other': '⚪'}[sev]
        print(f"  {emoji} {sev:<30} {cnt:3d} ({pct:5.1f}%)")
        for utt, gt, pred in severity_examples.get(sev, [])[:2]:
            print(f"      ex: \"{utt}\" | gt: {gt['fn']}/{gt['exec']}/{gt['dir']} → pred: {pred['fn']}/{pred['exec_type']}/{pred['param_direction']}")

    # User-centric score: catastrophic은 -1.0, annoying -0.5, minor -0.2, graceful 0
    user_score = (len(test) - severity_counts['catastrophic']*1.0
                  - severity_counts['annoying_query_as_control']*0.5
                  - severity_counts['annoying_control_as_query']*0.5
                  - severity_counts['minor_dir_only']*0.2)
    print(f"\nUser-weighted score: {user_score:.1f} / {len(test)} = {user_score/len(test)*100:.2f}%")
    print(f"(binary accuracy: {(len(test)-total)/len(test)*100:.2f}%)")


if __name__ == '__main__':
    main()
