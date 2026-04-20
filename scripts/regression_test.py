#!/usr/bin/env python3
"""Regression Test — 핵심 동작 자동 검증

실패 시 exit code 1 (CI 통합 용).
현재 배포 모델 (Ensemble ONNX)이 반드시 통과해야 하는 최소 정확도 + 필수 케이스.

사용법:
    python3 scripts/regression_test.py              # 전체 실행
    python3 scripts/regression_test.py --fast       # 빠른 검증만
    python3 scripts/regression_test.py --json       # JSON 결과
"""
import os, sys, json, re, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from preprocess import preprocess
from transformers import AutoTokenizer

# 요구 성능 기준 (Ensemble ONNX 기준)
THRESHOLDS = {
    'test_suite_combo': 93.0,    # 93% 이상 통과
    'test_suite_fn': 97.0,
    'test_suite_exec': 97.0,
    'test_suite_dir': 96.0,
    'koelectra_fn': 97.0,
    'stt_resistance': 90.0,       # STT 내성 90% 이상
    'latency_ms': 5.0,             # 5ms 이하
}

# 필수 케이스 - 이건 반드시 맞아야 함 (golden set)
GOLDEN_CASES = [
    ("거실 에어컨 켜줘", "ac_control", "control_then_confirm", "on"),
    ("거실 에어컨 꺼줘", "ac_control", "control_then_confirm", "off"),
    ("거실 불 켜줘", "light_control", "control_then_confirm", "on"),
    ("거실 불 꺼줘", "light_control", "control_then_confirm", "off"),
    ("안방 난방 올려", "heat_control", "control_then_confirm", "up"),
    ("안방 난방 내려", "heat_control", "control_then_confirm", "down"),
    ("현관문 열어", "door_control", "control_then_confirm", "open"),
    ("현관문 잠가", "door_control", "control_then_confirm", "close"),
    ("가스 잠가줘", "gas_control", "control_then_confirm", "close"),
    ("오늘 날씨", "weather_query", "query_then_respond", "none"),
    ("엘리베이터 불러줘", "elevator_call", "control_then_confirm", "on"),
    ("외출할게", "security_mode", "control_then_confirm", "on"),
]

# STT 오류 내성 테스트
STT_CASES = [
    ("에어콘 켜줘", "ac_control"),
    ("뉴슈 알려줘", "news_query"),
    ("도어렉 열어", "door_control"),
    ("미세문지 어때", "weather_query"),
    ("남방 켜줘", "heat_control"),
    ("에어컨꺼", "ac_control"),
    ("오늘날씨어때", "weather_query"),
    ("씨원하게 해줘", "ac_control"),
    ("환기해줘", "vent_control"),
    ("커턴 열어", "curtain_control"),
]


def predict(text, sess, tok):
    clean = preprocess(text)
    tk = tok(clean, padding='max_length', truncation=True, max_length=32, return_tensors='np')
    outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
    return {
        'fn': HEAD_I2L['fn'][outs[0][0].argmax()],
        'exec_type': HEAD_I2L['exec_type'][outs[1][0].argmax()],
        'param_direction': HEAD_I2L['param_direction'][outs[2][0].argmax()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='빠른 검증만 (golden + STT)')
    parser.add_argument('--json', action='store_true', help='JSON 결과 출력')
    parser.add_argument('--model', default='checkpoints/nlu_v28_v46_ensemble.onnx')
    args = parser.parse_args()

    if not args.json:
        print(f"🧪 NLU Regression Test")
        print(f"   Model: {args.model}")
        print(f"   Mode: {'Fast' if args.fast else 'Full'}")

    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    results = {'pass': True, 'tests': {}}

    # 1. Golden cases (필수)
    golden_pass = 0
    golden_fail = []
    for text, exp_fn, exp_exec, exp_dir in GOLDEN_CASES:
        p = predict(text, sess, tok)
        if p['fn'] == exp_fn and p['exec_type'] == exp_exec and p['param_direction'] == exp_dir:
            golden_pass += 1
        else:
            golden_fail.append({
                'text': text, 'expected': {'fn': exp_fn, 'exec': exp_exec, 'dir': exp_dir},
                'got': {'fn': p['fn'], 'exec': p['exec_type'], 'dir': p['param_direction']},
            })
    golden_rate = golden_pass / len(GOLDEN_CASES) * 100
    golden_ok = golden_rate == 100.0
    results['tests']['golden'] = {
        'pass': golden_ok, 'rate': golden_rate,
        'passed': golden_pass, 'total': len(GOLDEN_CASES),
        'failures': golden_fail,
    }
    if not golden_ok: results['pass'] = False

    if not args.json:
        print(f"\n[1] Golden Cases (필수 {len(GOLDEN_CASES)}개)")
        print(f"    {'✓' if golden_ok else '✗'} {golden_pass}/{len(GOLDEN_CASES)} = {golden_rate:.1f}%")
        for f in golden_fail:
            print(f"    ✗ \"{f['text']}\" exp={f['expected']} got={f['got']}")

    # 2. STT 내성
    stt_pass = 0
    stt_fail = []
    for text, exp_fn in STT_CASES:
        p = predict(text, sess, tok)
        if p['fn'] == exp_fn:
            stt_pass += 1
        else:
            stt_fail.append({'text': text, 'expected_fn': exp_fn, 'got_fn': p['fn']})
    stt_rate = stt_pass / len(STT_CASES) * 100
    stt_ok = stt_rate >= THRESHOLDS['stt_resistance']
    results['tests']['stt'] = {
        'pass': stt_ok, 'rate': stt_rate, 'threshold': THRESHOLDS['stt_resistance'],
        'passed': stt_pass, 'total': len(STT_CASES),
        'failures': stt_fail,
    }
    if not stt_ok: results['pass'] = False

    if not args.json:
        print(f"\n[2] STT 내성 ({len(STT_CASES)}개, ≥{THRESHOLDS['stt_resistance']}%)")
        print(f"    {'✓' if stt_ok else '✗'} {stt_pass}/{len(STT_CASES)} = {stt_rate:.1f}%")
        for f in stt_fail:
            print(f"    ✗ \"{f['text']}\" expected={f['expected_fn']} got={f['got_fn']}")

    if not args.fast:
        # 3. Test Suite 전체
        suite = json.load(open('data/test_suite.json'))
        fn_ok = exec_ok = dir_ok = all_ok = 0
        for t in suite:
            p = predict(t['utterance'], sess, tok)
            if p['fn'] == t['fn']: fn_ok += 1
            if p['exec_type'] == t['exec']: exec_ok += 1
            if p['param_direction'] == t['dir']: dir_ok += 1
            if p['fn'] == t['fn'] and p['exec_type'] == t['exec'] and p['param_direction'] == t['dir']:
                all_ok += 1
        n = len(suite)
        ts = {
            'combo': all_ok/n*100, 'fn': fn_ok/n*100,
            'exec': exec_ok/n*100, 'dir': dir_ok/n*100,
        }
        ts_ok = (ts['combo'] >= THRESHOLDS['test_suite_combo'] and
                 ts['fn'] >= THRESHOLDS['test_suite_fn'] and
                 ts['exec'] >= THRESHOLDS['test_suite_exec'] and
                 ts['dir'] >= THRESHOLDS['test_suite_dir'])
        results['tests']['test_suite'] = {
            'pass': ts_ok, 'metrics': ts, 'thresholds': {
                'combo': THRESHOLDS['test_suite_combo'],
                'fn': THRESHOLDS['test_suite_fn'],
                'exec': THRESHOLDS['test_suite_exec'],
                'dir': THRESHOLDS['test_suite_dir'],
            }
        }
        if not ts_ok: results['pass'] = False

        if not args.json:
            print(f"\n[3] Test Suite ({n}개)")
            print(f"    {'✓' if ts_ok else '✗'} combo={ts['combo']:.2f}% (≥{THRESHOLDS['test_suite_combo']}%)")
            print(f"      fn={ts['fn']:.2f} exec={ts['exec']:.2f} dir={ts['dir']:.2f}")

        # 4. KoELECTRA
        ke_val = json.load(open('data/koelectra_converted_val.json'))
        ke_ok = 0
        for d in ke_val:
            p = predict(d['utterance'], sess, tok)
            if p['fn'] == d['labels']['fn']: ke_ok += 1
        ke_rate = ke_ok/len(ke_val)*100
        ke_pass = ke_rate >= THRESHOLDS['koelectra_fn']
        results['tests']['koelectra'] = {
            'pass': ke_pass, 'rate': ke_rate,
            'threshold': THRESHOLDS['koelectra_fn'],
            'passed': ke_ok, 'total': len(ke_val),
        }
        if not ke_pass: results['pass'] = False

        if not args.json:
            print(f"\n[4] KoELECTRA fn ({len(ke_val)}개)")
            print(f"    {'✓' if ke_pass else '✗'} {ke_rate:.2f}% (≥{THRESHOLDS['koelectra_fn']}%)")

    # 5. Latency
    tk = tok("거실 에어컨 켜줘", padding='max_length', truncation=True, max_length=32, return_tensors='np')
    input_ids = tk['input_ids'].astype(np.int64)
    for _ in range(10): sess.run(None, {'input_ids': input_ids})  # warmup
    N = 100
    start = time.time()
    for _ in range(N): sess.run(None, {'input_ids': input_ids})
    latency = (time.time() - start) * 1000 / N
    lat_ok = latency <= THRESHOLDS['latency_ms']
    results['tests']['latency'] = {
        'pass': lat_ok, 'ms': latency, 'threshold': THRESHOLDS['latency_ms']
    }
    if not lat_ok: results['pass'] = False

    if not args.json:
        print(f"\n[5] Latency (CPU)")
        print(f"    {'✓' if lat_ok else '✗'} {latency:.2f}ms (≤{THRESHOLDS['latency_ms']}ms)")

    # 최종 결과
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*50}")
        if results['pass']:
            print(f"   ✅ ALL TESTS PASSED")
        else:
            print(f"   ❌ FAILED")
            failed = [k for k, v in results['tests'].items() if not v.get('pass', True)]
            print(f"   실패 테스트: {', '.join(failed)}")
        print('='*50)

    sys.exit(0 if results['pass'] else 1)


if __name__ == '__main__':
    main()
