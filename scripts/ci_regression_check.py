#!/usr/bin/env python3
"""CI 회귀 테스트 — 모델 또는 규칙 변경 시 골든셋 모두 돌리고 회귀 감지.

산업 표준 적용:
- Microsoft NLU.DevOps: 골든셋 + CI 자동 검증
- Hamming AI: -2%p 이상 회귀 시 alert
- 우리: 골든셋 4종 (15/99/491/56) 모두 자동 실행

사용:
  python3 scripts/ci_regression_check.py              # 측정 + baseline 비교
  python3 scripts/ci_regression_check.py --update     # baseline 업데이트
"""
import sys, json, re, argparse, time
from pathlib import Path
sys.path.insert(0, 'scripts')

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess
from model_cnn_multihead import HEAD_I2L, HEAD_NAMES


GOLDEN_SETS = [
    ('test_suite', 'data/test_suite.json', 'TS 3043'),
    ('golden_99', 'data/golden_test_100.json', '99 골든셋 (over-fit 위험 small)'),
    ('golden_491', 'data/golden_test_500.json', '491 골든셋 (정직 측정)'),
    ('golden_indirect_56', 'data/golden_indirect_55.json', '비유/완곡/STT 56개'),
    ('golden_ruel_219', 'data/golden_ruel_219.json', '르엘 공식 219 (자동 매핑)'),
]

BASELINE_PATH = 'data/ci_baseline.json'
REGRESSION_THRESHOLD = -2.0  # %p


def normalize_field(d, expected_key):
    """test_suite 형식이 다양해서 호환 처리"""
    if expected_key == 'fn':
        return d.get('fn', d.get('labels', {}).get('fn', ''))
    elif expected_key == 'exec':
        return d.get('exec', d.get('exec_type', d.get('labels', {}).get('exec_type', '')))
    elif expected_key == 'dir':
        return d.get('dir', d.get('param_direction', d.get('labels', {}).get('param_direction', '')))
    return ''


def evaluate_set(sess, tok, golden_path):
    """골든셋 하나 평가 → fn/exec/dir/combo 정확도"""
    if not Path(golden_path).exists():
        return None
    data = json.load(open(golden_path))
    fn_ok = ex_ok = dir_ok = combo = 0
    n = 0
    t_start = time.time()
    for t in data:
        utt = t.get('utterance', '')
        if not utt: continue
        # GT
        gt_fn = normalize_field(t, 'fn')
        gt_ex = normalize_field(t, 'exec')
        gt_dir = normalize_field(t, 'dir')
        if not gt_fn: continue
        # 예측
        try:
            text = re.sub(r'\s+', ' ', ''.join(c if c.isprintable() or c == ' ' else ' ' for c in utt)).strip()
            p = predict_with_rules(text, sess, tok)
        except Exception:
            continue
        n += 1
        if p['fn'] == gt_fn: fn_ok += 1
        if p['exec_type'] == gt_ex: ex_ok += 1
        if p['param_direction'] == gt_dir: dir_ok += 1
        if p['fn'] == gt_fn and p['exec_type'] == gt_ex and p['param_direction'] == gt_dir:
            combo += 1
    elapsed = time.time() - t_start
    return {
        'n': n,
        'fn': fn_ok, 'fn_pct': fn_ok / n * 100 if n else 0,
        'exec': ex_ok, 'exec_pct': ex_ok / n * 100 if n else 0,
        'dir': dir_ok, 'dir_pct': dir_ok / n * 100 if n else 0,
        'combo': combo, 'combo_pct': combo / n * 100 if n else 0,
        'elapsed_sec': round(elapsed, 1),
    }


def compare_baseline(current, baseline):
    """baseline 대비 회귀 확인 → 회귀 항목 list 반환"""
    regressions = []
    for set_name, cur in current.items():
        if cur is None: continue
        base = baseline.get(set_name)
        if base is None: continue
        for metric in ['fn_pct', 'combo_pct']:
            cur_v = cur[metric]
            base_v = base[metric]
            delta = cur_v - base_v
            if delta < REGRESSION_THRESHOLD:
                regressions.append({
                    'set': set_name, 'metric': metric,
                    'baseline': base_v, 'current': cur_v, 'delta': delta
                })
    return regressions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true', help='baseline 업데이트')
    parser.add_argument('--model', default='checkpoints/nlu_v28_v46_ensemble.onnx')
    args = parser.parse_args()

    print(f"=== CI 회귀 테스트 ===")
    print(f"모델: {args.model}")
    print()

    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    results = {}
    for set_name, path, desc in GOLDEN_SETS:
        print(f"  {desc} ({path}) 평가 중...", end=' ', flush=True)
        r = evaluate_set(sess, tok, path)
        if r is None:
            print("(파일 없음, 스킵)")
            continue
        results[set_name] = r
        print(f"combo {r['combo_pct']:.1f}% ({r['elapsed_sec']}s)")

    print()
    print(f"{'골든셋':<25} {'fn%':>8} {'exec%':>8} {'dir%':>8} {'combo%':>10} {'n':>6}")
    print("=" * 78)
    for set_name, r in results.items():
        print(f"{set_name:<25} {r['fn_pct']:>7.2f}% {r['exec_pct']:>7.2f}% {r['dir_pct']:>7.2f}% {r['combo_pct']:>9.2f}% {r['n']:>6}")

    if args.update:
        with open(BASELINE_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Baseline 업데이트: {BASELINE_PATH}")
        return

    # baseline 비교
    if not Path(BASELINE_PATH).exists():
        print(f"\n⚠ Baseline 없음: {BASELINE_PATH}")
        print(f"   --update로 baseline 생성 후 다음 실행부터 비교")
        return

    baseline = json.load(open(BASELINE_PATH))
    regressions = compare_baseline(results, baseline)

    print(f"\n=== Baseline 비교 ===")
    if not regressions:
        print(f"  ✅ 회귀 없음 (모든 골든셋 baseline 대비 -{abs(REGRESSION_THRESHOLD)}%p 이내)")
        sys.exit(0)
    else:
        print(f"  ❌ 회귀 {len(regressions)}건:")
        for r in regressions:
            print(f"    {r['set']} / {r['metric']}: {r['baseline']:.1f}% → {r['current']:.1f}% (Δ {r['delta']:+.1f}%p)")
        sys.exit(1)


if __name__ == '__main__':
    main()
