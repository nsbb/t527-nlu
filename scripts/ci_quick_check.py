#!/usr/bin/env python3
"""빠른 회귀 체크 (~2초) — pre-commit hook용.

골든셋 99 + 르엘 219만 측정. baseline 대비 -2%p 이상 떨어지면 exit 1.
baseline은 data/ci_baseline.json. 의도된 개선이면 --update로 갱신.

사용:
  python3 scripts/ci_quick_check.py            # 체크 (회귀시 exit 1)
  python3 scripts/ci_quick_check.py --update   # 현재 결과를 새 baseline으로
"""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from preprocess import preprocess

ROOT = Path(__file__).parent.parent
BASELINE = ROOT / 'data' / 'ci_baseline_quick.json'
THRESHOLD = -2.0  # %p

ONNX_PATH = ROOT / 'checkpoints' / 'nlu_v28_v72_ensemble.onnx'
TOK_PATH  = ROOT / 'tokenizer'
SETS = [
    ('golden_99', ROOT / 'data' / 'golden_test_100.json'),
    ('golden_ruel_219', ROOT / 'data' / 'golden_ruel_219.json'),
]

HEAD_FN = ['light_control','heat_control','ac_control','vent_control','gas_control',
           'door_control','curtain_control','elevator_call','security_mode',
           'schedule_manage','weather_query','news_query','traffic_query',
           'energy_query','home_info','system_meta','market_query',
           'medical_query','vehicle_manage','unknown']
HEAD_EX = ['query_then_respond','control_then_confirm','query_then_judge','direct_respond','clarify']
HEAD_DIR = ['none','up','down','set','on','off','open','close','stop']


def eval_set(sess, tok, path):
    data = json.load(open(path))
    fn = ex = dr = cb = 0
    for it in data:
        enc = tok(preprocess(it['utterance']), padding='max_length', max_length=32, truncation=True, return_tensors='np')
        outs = sess.run(None, {'input_ids': enc['input_ids'].astype(np.int64)})
        d = dict(zip([o.name for o in sess.get_outputs()], outs))
        f = HEAD_FN[int(d['fn_logits'].argmax())]
        e = HEAD_EX[int(d['exec_logits'].argmax())]
        di = HEAD_DIR[int(d['dir_logits'].argmax())]
        if f == it['fn']: fn += 1
        if e == it['exec']: ex += 1
        if di == it['dir']: dr += 1
        if f == it['fn'] and e == it['exec'] and di == it['dir']: cb += 1
    n = len(data)
    return {'fn': fn/n*100, 'exec': ex/n*100, 'dir': dr/n*100, 'combo': cb/n*100, 'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true', help='현재 결과를 새 baseline으로 저장')
    args = ap.parse_args()

    if not ONNX_PATH.exists():
        print(f'⚠️  ONNX 없음: {ONNX_PATH} — 스킵 (개발 환경 미준비)')
        return 0

    t0 = time.time()
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH))

    results = {}
    for name, path in SETS:
        if not path.exists(): continue
        results[name] = eval_set(sess, tok, path)

    elapsed = time.time() - t0
    print(f'⚡ Quick CI check ({elapsed:.1f}s)')
    for name, r in results.items():
        print(f"  {name:20s} fn {r['fn']:5.1f}%  combo {r['combo']:5.1f}%  (n={r['n']})")

    # baseline 비교 / 업데이트
    if args.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\n✅ baseline 갱신: {BASELINE}')
        return 0

    if not BASELINE.exists():
        print(f'\nℹ️  baseline 없음 — 첫 실행. --update 로 초기 baseline 저장하세요.')
        return 0

    base = json.load(open(BASELINE))
    regressed = []
    for name, r in results.items():
        b = base.get(name, {})
        for k in ['fn', 'combo']:
            delta = r[k] - b.get(k, r[k])
            if delta < THRESHOLD:
                regressed.append((name, k, b[k], r[k], delta))

    if regressed:
        print(f'\n❌ REGRESSION 감지 (임계값 {THRESHOLD}%p):')
        for name, k, old, new, delta in regressed:
            print(f"   {name}/{k}: {old:.1f}% → {new:.1f}% ({delta:+.1f}%p)")
        print('\n의도된 개선이면 `python3 scripts/ci_quick_check.py --update` 로 baseline 갱신 후 commit')
        return 1
    print('\n✅ 회귀 없음')
    return 0


if __name__ == '__main__':
    sys.exit(main())
