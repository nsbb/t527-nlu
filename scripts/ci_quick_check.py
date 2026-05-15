#!/usr/bin/env python3
"""빠른 회귀 체크 (~2초) — pre-commit hook용.

진짜 production GT (gt_known_v2 + gt_unknown = 219) + golden_test_100 사용.
golden_ruel_219.json은 자동 매핑이라 잘못된 라벨 — 사용 안 함.

baseline은 data/ci/ci_baseline_quick.json. 의도된 개선이면 --update로 갱신.

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
BASELINE = ROOT / 'data' / 'ci' / 'ci_baseline_quick.json'
THRESHOLD = -2.0  # %p

ONNX_PATH = ROOT / 'checkpoints' / 'nlu_v28_v72_ensemble.onnx'
TOK_PATH  = ROOT / 'tokenizer'

# 진짜 production GT — gt_known_v2 + gt_unknown = 219개 사람 검증된 라벨
# (다른 golden_test_*, golden_indirect_* 는 자동 생성 시험용 → 2026-05-15 폐기)
GOLDEN_RUEL_FILES = [
    ROOT / 'data' / 'golden' / 'gt_known_scenarios_v2.json',
    ROOT / 'data' / 'golden' / 'gt_unknown_scenarios.json',
]

HEAD_FN = ['light_control','heat_control','ac_control','vent_control','gas_control',
           'door_control','curtain_control','elevator_call','security_mode',
           'schedule_manage','weather_query','news_query','traffic_query',
           'energy_query','home_info','system_meta','market_query',
           'medical_query','vehicle_manage','unknown']
HEAD_EX = ['query_then_respond','control_then_confirm','query_then_judge','direct_respond','clarify']
HEAD_DIR = ['none','up','down','set','on','off','open','close','stop']


def load_set(path):
    """golden_test 형식 (utterance/fn/exec/dir) 또는 gt_known 형식 (labels) 모두 처리."""
    data = json.load(open(path))
    items = []
    for it in data:
        utt = it['utterance']
        if 'labels' in it:
            l = it['labels']
            items.append({'utterance': utt, 'fn': l['fn'], 'exec': l['exec_type'], 'dir': l['param_direction']})
        else:
            items.append({'utterance': utt, 'fn': it['fn'], 'exec': it['exec'], 'dir': it['dir']})
    return items


def eval_set(sess, tok, items):
    fn = ex = dr = cb = 0
    for it in items:
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
    n = len(items)
    return {'fn': fn/n*100, 'exec': ex/n*100, 'dir': dr/n*100, 'combo': cb/n*100, 'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--update', action='store_true', help='현재 결과를 새 baseline으로 저장')
    args = ap.parse_args()

    if not ONNX_PATH.exists():
        print(f'⚠️  ONNX 없음: {ONNX_PATH} — 스킵')
        return 0

    t0 = time.time()
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained(str(TOK_PATH))

    results = {}
    # ruel 219 = gt_known_v2 + gt_unknown (단일 production GT)
    ruel_items = []
    for p in GOLDEN_RUEL_FILES:
        if p.exists():
            ruel_items += load_set(p)
    if ruel_items:
        results['ruel_219'] = eval_set(sess, tok, ruel_items)

    elapsed = time.time() - t0
    print(f'⚡ Quick CI check ({elapsed:.1f}s)')
    for name, r in results.items():
        print(f"  {name:12s} fn {r['fn']:5.1f}%  combo {r['combo']:5.1f}%  (n={r['n']})")

    if args.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\n✅ baseline 갱신: {BASELINE}')
        return 0

    if not BASELINE.exists():
        print(f'\nℹ️  baseline 없음 — --update 로 초기 baseline 저장하세요.')
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
        print('\n의도된 개선이면 `python3 scripts/ci_quick_check.py --update`')
        return 1
    print('\n✅ 회귀 없음')
    return 0


if __name__ == '__main__':
    sys.exit(main())
