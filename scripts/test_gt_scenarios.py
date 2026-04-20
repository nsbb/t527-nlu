#!/usr/bin/env python3
"""GT 시나리오 전수 자동 테스트 (엑셀 기반 219개)
- 204 known + 15 unknown = 219개 시나리오
- 3개 모델 비교 (v28 / v46 / ensemble)
- 카테고리별 정확도 + 실패 케이스 상세

사용법:
    python3 scripts/test_gt_scenarios.py              # 전체
    python3 scripts/test_gt_scenarios.py --fail-only  # 실패만
    python3 scripts/test_gt_scenarios.py --export gt_test_results.csv
"""
import os, sys, json, re, argparse, csv
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES, CNNMultiHead
from preprocess import preprocess
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import onnxruntime as ort


def load_models():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m28.load_state_dict(torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)['state'])
    m28.eval()

    m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    m46.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)['state'])
    m46.eval()

    ens = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                               providers=['CPUExecutionProvider'])
    return {'v28': m28, 'v46': m46, 'ens': ens}, tok


def predict_torch(model, tok, text):
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad(): l = model(tk['input_ids'])
    conf = F.softmax(l['fn'], dim=1).max().item()
    p = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
    return p, conf


def predict_onnx(sess, tok, text):
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='np')
    outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
    p = {
        'fn': HEAD_I2L['fn'][outs[0][0].argmax()],
        'exec_type': HEAD_I2L['exec_type'][outs[1][0].argmax()],
        'param_direction': HEAD_I2L['param_direction'][outs[2][0].argmax()],
        'param_type': HEAD_I2L['param_type'][outs[3][0].argmax()],
        'judge': HEAD_I2L['judge'][outs[4][0].argmax()],
    }
    fn_probs = np.exp(outs[0][0] - outs[0][0].max()); fn_probs /= fn_probs.sum()
    return p, float(fn_probs.max())


def apply_post(p):
    if p['param_direction'] in ('open', 'close', 'stop'): p['param_type'] = 'none'
    if p['judge'] != 'none': p['param_type'] = 'none'
    if p['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
        p['param_type'] = 'none'
    return p


def compare(pred, gt):
    """fn/exec/dir 3가지 필드 비교"""
    fn_ok = pred['fn'] == gt.get('fn')
    exec_ok = pred['exec_type'] == gt.get('exec_type')
    dir_ok = pred['param_direction'] == gt.get('param_direction')
    return fn_ok, exec_ok, dir_ok, fn_ok and exec_ok and dir_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fail-only', action='store_true')
    parser.add_argument('--export', help='CSV 출력 파일')
    parser.add_argument('--use-preprocess', action='store_true', help='전처리 적용 (STT 교정)')
    args = parser.parse_args()

    print("🧪 GT 시나리오 전수 자동 테스트")
    print("=" * 70)

    models, tok = load_models()

    # GT 로드
    gt_known = json.load(open('data/gt_known_scenarios.json'))
    gt_unk = json.load(open('data/gt_unknown_scenarios.json'))
    all_gt = gt_known + gt_unk
    print(f"\n총 시나리오: {len(all_gt)}개 (known {len(gt_known)} + unknown {len(gt_unk)})")

    # 각 모델별 결과
    results = {'v28': [], 'v46': [], 'ens': []}
    detail_rows = []

    for gt in all_gt:
        text = gt['utterance']
        if args.use_preprocess:
            text = preprocess(text)

        exp_labels = gt['labels']
        exp_fn = exp_labels.get('fn')
        exp_exec = exp_labels.get('exec_type')
        exp_dir = exp_labels.get('param_direction')

        row = {
            'scenario_id': gt['scenario_id'],
            'cat': gt.get('cat', ''),
            'func': gt.get('func', ''),
            'stype': gt.get('stype', ''),
            'utterance': gt['utterance'],
            'exp_fn': exp_fn, 'exp_exec': exp_exec, 'exp_dir': exp_dir,
        }

        # 각 모델
        for name, model in [('v28', models['v28']), ('v46', models['v46']), ('ens', models['ens'])]:
            if name == 'ens':
                p, conf = predict_onnx(model, tok, text)
            else:
                p, conf = predict_torch(model, tok, text)
            p = apply_post(p)
            fn_ok, exec_ok, dir_ok, all_ok = compare(p, exp_labels)
            results[name].append({
                'gt': gt, 'pred': p, 'conf': conf,
                'fn_ok': fn_ok, 'exec_ok': exec_ok, 'dir_ok': dir_ok, 'all_ok': all_ok,
            })
            row[f'{name}_fn'] = p['fn']
            row[f'{name}_exec'] = p['exec_type']
            row[f'{name}_dir'] = p['param_direction']
            row[f'{name}_conf'] = f"{conf:.2f}"
            row[f'{name}_combo'] = 'O' if all_ok else 'X'

        detail_rows.append(row)

    # 요약
    print(f"\n=== 모델별 GT 정확도 ===")
    for name, desc in [('v28', 'v28 (GT 전용)'), ('v46', 'v46 (일반화)'), ('ens', 'Ensemble v28+v46')]:
        rs = results[name]
        n = len(rs)
        fn_ok = sum(r['fn_ok'] for r in rs)
        exec_ok = sum(r['exec_ok'] for r in rs)
        dir_ok = sum(r['dir_ok'] for r in rs)
        all_ok = sum(r['all_ok'] for r in rs)
        print(f"  {desc}:")
        print(f"    fn: {fn_ok}/{n} = {fn_ok/n*100:.1f}%  "
              f"exec: {exec_ok}/{n} = {exec_ok/n*100:.1f}%  "
              f"dir: {dir_ok}/{n} = {dir_ok/n*100:.1f}%  "
              f"combo: {all_ok}/{n} = {all_ok/n*100:.1f}%")

    # 카테고리별
    print(f"\n=== 카테고리별 Ensemble 정확도 ===")
    by_cat = defaultdict(lambda: {'total': 0, 'ok': 0})
    for r in results['ens']:
        c = r['gt'].get('cat', '기타')
        by_cat[c]['total'] += 1
        if r['all_ok']: by_cat[c]['ok'] += 1
    for c in sorted(by_cat):
        b = by_cat[c]
        print(f"  {c:8s}: {b['ok']}/{b['total']} = {b['ok']/b['total']*100:.1f}%")

    # Func별 (더 세밀)
    print(f"\n=== Func별 Ensemble 실패 ===")
    by_func = defaultdict(lambda: {'total': 0, 'ok': 0, 'fails': []})
    for r in results['ens']:
        f = r['gt'].get('func', '기타')
        by_func[f]['total'] += 1
        if r['all_ok']:
            by_func[f]['ok'] += 1
        else:
            by_func[f]['fails'].append(r)
    fail_funcs = [(f, b) for f, b in by_func.items() if len(b['fails']) > 0]
    fail_funcs.sort(key=lambda x: -len(x[1]['fails']))
    for f, b in fail_funcs[:10]:
        print(f"  {f:12s}: {b['ok']}/{b['total']} — 실패 {len(b['fails'])}건")

    # 실패 케이스 상세
    ens_fails = [r for r in results['ens'] if not r['all_ok']]
    print(f"\n=== Ensemble 실패 {len(ens_fails)}건 상세 ===")
    for r in ens_fails[:30]:
        g = r['gt']; p = r['pred']
        diffs = []
        if not r['fn_ok']: diffs.append(f"fn: {g['labels']['fn']} → {p['fn']}")
        if not r['exec_ok']: diffs.append(f"exec: {g['labels']['exec_type']} → {p['exec_type']}")
        if not r['dir_ok']: diffs.append(f"dir: {g['labels']['param_direction']} → {p['param_direction']}")
        print(f"  [{g.get('cat', '')}|{g.get('func', '')}|{g.get('stype', '')}] \"{g['utterance']}\"")
        print(f"    {' / '.join(diffs)}  (conf {r['conf']:.2f})")

    if len(ens_fails) > 30:
        print(f"  ... 그 외 {len(ens_fails)-30}건")

    # v28 vs v46 비교 (앙상블의 장점 확인)
    print(f"\n=== 모델 간 비교 (어느 모델이 어디서 강한지) ===")
    v28_only_ok = v46_only_ok = both_ok = both_fail = 0
    for i in range(len(all_gt)):
        r28 = results['v28'][i]['all_ok']
        r46 = results['v46'][i]['all_ok']
        if r28 and r46: both_ok += 1
        elif r28 and not r46: v28_only_ok += 1
        elif not r28 and r46: v46_only_ok += 1
        else: both_fail += 1
    print(f"  둘 다 맞음:     {both_ok}")
    print(f"  v28만 맞음:     {v28_only_ok}")
    print(f"  v46만 맞음:     {v46_only_ok}")
    print(f"  둘 다 틀림:     {both_fail}")

    # CSV export
    if args.export:
        with open(args.export, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            for row in detail_rows:
                writer.writerow(row)
        print(f"\n✓ CSV 출력: {args.export} ({len(detail_rows)}행)")


if __name__ == '__main__':
    main()
