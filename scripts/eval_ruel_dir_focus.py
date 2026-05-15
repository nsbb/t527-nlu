#!/usr/bin/env python3
"""219 르엘 골든셋 → v46 raw + ensemble 추론 → dir 오류 패턴 추출.

목적: dir 헤드 약점 정량화, STT 변형 영향 측정.
출력: data/analysis/ruel_errors_dir_focus.csv
"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import onnxruntime as ort
from collections import Counter
from preprocess import preprocess
from transformers import AutoTokenizer

CHK = '/home/nsbb/travail/claude/T527/t527-nlu/checkpoints'
TOK = '/home/nsbb/travail/claude/T527/t527-nlu/tokenizer/'

HEAD_FN = ['light_control','heat_control','ac_control','vent_control','gas_control',
           'door_control','curtain_control','elevator_call','security_mode',
           'schedule_manage','weather_query','news_query','traffic_query',
           'energy_query','home_info','system_meta','market_query',
           'medical_query','vehicle_manage','unknown']
HEAD_EXEC = ['query_then_respond','control_then_confirm','query_then_judge','direct_respond','clarify']
HEAD_DIR = ['none','up','down','set','on','off','open','close','stop']

def main():
    data = json.load(open('/home/nsbb/travail/claude/T527/t527-nlu/data/golden/golden_test_500.json'))
    tok = AutoTokenizer.from_pretrained(TOK)
    s = ort.InferenceSession(f'{CHK}/nlu_v28_v46_ensemble.onnx', providers=['CPUExecutionProvider'])

    rows = []
    fn_ok = exec_ok = dir_ok = combo_ok = 0
    for it in data:
        utt = it['utterance']
        gt_fn = it.get('fn'); gt_exec = it.get('exec'); gt_dir = it.get('dir')
        pp = preprocess(utt)
        enc = tok(pp, padding='max_length', max_length=32, truncation=True, return_tensors='np')
        outs = s.run(None, {'input_ids': enc['input_ids'].astype(np.int64)})
        d = dict(zip([o.name for o in s.get_outputs()], outs))
        fn_p = HEAD_FN[int(d['fn_logits'].argmax())]
        ex_p = HEAD_EXEC[int(d['exec_logits'].argmax())]
        dir_p = HEAD_DIR[int(d['dir_logits'].argmax())]

        is_fn_ok = fn_p == gt_fn
        is_exec_ok = ex_p == gt_exec
        is_dir_ok = dir_p == gt_dir
        if is_fn_ok: fn_ok += 1
        if is_exec_ok: exec_ok += 1
        if is_dir_ok: dir_ok += 1
        if is_fn_ok and is_exec_ok and is_dir_ok: combo_ok += 1

        if not (is_fn_ok and is_exec_ok and is_dir_ok):
            stt_changed = (utt != pp)
            rows.append({
                'utterance': utt, 'preprocessed': pp, 'stt_changed': stt_changed,
                'fn_exp': gt_fn, 'fn_pred': fn_p, 'fn_ok': is_fn_ok,
                'exec_exp': gt_exec, 'exec_pred': ex_p, 'exec_ok': is_exec_ok,
                'dir_exp': gt_dir, 'dir_pred': dir_p, 'dir_ok': is_dir_ok,
            })

    N = len(data)
    print(f'=== 르엘 219 ensemble RAW (no PostRules) ===')
    print(f'fn:    {fn_ok}/{N} ({100*fn_ok/N:.1f}%)')
    print(f'exec:  {exec_ok}/{N} ({100*exec_ok/N:.1f}%)')
    print(f'dir:   {dir_ok}/{N} ({100*dir_ok/N:.1f}%)')
    print(f'combo: {combo_ok}/{N} ({100*combo_ok/N:.1f}%)')

    # dir 오류 패턴 분석
    dir_errs = [r for r in rows if not r['dir_ok']]
    print(f'\n=== dir 오류 {len(dir_errs)}건 패턴 ===')
    dir_pat = Counter((r['dir_exp'], r['dir_pred']) for r in dir_errs)
    for (e,p), c in dir_pat.most_common(10):
        print(f'  {e:6s} → {p:6s}: {c}건')

    # STT 변형 영향
    stt = [r for r in rows if r['stt_changed']]
    print(f'\n=== STT 변형 발화에서 깨진 케이스: {len(stt)}건 ===')
    for r in stt[:10]:
        flags = []
        if not r['fn_ok']: flags.append(f"fn:{r['fn_exp']}→{r['fn_pred']}")
        if not r['exec_ok']: flags.append(f"ex:{r['exec_exp']}→{r['exec_pred']}")
        if not r['dir_ok']: flags.append(f"dir:{r['dir_exp']}→{r['dir_pred']}")
        print(f"  '{r['utterance']}' → '{r['preprocessed']}'  [{'; '.join(flags)}]")

    # CSV 저장
    out = '/home/nsbb/travail/claude/T527/t527-nlu/data/analysis/ruel_errors_dir_focus.csv'
    with open(out, 'w', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader(); w.writerows(rows)
    print(f'\n저장: {out} ({len(rows)} 오답)')

if __name__ == '__main__':
    main()
