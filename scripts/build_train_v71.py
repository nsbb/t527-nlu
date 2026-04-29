#!/usr/bin/env python3
"""
build_train_v71.py — v70 기반 데이터 수정 + 신규 소스 통합

수정 사항:
  1. train_final_v70.json: 어두침침 52개 dir=set → dir=on
  2. llm_paraphrases.jsonl: 2,068개 param_direction 추론 후 추가
  3. indirect_expressions.csv: 76개 flat→multi-head 변환 후 추가

출력: data/train_final_v71.json
"""

import json, csv, re, os, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# ============================================================
# 1. v70 로드 + 어두침침 fix
# ============================================================
print("=== Step 1: v70 로드 + 어두침침 fix ===")
with open('data/train_final_v70.json', encoding='utf-8') as f:
    data = json.load(f)

n_fixed = 0
dark_pattern = re.compile(r'어두침침|어두컴컴|어둑어둑|어둡침침|너무\s*어두')
for item in data:
    if dark_pattern.search(item['utterance']) and item['labels'].get('param_direction') == 'set':
        item['labels']['param_direction'] = 'on'
        n_fixed += 1

print(f"  어두침침 fix: {n_fixed}개 (dir=set → dir=on)")

# ============================================================
# 2. llm_paraphrases.jsonl — param_direction 추론
# ============================================================
print("\n=== Step 2: llm_paraphrases param_direction 추론 ===")

QUERY_FNS = {
    'weather_query', 'news_query', 'traffic_query', 'energy_query',
    'home_info', 'market_query', 'medical_query', 'info_query',
}


def infer_direction(utterance: str, fn: str) -> str:
    u = utterance

    if fn in QUERY_FNS:
        return 'none'

    # stop
    if re.search(r'정지|멈춰|세워|스탑', u):
        return 'stop'

    # open/close (door, curtain, gas)
    if fn in ('door_control', 'curtain_control', 'gas_control'):
        if re.search(r'열어|열거|열려|개방|올려', u):
            return 'open'
        if re.search(r'닫아|닫거|닫혀|잠가|잠궈|내려|잠금', u):
            return 'close'

    # up/down
    if re.search(r'올려|높여|높게|올리|강하게|강풍|쎄게|세게|더\s*높', u):
        return 'up'
    if re.search(r'낮춰|내려|낮게|낮추|줄여|약하게|약풍|더\s*낮', u):
        return 'down'

    # off
    if re.search(r'꺼|끄|종료|차단|끊어', u):
        return 'off'

    # ac mode set
    if fn == 'ac_control' and re.search(r'냉방|제습|송풍|자동|수면|슬립|쿨링|히팅|냉방모드|제습모드', u):
        return 'set'

    # heat set (온도 설정)
    if fn == 'heat_control' and re.search(r'도로\s*설정|도\s*설정|\d+도', u):
        return 'set'

    # schedule_manage / vehicle_manage / system_meta: mostly none
    if fn in ('schedule_manage', 'vehicle_manage', 'system_meta', 'unknown'):
        return 'none'

    # security_mode on (외출/귀가)
    if fn == 'security_mode':
        if re.search(r'해제|귀가|들어|왔', u):
            return 'off'
        return 'on'

    # elevator: on
    if fn == 'elevator_call':
        return 'on'

    # default for control fns: on (켜줘, 틀어, 가동, etc.)
    if fn in ('light_control', 'heat_control', 'ac_control', 'vent_control', 'gas_control'):
        return 'on'

    return 'none'


with open('data/llm_paraphrases.jsonl', encoding='utf-8') as f:
    para_lines = [json.loads(l) for l in f]

llm_items = []
dir_dist = Counter()
for item in para_lines:
    fn = item['labels']['fn']
    direction = infer_direction(item['utterance'], fn)
    dir_dist[direction] += 1

    # Convert to v70 label schema (fn, exec_type, param_direction, param_type, judge)
    exec_type = item['labels'].get('exec_type', 'control_then_confirm')
    # Fix exec_type for query fns
    if fn in QUERY_FNS and exec_type == 'control_then_confirm':
        exec_type = 'query_then_respond'

    new_item = {
        'utterance': item['utterance'],
        'labels': {
            'fn': fn,
            'exec_type': exec_type,
            'param_direction': direction,
            'param_type': 'none',
            'judge': 'none',
        },
        'source': 'llm_para',
    }
    if 'scenario_id' in item:
        new_item['scenario_id'] = item['scenario_id']
    llm_items.append(new_item)

print(f"  llm_paraphrases: {len(llm_items)}개 추가")
print(f"  direction 분포: {dict(dir_dist.most_common())}")

# ============================================================
# 3. indirect_expressions.csv — flat→multi-head
# ============================================================
print("\n=== Step 3: indirect_expressions.csv 변환 ===")

FLAT_TO_MULTIHEAD = {
    'heating_on':      ('heat_control',    'control_then_confirm', 'on'),
    'heating_off':     ('heat_control',    'control_then_confirm', 'off'),
    'heating_up':      ('heat_control',    'control_then_confirm', 'up'),
    'heating_down':    ('heat_control',    'control_then_confirm', 'down'),
    'light_on':        ('light_control',   'control_then_confirm', 'on'),
    'light_off':       ('light_control',   'control_then_confirm', 'off'),
    'light_up':        ('light_control',   'control_then_confirm', 'up'),
    'light_down':      ('light_control',   'control_then_confirm', 'down'),
    'ac_on':           ('ac_control',      'control_then_confirm', 'on'),
    'ac_off':          ('ac_control',      'control_then_confirm', 'off'),
    'ac_up':           ('ac_control',      'control_then_confirm', 'up'),
    'ac_down':         ('ac_control',      'control_then_confirm', 'down'),
    'ventilation_on':  ('vent_control',    'control_then_confirm', 'on'),
    'ventilation_off': ('vent_control',    'control_then_confirm', 'off'),
    'door_open':       ('door_control',    'control_then_confirm', 'open'),
    'door_close':      ('door_control',    'control_then_confirm', 'close'),
    'weather_query':   ('weather_query',   'query_then_respond',   'none'),
    'air_query':       ('weather_query',   'query_then_respond',   'none'),
    'unknown':         ('unknown',         'direct_respond',       'none'),
}

indirect_items = []
skipped = []
with open('data/indirect_expressions.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        intent = row['intent'].strip()
        utterance = row['utterance'].strip()
        if intent not in FLAT_TO_MULTIHEAD:
            skipped.append((intent, utterance))
            # Try to infer from intent string
            if intent.endswith('_on'):
                fn_part = intent[:-3]
                mapping = {
                    'heating': 'heat_control', 'light': 'light_control',
                    'ac': 'ac_control', 'ventilation': 'vent_control',
                }
                fn = mapping.get(fn_part, 'unknown')
                fn_lbl, exec_lbl, dir_lbl = fn, 'control_then_confirm', 'on'
            else:
                fn_lbl, exec_lbl, dir_lbl = 'unknown', 'direct_respond', 'none'
        else:
            fn_lbl, exec_lbl, dir_lbl = FLAT_TO_MULTIHEAD[intent]

        indirect_items.append({
            'utterance': utterance,
            'labels': {
                'fn': fn_lbl,
                'exec_type': exec_lbl,
                'param_direction': dir_lbl,
                'param_type': 'none',
                'judge': 'none',
            },
            'source': 'indirect_expr',
        })

print(f"  indirect_expressions: {len(indirect_items)}개 변환")
if skipped:
    print(f"  매핑 없음 (처리됨): {len(skipped)}개")
    for intent, utt in skipped[:5]:
        print(f"    [{intent}] {utt}")

# intent 분포
intent_dist = Counter()
for item in indirect_items:
    intent_dist[f"{item['labels']['fn']}_{item['labels']['param_direction']}"] += 1
print(f"  분포: {dict(intent_dist.most_common())}")

# ============================================================
# 4. 통합 + 출력
# ============================================================
print("\n=== Step 4: 통합 ===")

combined = data + llm_items + indirect_items

# 중복 제거 (utterance 기준)
seen = set()
deduped = []
dup_count = 0
for item in combined:
    key = item['utterance'].strip()
    if key in seen:
        dup_count += 1
        continue
    seen.add(key)
    deduped.append(item)

print(f"  v70: {len(data)}")
print(f"  + llm_para: {len(llm_items)}")
print(f"  + indirect: {len(indirect_items)}")
print(f"  중복 제거: {dup_count}")
print(f"  최종: {len(deduped)}")

# Source distribution
src_dist = Counter(x.get('source', '?') for x in deduped)
print(f"\n소스 분포:")
for k, v in src_dist.most_common():
    print(f"  {k}: {v}")

# Label sanity check
fn_dist = Counter(x['labels']['fn'] for x in deduped)
print(f"\nfn 분포 (top10):")
for k, v in fn_dist.most_common(10):
    print(f"  {k}: {v}")

out_path = 'data/train_final_v71.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(deduped, f, ensure_ascii=False, indent=2)
print(f"\n저장 완료: {out_path}")
