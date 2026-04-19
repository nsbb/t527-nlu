#!/usr/bin/env python3
"""KoELECTRA train/val JSONL → 멀티헤드 5-head JSON 변환기

사용법:
    python3 scripts/convert_koelectra_to_multihead.py

입력:
    ../wewonnim/koelectra_wallpad/data/train_v8.jsonl  (13,540개)
    ../wewonnim/koelectra_wallpad/data/val_v8.jsonl    (1,536개)

출력:
    data/koelectra_converted_train.json   — 멀티헤드 5축 학습데이터
    data/koelectra_converted_val.json     — 멀티헤드 5축 검증데이터
    data/koelectra_conversion_report.txt  — 변환 통계 리포트

발화문 텍스트에서 param_type/judge를 추가 추론하여 라벨만으로
결정 못 하는 축도 보강함.
"""
import json, re, sys, os
from collections import Counter
from pathlib import Path

# ============================================================
# 매핑 테이블: KoELECTRA label → multi-head 5축
# ============================================================
# 애매한 11개는 다음 규칙으로 확정:
#   call_*          → home_info (통화 = 단지 정보 서비스)
#   access_history  → security_mode
#   card_manage     → security_mode
#   visitor_video   → security_mode
#   emergency_report→ security_mode
#   cooktop_*       → gas_control (쿡탑 = 가스 계열)
#   outlet_*        → light_control (콘센트 = 전기 제어 계열)

LABEL_MAP = {
    # === AC (9) ===
    'ac_control_cool':  {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'mode',        'judge': 'none'},
    'ac_control_off':   {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',   'param_type': 'none',        'judge': 'none'},
    'ac_control_on':    {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',    'param_type': 'none',        'judge': 'none'},
    'ac_control_warm':  {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'mode',        'judge': 'none'},
    'ac_mode':          {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'mode',        'judge': 'none'},
    'ac_query':         {'fn': 'ac_control', 'exec_type': 'query_then_respond',   'param_direction': 'none',  'param_type': 'none',        'judge': 'none'},
    'ac_schedule':      {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'none',        'judge': 'none'},
    'ac_temp':          {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'temperature', 'judge': 'none'},
    'ac_wind':          {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'speed',       'judge': 'none'},

    # === Heating (5) ===
    'heat_control_down': {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'down', 'param_type': 'temperature', 'judge': 'none'},
    'heat_control_up':   {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'up',   'param_type': 'temperature', 'judge': 'none'},
    'heat_query':        {'fn': 'heat_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none',        'judge': 'none'},
    'heat_schedule':     {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none',        'judge': 'none'},
    'heat_temp':         {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'temperature', 'judge': 'none'},

    # === Light (6) ===
    'light_control_dim': {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'down', 'param_type': 'brightness', 'judge': 'none'},
    'light_control_off': {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',  'param_type': 'none',       'judge': 'none'},
    'light_control_on':  {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none',       'judge': 'none'},
    'light_query':       {'fn': 'light_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none',       'judge': 'none'},
    'light_scene':       {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'mode',       'judge': 'none'},
    'light_schedule':    {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none',       'judge': 'none'},

    # === Ventilation (4) ===
    'vent_control_off': {'fn': 'vent_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',  'param_type': 'none', 'judge': 'none'},
    'vent_control_on':  {'fn': 'vent_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'vent_query':       {'fn': 'vent_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'vent_schedule':    {'fn': 'vent_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},

    # === Gas + Cooktop (4) ===
    'gas_control_lock':    {'fn': 'gas_control', 'exec_type': 'control_then_confirm', 'param_direction': 'close', 'param_type': 'none', 'judge': 'none'},
    'gas_control_open':    {'fn': 'gas_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},
    'gas_query':           {'fn': 'gas_control', 'exec_type': 'query_then_respond',   'param_direction': 'none',  'param_type': 'none', 'judge': 'none'},
    'cooktop_control_off': {'fn': 'gas_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',   'param_type': 'none', 'judge': 'none'},

    # === Door (3) ===
    'door_open':  {'fn': 'door_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},
    'door_query': {'fn': 'door_control', 'exec_type': 'query_then_respond',   'param_direction': 'none',  'param_type': 'none', 'judge': 'none'},
    'lobby_open': {'fn': 'door_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},

    # === Curtain + Louver (4) ===
    'curtain_control_close': {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'close', 'param_type': 'none', 'judge': 'none'},
    'curtain_control_open':  {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},
    'louver_control_close':  {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'close', 'param_type': 'none', 'judge': 'none'},
    'louver_control_open':   {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},

    # === Outlet → light_control (2) ===
    'outlet_control_off': {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off', 'param_type': 'none', 'judge': 'none'},
    'outlet_control_on':  {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',  'param_type': 'none', 'judge': 'none'},

    # === Security (3+4) ===
    'security_mode_home': {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'off',  'param_type': 'none', 'judge': 'none'},
    'security_mode_out':  {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'security_query':     {'fn': 'security_mode', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'access_history':     {'fn': 'security_mode', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'card_manage':        {'fn': 'security_mode', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'visitor_video':      {'fn': 'security_mode', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'emergency_report':   {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},

    # === Elevator (1) ===
    'elevator_call': {'fn': 'elevator_call', 'exec_type': 'control_then_confirm', 'param_direction': 'on', 'param_type': 'none', 'judge': 'none'},

    # === Schedule (2) ===
    'morning_call_query': {'fn': 'schedule_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'morning_call_set':   {'fn': 'schedule_manage', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},

    # === Weather (6) ===
    'weather_today':       {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'weather_tomorrow':    {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'weather_week':        {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'weather_rain':        {'fn': 'weather_query', 'exec_type': 'query_then_judge',   'param_direction': 'none', 'param_type': 'none', 'judge': 'outdoor_activity'},
    'weather_temperature': {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'airquality_query':    {'fn': 'weather_query', 'exec_type': 'query_then_judge',   'param_direction': 'none', 'param_type': 'none', 'judge': 'air_quality'},

    # === Info/Query (6+4) ===
    'news_query':        {'fn': 'news_query',    'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'traffic_query':     {'fn': 'traffic_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'energy_query':      {'fn': 'energy_query',  'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'home_status_query': {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'time_query':        {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'notice_query':      {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'community_query':   {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'info_query':        {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'call_front':        {'fn': 'home_info',     'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'call_guard':        {'fn': 'home_info',     'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'call_neighbor':     {'fn': 'home_info',     'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'call_history':      {'fn': 'home_info',     'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === System (3) ===
    'manual_query':    {'fn': 'system_meta', 'exec_type': 'direct_respond',       'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'settings':        {'fn': 'system_meta', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'password_change': {'fn': 'system_meta', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},

    # === Market (2) ===
    'oil_query':   {'fn': 'market_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'stock_query': {'fn': 'market_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Medical (1) ===
    'medical_query': {'fn': 'medical_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Vehicle (5) ===
    'parking_query':        {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'ev_charge_query':      {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'ev_charge_history':    {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'visitor_car_query':    {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'visitor_car_register': {'fn': 'vehicle_manage', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},
    'car_access_history':   {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Unknown / General (2) ===
    'unknown': {'fn': 'unknown', 'exec_type': 'direct_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'general': {'fn': 'unknown', 'exec_type': 'direct_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
}


# ============================================================
# 발화문 기반 보정 (라벨만으로 부족한 param_type/judge 추론)
# ============================================================
def refine_from_utterance(utt: str, labels: dict) -> dict:
    """발화문 텍스트에서 param_type, judge를 추가 추론"""
    labels = dict(labels)  # copy

    # param_type 보정 (라벨이 none인데 발화에 힌트가 있을 때)
    if labels['param_type'] == 'none':
        if re.search(r'온도|도로|(\d+)\s*도', utt):
            labels['param_type'] = 'temperature'
        elif re.search(r'밝기|밝게|어둡게|은은|아늑', utt):
            labels['param_type'] = 'brightness'
        elif re.search(r'모드|제습|송풍|자동|냉방|난방', utt):
            labels['param_type'] = 'mode'
        elif re.search(r'풍량|세기|바람|볼륨|소리|약하게', utt):
            labels['param_type'] = 'speed'

    # judge 보정
    if labels['judge'] == 'none':
        if re.search(r'미세먼지|공기|환기해도|창문.*괜찮', utt):
            labels['judge'] = 'air_quality'
        elif re.search(r'옷|입고|겉옷|반팔|코트|뭐 입', utt):
            labels['judge'] = 'clothing'
        elif re.search(r'세차|나가도|캠핑|소풍|야외|산책', utt):
            labels['judge'] = 'outdoor_activity'
        elif re.search(r'주유|떨어질|오를까|가격', utt):
            labels['judge'] = 'cost_trend'

    # judge가 있으면 exec_type도 보정
    if labels['judge'] != 'none' and labels['exec_type'] == 'query_then_respond':
        labels['exec_type'] = 'query_then_judge'

    return labels


# ============================================================
# 변환 함수
# ============================================================
def convert_jsonl(input_path: str) -> tuple:
    """KoELECTRA JSONL → multi-head JSON 변환"""
    converted = []
    skipped = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            utt = entry['utterance']
            label = entry['label']

            if label not in LABEL_MAP:
                skipped.append({'line': line_no, 'label': label, 'utterance': utt})
                continue

            base_labels = LABEL_MAP[label]
            refined = refine_from_utterance(utt, base_labels)

            converted.append({
                'utterance': utt,
                'labels': refined,
                'source': 'koelectra_v8',
            })

    return converted, skipped


def write_report(train_data, val_data, train_skipped, val_skipped, report_path):
    """변환 통계 리포트"""
    lines = []
    lines.append("=" * 60)
    lines.append("KoELECTRA → Multi-head 변환 리포트")
    lines.append("=" * 60)
    lines.append(f"\n입력:")
    lines.append(f"  train: {len(train_data) + len(train_skipped)}개 → 변환 {len(train_data)}개, 스킵 {len(train_skipped)}개")
    lines.append(f"  val:   {len(val_data) + len(val_skipped)}개 → 변환 {len(val_data)}개, 스킵 {len(val_skipped)}개")

    # fn 분포
    fn_c = Counter(d['labels']['fn'] for d in train_data)
    lines.append(f"\n--- train fn 분포 ({len(fn_c)}개 카테고리) ---")
    for k, v in fn_c.most_common():
        lines.append(f"  {k:20s}: {v:5d}")

    # exec_type 분포
    exec_c = Counter(d['labels']['exec_type'] for d in train_data)
    lines.append(f"\n--- train exec_type 분포 ---")
    for k, v in exec_c.most_common():
        lines.append(f"  {k:25s}: {v:5d}")

    # param_type 보정 통계
    param_refined = sum(1 for d in train_data if d['labels']['param_type'] != 'none')
    lines.append(f"\n--- param_type 분포 ---")
    pt_c = Counter(d['labels']['param_type'] for d in train_data)
    for k, v in pt_c.most_common():
        lines.append(f"  {k:15s}: {v:5d}")

    # judge 보정 통계
    judge_refined = sum(1 for d in train_data if d['labels']['judge'] != 'none')
    lines.append(f"\n--- judge 분포 ---")
    jc = Counter(d['labels']['judge'] for d in train_data)
    for k, v in jc.most_common():
        lines.append(f"  {k:20s}: {v:5d}")

    # 스킵된 라벨
    if train_skipped or val_skipped:
        all_skipped = train_skipped + val_skipped
        skip_labels = Counter(s['label'] for s in all_skipped)
        lines.append(f"\n--- 스킵된 라벨 ({len(all_skipped)}개) ---")
        for k, v in skip_labels.most_common():
            lines.append(f"  {k:25s}: {v:5d}")

    # 샘플
    lines.append(f"\n--- 변환 샘플 (처음 5개) ---")
    for d in train_data[:5]:
        lines.append(f"  [{d['labels']['fn']:15s}] {d['utterance'][:40]}")
        lines.append(f"    → exec={d['labels']['exec_type']}, dir={d['labels']['param_direction']}, "
                      f"pt={d['labels']['param_type']}, judge={d['labels']['judge']}")

    report = '\n'.join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    return report


# ============================================================
# main
# ============================================================
def main():
    base = Path(__file__).resolve().parent.parent
    koelectra_dir = base.parent / 'wewonnim' / 'koelectra_wallpad' / 'data'
    out_dir = base / 'data'

    train_in = koelectra_dir / 'train_v8.jsonl'
    val_in = koelectra_dir / 'val_v8.jsonl'

    if not train_in.exists():
        print(f"ERROR: {train_in} 없음")
        sys.exit(1)

    print(f"입력: {train_in}")
    print(f"      {val_in}")

    train_data, train_skip = convert_jsonl(str(train_in))
    val_data, val_skip = convert_jsonl(str(val_in))

    # 저장
    train_out = out_dir / 'koelectra_converted_train.json'
    val_out = out_dir / 'koelectra_converted_val.json'
    report_out = out_dir / 'koelectra_conversion_report.txt'

    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_out, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    report = write_report(train_data, val_data, train_skip, val_skip, str(report_out))
    print(report)

    print(f"\n출력:")
    print(f"  {train_out}  ({len(train_data)}개)")
    print(f"  {val_out}  ({len(val_data)}개)")
    print(f"  {report_out}")


if __name__ == '__main__':
    main()
