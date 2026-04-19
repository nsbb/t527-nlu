#!/usr/bin/env python3
"""KoELECTRA v8 79개 flat label → Multi-head 5축 자동 매핑
결과: 자동변환 가능 비율 + 애매한 것들 목록"""

# KoELECTRA v8 label → multi-head 5축 매핑
# 규칙: label 이름에서 fn, direction 추출 + exec_type/param_type/judge 추론

KOELECTRA_TO_MULTIHEAD = {
    # === AC ===
    'ac_control_cool':  {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'mode',        'judge': 'none'},
    'ac_control_off':   {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',   'param_type': 'none',        'judge': 'none'},
    'ac_control_on':    {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',    'param_type': 'none',        'judge': 'none'},
    'ac_control_warm':  {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'mode',        'judge': 'none'},
    'ac_mode':          {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'mode',        'judge': 'none'},
    'ac_query':         {'fn': 'ac_control', 'exec_type': 'query_then_respond',   'param_direction': 'none',  'param_type': 'none',        'judge': 'none'},
    'ac_schedule':      {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'none',        'judge': 'none'},
    'ac_temp':          {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'temperature', 'judge': 'none'},
    'ac_wind':          {'fn': 'ac_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',   'param_type': 'speed',       'judge': 'none'},

    # === Heating ===
    'heat_control_down': {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'down', 'param_type': 'temperature', 'judge': 'none'},
    'heat_control_up':   {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'up',   'param_type': 'temperature', 'judge': 'none'},
    'heat_query':        {'fn': 'heat_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none',        'judge': 'none'},
    'heat_schedule':     {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none',        'judge': 'none'},
    'heat_temp':         {'fn': 'heat_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'temperature', 'judge': 'none'},

    # === Light ===
    'light_control_dim': {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'down', 'param_type': 'brightness', 'judge': 'none'},
    'light_control_off': {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',  'param_type': 'none',       'judge': 'none'},
    'light_control_on':  {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none',       'judge': 'none'},
    'light_query':       {'fn': 'light_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none',       'judge': 'none'},
    'light_scene':       {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'mode',       'judge': 'none'},
    'light_schedule':    {'fn': 'light_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none',       'judge': 'none'},

    # === Ventilation ===
    'vent_control_off': {'fn': 'vent_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off',  'param_type': 'none', 'judge': 'none'},
    'vent_control_on':  {'fn': 'vent_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'vent_query':       {'fn': 'vent_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'vent_schedule':    {'fn': 'vent_control', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},

    # === Gas ===
    'gas_control_lock': {'fn': 'gas_control', 'exec_type': 'control_then_confirm', 'param_direction': 'close', 'param_type': 'none', 'judge': 'none'},
    'gas_control_open': {'fn': 'gas_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},
    'gas_query':        {'fn': 'gas_control', 'exec_type': 'query_then_respond',   'param_direction': 'none',  'param_type': 'none', 'judge': 'none'},

    # === Door ===
    'door_open':   {'fn': 'door_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},
    'door_query':  {'fn': 'door_control', 'exec_type': 'query_then_respond',   'param_direction': 'none',  'param_type': 'none', 'judge': 'none'},
    'lobby_open':  {'fn': 'door_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},

    # === Curtain / Louver ===
    'curtain_control_close': {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'close', 'param_type': 'none', 'judge': 'none'},
    'curtain_control_open':  {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},
    'louver_control_close':  {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'close', 'param_type': 'none', 'judge': 'none'},
    'louver_control_open':   {'fn': 'curtain_control', 'exec_type': 'control_then_confirm', 'param_direction': 'open',  'param_type': 'none', 'judge': 'none'},

    # === Security ===
    'security_mode_home': {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'off',  'param_type': 'none', 'judge': 'none'},
    'security_mode_out':  {'fn': 'security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'security_query':     {'fn': 'security_mode', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Elevator ===
    'elevator_call': {'fn': 'elevator_call', 'exec_type': 'control_then_confirm', 'param_direction': 'on', 'param_type': 'none', 'judge': 'none'},

    # === Schedule ===
    'morning_call_query': {'fn': 'schedule_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'morning_call_set':   {'fn': 'schedule_manage', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},

    # === Weather ===
    'weather_today':       {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'weather_tomorrow':    {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'weather_week':        {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'weather_rain':        {'fn': 'weather_query', 'exec_type': 'query_then_judge',   'param_direction': 'none', 'param_type': 'none', 'judge': 'outdoor_activity'},
    'weather_temperature': {'fn': 'weather_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'airquality_query':    {'fn': 'weather_query', 'exec_type': 'query_then_judge',   'param_direction': 'none', 'param_type': 'none', 'judge': 'air_quality'},

    # === Info/Query ===
    'news_query':        {'fn': 'news_query',    'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'traffic_query':     {'fn': 'traffic_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'energy_query':      {'fn': 'energy_query',  'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'home_status_query': {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'time_query':        {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'notice_query':      {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'community_query':   {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'info_query':        {'fn': 'home_info',     'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === System ===
    'manual_query':    {'fn': 'system_meta', 'exec_type': 'direct_respond',       'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'settings':        {'fn': 'system_meta', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'password_change': {'fn': 'system_meta', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},

    # === Market ===
    'oil_query':   {'fn': 'market_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'stock_query': {'fn': 'market_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Medical ===
    'medical_query': {'fn': 'medical_query', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Vehicle ===
    'parking_query':        {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'ev_charge_query':      {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'ev_charge_history':    {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'visitor_car_query':    {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'visitor_car_register': {'fn': 'vehicle_manage', 'exec_type': 'control_then_confirm', 'param_direction': 'set',  'param_type': 'none', 'judge': 'none'},
    'car_access_history':   {'fn': 'vehicle_manage', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # === Unknown / General ===
    'unknown': {'fn': 'unknown', 'exec_type': 'direct_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'general': {'fn': 'unknown', 'exec_type': 'direct_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # ============================================================
    # 아래는 우리 fn 축에 정확히 맞는 카테고리가 없어서 결정 필요
    # ============================================================

    # --- 통화 (call) → fn 축에 없음. home_info? 새 fn "call_control"? ---
    'call_front':    {'fn': '???_call_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'call_guard':    {'fn': '???_call_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'call_neighbor': {'fn': '???_call_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',   'param_type': 'none', 'judge': 'none'},
    'call_history':  {'fn': '???_call_control', 'exec_type': 'query_then_respond',   'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},

    # --- 출입/카드 → security_mode로 통합? ---
    'access_history': {'fn': '???_security_mode', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'card_manage':    {'fn': '???_security_mode', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'visitor_video':  {'fn': '???_security_mode', 'exec_type': 'query_then_respond', 'param_direction': 'none', 'param_type': 'none', 'judge': 'none'},
    'emergency_report': {'fn': '???_security_mode', 'exec_type': 'control_then_confirm', 'param_direction': 'on', 'param_type': 'none', 'judge': 'none'},

    # --- 쿡탑/콘센트 → fn 축에 없음. 새 fn? 기존에 합침? ---
    'cooktop_control_off': {'fn': '???_gas_control',   'exec_type': 'control_then_confirm', 'param_direction': 'off', 'param_type': 'none', 'judge': 'none'},
    'outlet_control_off':  {'fn': '???_outlet_control', 'exec_type': 'control_then_confirm', 'param_direction': 'off', 'param_type': 'none', 'judge': 'none'},
    'outlet_control_on':   {'fn': '???_outlet_control', 'exec_type': 'control_then_confirm', 'param_direction': 'on',  'param_type': 'none', 'judge': 'none'},
}


def analyze():
    total = len(KOELECTRA_TO_MULTIHEAD)
    auto = sum(1 for v in KOELECTRA_TO_MULTIHEAD.values() if not any('???' in str(vv) for vv in v.values()))
    ambiguous = [(k, v) for k, v in KOELECTRA_TO_MULTIHEAD.items() if any('???' in str(vv) for vv in v.values())]

    print(f"=== KoELECTRA v8 → Multi-head 5축 매핑 결과 ===\n")
    print(f"전체: {total}개 label")
    print(f"자동 변환 가능: {auto}개 ({auto/total*100:.0f}%)")
    print(f"결정 필요:      {len(ambiguous)}개 ({len(ambiguous)/total*100:.0f}%)")

    print(f"\n--- 결정 필요한 {len(ambiguous)}개 ---")
    for label, mapping in ambiguous:
        fn_val = mapping['fn']
        suggestion = fn_val.replace('???_', '')
        print(f"  {label:25s} → fn 후보: {suggestion}")

    # fn 축에 추가해야 할 후보
    new_fns = set()
    for label, mapping in ambiguous:
        fn_val = mapping['fn'].replace('???_', '')
        if fn_val not in ['security_mode', 'gas_control']:  # 기존에 있는건 제외
            new_fns.add(fn_val)

    print(f"\n--- fn 축 확장 옵션 ---")
    print(f"옵션 A: fn 축에 새 카테고리 추가 → {new_fns}")
    print(f"         20 → {20 + len(new_fns)}개 (영향 작음)")
    print(f"옵션 B: 기존 카테고리에 통합")
    print(f"         call → home_info, outlet → light_control, cooktop → gas_control")
    print(f"         access/card/visitor_video/emergency → security_mode")

    # exec_type, param_direction, param_type, judge는 전부 자동
    print(f"\n--- 5축 별 자동 변환율 ---")
    for axis in ['exec_type', 'param_direction', 'param_type', 'judge']:
        ok = sum(1 for v in KOELECTRA_TO_MULTIHEAD.values() if '???' not in str(v[axis]))
        print(f"  {axis:18s}: {ok}/{total} ({ok/total*100:.0f}%)")
    fn_ok = sum(1 for v in KOELECTRA_TO_MULTIHEAD.values() if '???' not in str(v['fn']))
    print(f"  {'fn':18s}: {fn_ok}/{total} ({fn_ok/total*100:.0f}%)")


if __name__ == '__main__':
    analyze()
