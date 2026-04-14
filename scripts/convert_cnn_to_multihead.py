#!/usr/bin/env python3
"""CNN 94-intent 데이터 → multi-head 라벨 변환
기존 train_ruel_v7.csv (16,879개) → train_multihead.json

출력 포맷:
{
  "utterance": "거실 불 켜줘",
  "flat_intent": "light_on",
  "labels": {
    "fn": "light_control",
    "exec_type": "control_then_confirm",
    "param_direction": "on",
    "param_type": "none",
    "judge": "none"
  }
}
"""
import csv, json, sys
from collections import Counter

# ============================================================
# 94 intent → multi-head 매핑 테이블
# ============================================================

INTENT_TO_FN = {
    # 디바이스 제어
    'light_on': 'light_control', 'light_off': 'light_control',
    'light_dim': 'light_control', 'light_brighten': 'light_control',
    'light_query': 'light_control', 'light_schedule': 'light_control',

    'heating_on': 'heat_control', 'heating_off': 'heat_control',
    'heating_up': 'heat_control', 'heating_down': 'heat_control',
    'heating_query': 'heat_control', 'heating_schedule_query': 'heat_control',
    'heating_schedule_set': 'heat_control', 'heating_schedule_cancel': 'heat_control',

    'ac_on': 'ac_control', 'ac_off': 'ac_control',
    'ac_temp': 'ac_control', 'ac_mode': 'ac_control',
    'ac_mode_schedule': 'ac_control', 'ac_mode_noroom': 'ac_control',
    'ac_wind': 'ac_control', 'ac_query': 'ac_control', 'ac_exception': 'ac_control',

    'ventilation_on': 'vent_control', 'ventilation_off': 'vent_control',
    'ventilation_mode': 'vent_control', 'ventilation_query': 'vent_control',
    'ventilation_schedule_query': 'vent_control', 'ventilation_schedule_set': 'vent_control',
    'ventilation_exception': 'vent_control',

    'curtain_open': 'curtain_control', 'curtain_close': 'curtain_control',
    'curtain_stop': 'curtain_control', 'curtain_query': 'curtain_control',
    'curtain_schedule': 'curtain_control',

    'door_open': 'door_control', 'doorlock_open': 'door_control', 'doorlock_query': 'door_control',

    'gas_close': 'gas_control', 'gas_query': 'gas_control',

    'security_activate': 'security_mode', 'security_query': 'security_mode',
    'security_return_set': 'security_mode',

    'elevator_call': 'elevator_call', 'elevator_query': 'elevator_call',

    'emergency': 'security_mode',

    # 시스템/어시스턴트
    'system_faq': 'system_meta', 'system_volume_set': 'home_info',
    'system_brightness_set': 'home_info', 'system_brightness_schedule': 'home_info',
    'system_exception': 'system_meta', 'password_change': 'system_meta',

    'alarm_query': 'schedule_manage', 'alarm_set': 'schedule_manage',

    'manual_capability': 'system_meta', 'manual_usage': 'system_meta',
    'manual_creator': 'system_meta', 'manual_name': 'system_meta',
    'manual_unsupported': 'system_meta',

    'home_status_query': 'home_info', 'notification_query': 'home_info',
    'notice_query': 'home_info', 'time_query': 'home_info',

    'complex_info': 'home_info', 'community_query': 'home_info',

    # 외부 API
    'weather_info': 'weather_query', 'weather_clothing': 'weather_query',
    'weather_activity': 'weather_query', 'weather_exception': 'weather_query',
    'dust_query': 'weather_query',

    'news_play': 'news_query', 'news_info': 'news_query', 'news_exception': 'news_query',

    'traffic_route_query': 'traffic_query', 'traffic_bus_query': 'traffic_query',
    'traffic_exception': 'traffic_query',

    'energy_usage_query': 'energy_query', 'energy_goal_set': 'energy_query',
    'energy_alert_on': 'energy_query', 'energy_alert_off': 'energy_query',

    'stock_index_query': 'market_query', 'stock_price_query': 'market_query',
    'stock_exception': 'market_query',
    'fuel_price_query': 'market_query', 'fuel_station_search': 'market_query',
    'fuel_exception': 'market_query',

    'medical_search': 'medical_query', 'medical_hours': 'medical_query',
    'medical_exception': 'medical_query',

    'ev_charging': 'vehicle_manage', 'car_history_query': 'vehicle_manage',
    'car_history_delete': 'vehicle_manage',
    'visitor_parking_query': 'vehicle_manage', 'visitor_parking_register': 'vehicle_manage',
}

INTENT_TO_EXEC = {
    # control_then_confirm
    'light_on': 'control_then_confirm', 'light_off': 'control_then_confirm',
    'light_dim': 'control_then_confirm', 'light_brighten': 'control_then_confirm',
    'light_schedule': 'control_then_confirm',
    'heating_on': 'control_then_confirm', 'heating_off': 'control_then_confirm',
    'heating_up': 'control_then_confirm', 'heating_down': 'control_then_confirm',
    'heating_schedule_set': 'control_then_confirm', 'heating_schedule_cancel': 'control_then_confirm',
    'ac_on': 'control_then_confirm', 'ac_off': 'control_then_confirm',
    'ac_temp': 'control_then_confirm', 'ac_mode': 'control_then_confirm',
    'ac_mode_schedule': 'control_then_confirm', 'ac_mode_noroom': 'control_then_confirm',
    'ac_wind': 'control_then_confirm',
    'ventilation_on': 'control_then_confirm', 'ventilation_off': 'control_then_confirm',
    'ventilation_mode': 'control_then_confirm', 'ventilation_schedule_set': 'control_then_confirm',
    'curtain_open': 'control_then_confirm', 'curtain_close': 'control_then_confirm',
    'curtain_stop': 'control_then_confirm', 'curtain_schedule': 'control_then_confirm',
    'door_open': 'control_then_confirm', 'doorlock_open': 'control_then_confirm',
    'gas_close': 'control_then_confirm',
    'security_activate': 'control_then_confirm', 'security_return_set': 'control_then_confirm',
    'emergency': 'control_then_confirm',
    'elevator_call': 'control_then_confirm',
    'alarm_set': 'control_then_confirm',
    'system_volume_set': 'control_then_confirm',
    'system_brightness_set': 'control_then_confirm', 'system_brightness_schedule': 'control_then_confirm',
    'password_change': 'control_then_confirm',
    'energy_goal_set': 'control_then_confirm',
    'energy_alert_on': 'control_then_confirm', 'energy_alert_off': 'control_then_confirm',
    'car_history_delete': 'control_then_confirm',
    'visitor_parking_register': 'control_then_confirm',

    # query_then_respond
    'light_query': 'query_then_respond',
    'heating_query': 'query_then_respond', 'heating_schedule_query': 'query_then_respond',
    'ac_query': 'query_then_respond',
    'ventilation_query': 'query_then_respond', 'ventilation_schedule_query': 'query_then_respond',
    'curtain_query': 'query_then_respond', 'doorlock_query': 'query_then_respond',
    'gas_query': 'query_then_respond', 'security_query': 'query_then_respond',
    'elevator_query': 'query_then_respond', 'alarm_query': 'query_then_respond',
    'home_status_query': 'query_then_respond', 'notification_query': 'query_then_respond',
    'notice_query': 'query_then_respond',
    'weather_info': 'query_then_respond', 'dust_query': 'query_then_respond',
    'news_play': 'query_then_respond', 'news_info': 'query_then_respond',
    'traffic_route_query': 'query_then_respond', 'traffic_bus_query': 'query_then_respond',
    'energy_usage_query': 'query_then_respond',
    'stock_index_query': 'query_then_respond', 'stock_price_query': 'query_then_respond',
    'fuel_price_query': 'query_then_respond', 'fuel_station_search': 'query_then_respond',
    'medical_search': 'query_then_respond', 'medical_hours': 'query_then_respond',
    'ev_charging': 'query_then_respond', 'car_history_query': 'query_then_respond',
    'visitor_parking_query': 'query_then_respond',
    'complex_info': 'query_then_respond', 'community_query': 'query_then_respond',

    # query_then_judge
    'weather_clothing': 'query_then_judge',
    'weather_activity': 'query_then_judge',

    # direct_respond
    'manual_capability': 'direct_respond', 'manual_usage': 'direct_respond',
    'manual_creator': 'direct_respond', 'manual_name': 'direct_respond',
    'manual_unsupported': 'direct_respond',
    'system_faq': 'direct_respond', 'system_exception': 'direct_respond',
    'time_query': 'direct_respond',
    'ac_exception': 'direct_respond', 'ventilation_exception': 'direct_respond',
    'weather_exception': 'direct_respond', 'news_exception': 'direct_respond',
    'traffic_exception': 'direct_respond',
    'stock_exception': 'direct_respond', 'fuel_exception': 'direct_respond',
    'medical_exception': 'direct_respond',
}

INTENT_TO_DIRECTION = {
    # on
    'light_on': 'on', 'heating_on': 'on', 'ac_on': 'on',
    'ventilation_on': 'on', 'security_activate': 'on', 'emergency': 'on',
    'elevator_call': 'on', 'energy_alert_on': 'on',
    # off
    'light_off': 'off', 'heating_off': 'off', 'ac_off': 'off',
    'ventilation_off': 'off', 'heating_schedule_cancel': 'off', 'energy_alert_off': 'off',
    # up
    'light_brighten': 'up', 'heating_up': 'up',
    # down
    'light_dim': 'down', 'heating_down': 'down',
    # open
    'curtain_open': 'open', 'door_open': 'open', 'doorlock_open': 'open',
    # close
    'curtain_close': 'close', 'gas_close': 'close',
    # stop
    'curtain_stop': 'stop',
    # set
    'ac_temp': 'set', 'ac_mode': 'set', 'ac_mode_schedule': 'set',
    'ac_mode_noroom': 'set', 'ac_wind': 'set',
    'ventilation_mode': 'set', 'ventilation_schedule_set': 'set',
    'light_schedule': 'set', 'heating_schedule_set': 'set', 'curtain_schedule': 'set',
    'security_return_set': 'set', 'alarm_set': 'set',
    'system_volume_set': 'set', 'system_brightness_set': 'set',
    'system_brightness_schedule': 'set', 'password_change': 'set',
    'energy_goal_set': 'set', 'visitor_parking_register': 'set',
    'car_history_delete': 'set',
}

INTENT_TO_PARAM = {
    'light_dim': 'brightness', 'light_brighten': 'brightness',
    'system_brightness_set': 'brightness', 'system_brightness_schedule': 'brightness',
    'heating_up': 'temperature', 'heating_down': 'temperature', 'ac_temp': 'temperature',
    'ac_mode': 'mode', 'ac_mode_noroom': 'mode', 'ac_mode_schedule': 'mode',
    'ventilation_mode': 'mode',
    'ac_wind': 'speed',
    'system_volume_set': 'speed',  # volume은 speed로 매핑
}

INTENT_TO_JUDGE = {
    'weather_clothing': 'clothing',
    'weather_activity': 'outdoor_activity',
}


def convert(input_csv, output_json):
    data = []
    missing = Counter()

    with open(input_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt = row['question'].strip()
            intent = row['label'].strip()

            fn = INTENT_TO_FN.get(intent)
            if fn is None:
                missing[intent] += 1
                continue

            exec_type = INTENT_TO_EXEC.get(intent, 'query_then_respond')
            direction = INTENT_TO_DIRECTION.get(intent, 'none')
            param_type = INTENT_TO_PARAM.get(intent, 'none')
            judge = INTENT_TO_JUDGE.get(intent, 'none')

            data.append({
                'utterance': utt,
                'flat_intent': intent,
                'labels': {
                    'fn': fn,
                    'exec_type': exec_type,
                    'param_direction': direction,
                    'param_type': param_type,
                    'judge': judge,
                }
            })

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 통계
    fn_c = Counter(d['labels']['fn'] for d in data)
    exec_c = Counter(d['labels']['exec_type'] for d in data)
    dir_c = Counter(d['labels']['param_direction'] for d in data)
    param_c = Counter(d['labels']['param_type'] for d in data)
    judge_c = Counter(d['labels']['judge'] for d in data)

    print(f"변환 완료: {len(data)}개 ({len(missing)}개 intent 미매핑)")
    if missing:
        print(f"  미매핑: {dict(missing)}")

    print(f"\n=== fn ({len(fn_c)}개 클래스) ===")
    for k, v in fn_c.most_common():
        print(f"  {k}: {v}")

    print(f"\n=== exec_type ===")
    for k, v in exec_c.most_common():
        print(f"  {k}: {v}")

    print(f"\n=== param_direction ===")
    for k, v in dir_c.most_common():
        print(f"  {k}: {v}")

    print(f"\n=== param_type ===")
    for k, v in param_c.most_common():
        print(f"  {k}: {v}")

    print(f"\n=== judge ===")
    for k, v in judge_c.most_common():
        print(f"  {k}: {v}")

    return data


if __name__ == '__main__':
    input_csv = 'data/archive/train_ruel_v7.csv'
    output_json = 'data/train_multihead_v1.json'
    convert(input_csv, output_json)
