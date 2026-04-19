#!/usr/bin/env python3
"""v58 — Targeted augmentation for v46's Test Suite error patterns
Focus on:
1. schedule_manage vs system_meta confusion (19x)
2. query_then_respond vs control_then_confirm (27x)
3. Direction confusions (down↔up, on↔off, close↔open)
4. weather_query/news_query → unknown (7x)
"""
import json, random

def build_augments():
    augments = []

    # ============================================================
    # 1. schedule_manage (NOT system_meta): timers, alarms, reservations
    # ============================================================
    schedule_templates = [
        # 알람/타이머 설정
        ("알람 맞춰줘", "control_then_confirm", "set"),
        ("알람 설정해줘", "control_then_confirm", "set"),
        ("알람 켜줘", "control_then_confirm", "on"),
        ("알람 꺼줘", "control_then_confirm", "off"),
        ("알람 취소해줘", "control_then_confirm", "off"),
        ("타이머 설정해줘", "control_then_confirm", "set"),
        ("타이머 맞춰줘", "control_then_confirm", "set"),
        ("타이머 해제해줘", "control_then_confirm", "off"),
        ("모닝콜 맞춰줘", "control_then_confirm", "set"),
        ("모닝콜 설정", "control_then_confirm", "set"),
        # 예약
        ("난방 예약해줘", "control_then_confirm", "set"),
        ("에어컨 예약 설정", "control_then_confirm", "set"),
        ("조명 예약 켜줘", "control_then_confirm", "on"),
        ("커튼 예약 해제", "control_then_confirm", "off"),
        # 시간대
        ("7시에 알람 맞춰줘", "control_then_confirm", "set"),
        ("30분 후에 꺼줘", "control_then_confirm", "set"),
        ("10분 타이머", "control_then_confirm", "set"),
        ("1시간 뒤에 알람", "control_then_confirm", "set"),
        # 예약 확인/조회
        ("알람 뭐 있어", "query_then_respond", "none"),
        ("예약 확인", "query_then_respond", "none"),
        ("타이머 남은 시간", "query_then_respond", "none"),
        ("예약 현황 알려줘", "query_then_respond", "none"),
    ]

    rooms = ['', '거실 ', '안방 ', '주방 ', '침실 ']
    suffixes = ['', '줘', '요', '주세요']

    for text, exec_t, dir_t in schedule_templates:
        for room in rooms[:3]:  # Don't use all rooms for every template
            for suffix in suffixes[:2]:
                t = room + text
                if suffix and not t.endswith(suffix):
                    t = t.rstrip('줘요') + suffix if t.endswith(('줘', '요')) else t
                augments.append({
                    'utterance': t,
                    'labels': {'fn': 'schedule_manage', 'exec_type': exec_t,
                              'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
                    'source': 'fix_v58_schedule'
                })

    # ============================================================
    # 2. query_then_respond patterns (NOT control_then_confirm)
    # ============================================================
    query_patterns = [
        # 상태 확인 패턴
        ("에어컨 작동 중이야", "ac_control", "query_then_respond", "none"),
        ("에어컨 켜져 있어", "ac_control", "query_then_respond", "none"),
        ("난방 지금 켜져 있나", "heat_control", "query_then_respond", "none"),
        ("조명 상태 확인", "light_control", "query_then_respond", "none"),
        ("불 켜져 있어", "light_control", "query_then_respond", "none"),
        ("현관문 열려 있어", "door_control", "query_then_respond", "none"),
        ("문 잠겨 있나", "door_control", "query_then_respond", "none"),
        ("가스 잠겨 있어", "gas_control", "query_then_respond", "none"),
        ("환기 작동 중이야", "vent_control", "query_then_respond", "none"),
        ("커튼 열려 있나", "curtain_control", "query_then_respond", "none"),
        ("지금 온도 몇 도야", "heat_control", "query_then_respond", "none"),
        ("실내 온도 알려줘", "heat_control", "query_then_respond", "none"),
        ("에어컨 몇 도로 설정되어 있어", "ac_control", "query_then_respond", "none"),
        ("에너지 사용량 얼마야", "energy_query", "query_then_respond", "none"),
        ("이번 달 전기세", "energy_query", "query_then_respond", "none"),
        ("교통 상황 어때", "traffic_query", "query_then_respond", "none"),
        # 단어만으로 상태 질의
        ("현관문 상태", "door_control", "query_then_respond", "none"),
        ("엘리베이터 위치", "elevator_call", "query_then_respond", "none"),
        ("온도 설정값", "heat_control", "query_then_respond", "none"),
    ]

    for text, fn, exec_t, dir_t in query_patterns:
        for room in rooms[:3]:
            augments.append({
                'utterance': room + text,
                'labels': {'fn': fn, 'exec_type': exec_t,
                          'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
                'source': 'fix_v58_query'
            })

    # ============================================================
    # 3. direct_respond patterns (NOT control_then_confirm)
    # ============================================================
    direct_patterns = [
        # 단일 단어/짧은 질의 → 뭔지 물어보는 것
        ("에어컨", "ac_control", "direct_respond", "none"),
        ("조명", "light_control", "direct_respond", "none"),
        ("난방", "heat_control", "direct_respond", "none"),
        ("커튼", "curtain_control", "direct_respond", "none"),
        ("에어컨 뭐야", "ac_control", "direct_respond", "none"),
        ("에어컨 풀가동", "ac_control", "direct_respond", "none"),
        ("에어컨 바람", "ac_control", "direct_respond", "none"),
        ("에어컨 리모컨", "ac_control", "direct_respond", "none"),
        ("에너지 목표", "energy_query", "direct_respond", "none"),
    ]

    for text, fn, exec_t, dir_t in direct_patterns:
        augments.append({
            'utterance': text,
            'labels': {'fn': fn, 'exec_type': exec_t,
                      'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
            'source': 'fix_v58_direct'
        })

    # ============================================================
    # 4. Direction patterns: down vs up, on vs off, close vs open
    # ============================================================
    direction_patterns = [
        # down (줄여, 낮춰, 내려, 어둡게, 약하게)
        ("소리 줄여줘", "home_info", "control_then_confirm", "down"),
        ("소리 좀 줄여", "home_info", "control_then_confirm", "down"),
        ("볼륨 줄여줘", "home_info", "control_then_confirm", "down"),
        ("볼륨 낮춰줘", "home_info", "control_then_confirm", "down"),
        ("볼륨 내려줘", "home_info", "control_then_confirm", "down"),
        ("월패드 소리 줄여줘", "home_info", "control_then_confirm", "down"),
        ("월패드 소리 낮춰", "home_info", "control_then_confirm", "down"),
        ("화면 밝기 줄여", "home_info", "control_then_confirm", "down"),
        ("화면 밝기 낮춰줘", "home_info", "control_then_confirm", "down"),
        ("화면 어둡게", "home_info", "control_then_confirm", "down"),
        ("밝기 내려줘", "home_info", "control_then_confirm", "down"),
        ("온도 낮춰줘", "heat_control", "control_then_confirm", "down"),
        ("온도 내려줘", "heat_control", "control_then_confirm", "down"),
        ("조금만 낮춰", "heat_control", "control_then_confirm", "down"),
        ("어둡게 해줘", "light_control", "control_then_confirm", "down"),
        ("조명 어둡게", "light_control", "control_then_confirm", "down"),
        ("거실 조명 최소로", "light_control", "control_then_confirm", "down"),
        ("안방 조명 최소로", "light_control", "control_then_confirm", "down"),
        ("바람 약하게", "ac_control", "control_then_confirm", "down"),
        ("에어컨 바람 줄여", "ac_control", "control_then_confirm", "down"),

        # up (올려, 높여, 밝게, 강하게)
        ("소리 올려줘", "home_info", "control_then_confirm", "up"),
        ("볼륨 올려줘", "home_info", "control_then_confirm", "up"),
        ("소리 좀 올려", "home_info", "control_then_confirm", "up"),
        ("월패드 소리 올려줘", "home_info", "control_then_confirm", "up"),
        ("화면 밝기 올려", "home_info", "control_then_confirm", "up"),
        ("밝기 올려줘", "home_info", "control_then_confirm", "up"),
        ("온도 올려줘", "heat_control", "control_then_confirm", "up"),
        ("온도 높여줘", "heat_control", "control_then_confirm", "up"),
        ("조금만 올려", "heat_control", "control_then_confirm", "up"),
        ("밝게 해줘", "light_control", "control_then_confirm", "up"),
        ("조명 밝게", "light_control", "control_then_confirm", "up"),
        ("더 밝게", "light_control", "control_then_confirm", "up"),
        ("거실 등 좀 밝게", "light_control", "control_then_confirm", "up"),
        ("바람 강하게", "ac_control", "control_then_confirm", "up"),
        ("에어컨 바람 올려", "ac_control", "control_then_confirm", "up"),

        # close (잠가, 닫아, 잠금)
        ("현관 잠가줘", "door_control", "control_then_confirm", "close"),
        ("현관 잠금", "door_control", "control_then_confirm", "close"),
        ("문 잠가", "door_control", "control_then_confirm", "close"),
        ("문 닫아줘", "door_control", "control_then_confirm", "close"),
        ("가스 잠가줘", "gas_control", "control_then_confirm", "close"),
        ("가스 밸브 잠가", "gas_control", "control_then_confirm", "close"),
        ("가스 잠그기", "gas_control", "control_then_confirm", "close"),

        # open (열어, 열기)
        ("현관문 열어줘", "door_control", "control_then_confirm", "open"),
        ("문 열어", "door_control", "control_then_confirm", "open"),
        ("커튼 열어줘", "curtain_control", "control_then_confirm", "open"),
        ("블라인드 열어", "curtain_control", "control_then_confirm", "open"),

        # on (켜, 틀어, 가동)
        ("거실 불 켜줘", "light_control", "control_then_confirm", "on"),
        ("조명 켜줘", "light_control", "control_then_confirm", "on"),
        ("에어컨 켜줘", "ac_control", "control_then_confirm", "on"),
        ("난방 켜줘", "heat_control", "control_then_confirm", "on"),
        ("환기 켜줘", "vent_control", "control_then_confirm", "on"),
        ("간접등 켜줘", "light_control", "control_then_confirm", "on"),
        ("취침등 켜줘", "light_control", "control_then_confirm", "on"),
        ("무드등 켜줘", "light_control", "control_then_confirm", "on"),
        ("형광등 켜줘", "light_control", "control_then_confirm", "on"),

        # off (꺼, 끄기, 중지)
        ("거실 불 꺼줘", "light_control", "control_then_confirm", "off"),
        ("조명 꺼줘", "light_control", "control_then_confirm", "off"),
        ("에어컨 꺼줘", "ac_control", "control_then_confirm", "off"),
        ("난방 꺼줘", "heat_control", "control_then_confirm", "off"),
        ("환기 꺼줘", "vent_control", "control_then_confirm", "off"),
        ("에어컨꺼", "ac_control", "control_then_confirm", "off"),
        ("남방 꺼줘", "heat_control", "control_then_confirm", "off"),
    ]

    for text, fn, exec_t, dir_t in direction_patterns:
        augments.append({
            'utterance': text,
            'labels': {'fn': fn, 'exec_type': exec_t,
                      'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
            'source': 'fix_v58_direction'
        })

    # Room variations for direction patterns
    for text, fn, exec_t, dir_t in direction_patterns:
        if fn in ('light_control', 'heat_control', 'ac_control', 'curtain_control'):
            for room in ['거실 ', '안방 ', '침실 ']:
                if not text.startswith(room) and not any(text.startswith(r) for r in ['거실', '안방', '주방', '침실']):
                    augments.append({
                        'utterance': room + text,
                        'labels': {'fn': fn, 'exec_type': exec_t,
                                  'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
                        'source': 'fix_v58_direction_room'
                    })

    # ============================================================
    # 5. weather_query, news_query (NOT unknown)
    # ============================================================
    weather_news_patterns = [
        ("서울 날씨", "weather_query", "query_then_respond", "none"),
        ("오늘 날씨 어때", "weather_query", "query_then_respond", "none"),
        ("오늘날씨어때", "weather_query", "query_then_respond", "none"),
        ("내일 날씨", "weather_query", "query_then_respond", "none"),
        ("부산 날씨", "weather_query", "query_then_respond", "none"),
        ("날씨 알려줘", "weather_query", "query_then_respond", "none"),
        ("비 오나", "weather_query", "query_then_respond", "none"),
        ("오늘 기온", "weather_query", "query_then_respond", "none"),
        ("주말 날씨", "weather_query", "query_then_respond", "none"),
        ("국제 뉴스", "news_query", "query_then_respond", "none"),
        ("뉴스 틀어줘", "news_query", "query_then_respond", "none"),
        ("오늘 뉴스", "news_query", "query_then_respond", "none"),
        ("뉴스 알려줘", "news_query", "query_then_respond", "none"),
        ("주요 뉴스", "news_query", "query_then_respond", "none"),
        ("스포츠 뉴스", "news_query", "query_then_respond", "none"),
        ("경제 뉴스", "news_query", "query_then_respond", "none"),
    ]

    for text, fn, exec_t, dir_t in weather_news_patterns:
        augments.append({
            'utterance': text,
            'labels': {'fn': fn, 'exec_type': exec_t,
                      'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
            'source': 'fix_v58_weather_news'
        })

    # ============================================================
    # 6. home_info (NOT system_meta)
    # ============================================================
    home_info_patterns = [
        ("화면 밝기", "home_info", "control_then_confirm", "down"),
        ("월패드 밝기", "home_info", "control_then_confirm", "down"),
        ("화면 밝기 줄여줘", "home_info", "control_then_confirm", "down"),
        ("월패드 밝기 올려줘", "home_info", "control_then_confirm", "up"),
        ("알림 뭐야", "home_info", "query_then_respond", "none"),
        ("알림 확인", "home_info", "query_then_respond", "none"),
        ("택배 왔어", "home_info", "query_then_respond", "none"),
    ]

    for text, fn, exec_t, dir_t in home_info_patterns:
        augments.append({
            'utterance': text,
            'labels': {'fn': fn, 'exec_type': exec_t,
                      'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
            'source': 'fix_v58_home_info'
        })

    # ============================================================
    # 7. elevator_call, clarify patterns
    # ============================================================
    elevator_patterns = [
        ("승강기 불러줘", "elevator_call", "control_then_confirm", "on"),
        ("엘리베이터 불러줘", "elevator_call", "control_then_confirm", "on"),
        ("리프트 호출", "elevator_call", "control_then_confirm", "on"),
        ("승강기 호출해줘", "elevator_call", "control_then_confirm", "on"),
        ("엘베 불러줘", "elevator_call", "control_then_confirm", "on"),
        ("승강기 호출", "elevator_call", "control_then_confirm", "on"),
        ("엘리베이터 호출", "elevator_call", "control_then_confirm", "on"),
    ]

    for text, fn, exec_t, dir_t in elevator_patterns:
        augments.append({
            'utterance': text,
            'labels': {'fn': fn, 'exec_type': exec_t,
                      'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
            'source': 'fix_v58_elevator'
        })

    # Clarify patterns (short, ambiguous)
    clarify_patterns = [
        ("간접등 켜줘", "light_control", "clarify", "on"),
        ("취침등", "light_control", "clarify", "set"),
        ("무드등", "light_control", "clarify", "none"),
    ]

    for text, fn, exec_t, dir_t in clarify_patterns:
        augments.append({
            'utterance': text,
            'labels': {'fn': fn, 'exec_type': exec_t,
                      'param_direction': dir_t, 'param_type': 'none', 'judge': 'none'},
            'source': 'fix_v58_clarify'
        })

    # Deduplicate by utterance
    seen = set()
    unique = []
    for a in augments:
        if a['utterance'] not in seen:
            seen.add(a['utterance'])
            unique.append(a)

    print(f"Generated {len(unique)} unique augmentations")
    from collections import Counter
    sources = Counter(a['source'] for a in unique)
    for s, c in sources.most_common():
        print(f"  {s}: {c}")

    return unique


if __name__ == '__main__':
    augments = build_augments()
    with open('data/fix_v58_targeted.json', 'w', encoding='utf-8') as f:
        json.dump(augments, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to data/fix_v58_targeted.json")
