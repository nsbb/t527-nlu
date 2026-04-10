#!/usr/bin/env python3
"""Judge Engine — 판단형 발화 처리
handoff_semantic_action_parser_complete.md Section 8.2 기반
"""

JUDGE_RULES = {
    "outdoor_activity": {
        "conditions": [
            {"field": "precip_prob", "op": ">", "value": 40, "reason": "강수확률 {v}%"},
            {"field": "pm10_grade", "op": ">=", "value": 3, "reason": "미세먼지 나쁨"},
            {"field": "wind_speed", "op": ">", "value": 14, "reason": "강풍 예보"},
        ],
        "ok_response": "기온과 대기질 모두 양호합니다. 외출하기에 무리가 없습니다.",
        "fail_template": "{reason}이므로 야외활동은 권장하지 않습니다."
    },
    "clothing": {
        "ranges": [
            {"min": 28, "max": 99, "clothing": "반팔/반바지가 적당합니다"},
            {"min": 23, "max": 27, "clothing": "얇은 긴팔 또는 반팔이 적당합니다"},
            {"min": 17, "max": 22, "clothing": "가벼운 겉옷을 준비하세요"},
            {"min": 12, "max": 16, "clothing": "자켓이나 가디건을 챙기세요"},
            {"min": 5, "max": 11, "clothing": "코트나 패딩이 필요합니다"},
            {"min": -99, "max": 4, "clothing": "두꺼운 외투와 방한용품이 필요합니다"},
        ],
        "gap_threshold": 10,
        "gap_note": "일교차가 크니 겉옷을 함께 준비하세요.",
        "template": "오늘 {location} 기온은 최고 {high}도, 최저 {low}도입니다. {clothing}"
    },
    "air_quality": {
        "conditions": [
            {"field": "pm25_grade", "op": ">=", "value": 3, "reason": "초미세먼지 나쁨"},
            {"field": "pm10_grade", "op": ">=", "value": 3, "reason": "미세먼지 나쁨"},
        ],
        "ok_response": "미세먼지 양호 수준으로 창문을 열어 환기하기 적절합니다.",
        "fail_template": "{reason} 수준으로 창문 개방은 권장하지 않습니다. 마스크 착용을 권장합니다."
    },
    "cost_trend": {
        "rising": "최근 상승 추세입니다. 참고하시기 바랍니다.",
        "stable": "안정적인 수준입니다.",
        "falling": "하락 추세입니다. 지금 이용하시면 좋겠습니다."
    }
}


def evaluate_judge(judge_type, weather_data=None):
    """
    judge_type: 'outdoor_activity' | 'clothing' | 'air_quality' | 'cost_trend'
    weather_data: dict with fields like precip_prob, pm10_grade, wind_speed, temp_high, temp_low, etc.

    Returns: (is_ok: bool, response: str)
    """
    if weather_data is None:
        # 시뮬레이션 데이터
        weather_data = {
            'precip_prob': 10, 'pm10_grade': 1, 'pm25_grade': 1,
            'wind_speed': 5, 'temp_high': 22, 'temp_low': 14,
            'location': '서울', 'trend': 'stable'
        }

    rule = JUDGE_RULES.get(judge_type)
    if not rule:
        return True, "판단 정보를 확인할 수 없습니다."

    if judge_type == "outdoor_activity":
        for cond in rule["conditions"]:
            val = weather_data.get(cond["field"], 0)
            if cond["op"] == ">" and val > cond["value"]:
                reason = cond["reason"].format(v=val)
                return False, rule["fail_template"].format(reason=reason)
            elif cond["op"] == ">=" and val >= cond["value"]:
                reason = cond["reason"].format(v=val)
                return False, rule["fail_template"].format(reason=reason)
        return True, rule["ok_response"]

    elif judge_type == "clothing":
        high = weather_data.get('temp_high', 22)
        low = weather_data.get('temp_low', 14)
        location = weather_data.get('location', '서울')

        clothing = "적절한 옷을 준비하세요"
        for r in rule["ranges"]:
            if r["min"] <= high <= r["max"]:
                clothing = r["clothing"]
                break

        gap = high - low
        if gap >= rule["gap_threshold"]:
            clothing += f" {rule['gap_note']}"

        response = rule["template"].format(location=location, high=high, low=low, clothing=clothing)
        return True, response

    elif judge_type == "air_quality":
        for cond in rule["conditions"]:
            val = weather_data.get(cond["field"], 0)
            if cond["op"] == ">=" and val >= cond["value"]:
                reason = cond["reason"]
                return False, rule["fail_template"].format(reason=reason)
        return True, rule["ok_response"]

    elif judge_type == "cost_trend":
        trend = weather_data.get('trend', 'stable')
        return True, rule.get(trend, rule["stable"])

    return True, "판단할 수 없습니다."


# 복합 액션 매크로
COMPOSITE_MACROS = {
    "security_mode:away": {
        "name": "외출 모드",
        "actions": [
            {"fn": "light_control", "room": "all", "direction": "off"},
            {"fn": "vent_control", "direction": "off"},
            {"fn": "elevator_call", "direction": "on"},
            {"fn": "security_mode", "param": "away"},
        ],
        "response": "외출 감지 후 일괄 소등, 환기 끄기, 엘리베이터 호출을 실행합니다."
    },
    "security_mode:return": {
        "name": "복귀 모드",
        "actions": [
            {"fn": "light_control", "room": "living", "direction": "on"},
            {"fn": "vent_control", "direction": "on"},
        ],
        "response": "복귀 시 거실 조명, 환기시스템이 켜집니다."
    },
    "schedule_manage:morning": {
        "name": "모닝콜 연동",
        "actions": [
            {"fn": "light_control", "room": "bedroom_main", "direction": "on"},
            {"fn": "vent_control", "direction": "on"},
            {"fn": "curtain_control", "room": "all", "direction": "open"},
        ],
        "response": "모닝콜이 설정되었습니다. 조명, 환기, 전동커튼이 함께 작동합니다."
    }
}


if __name__ == '__main__':
    # 테스트
    print("=== Judge Engine 테스트 ===\n")

    # 좋은 날씨
    ok, resp = evaluate_judge("outdoor_activity", {'precip_prob': 10, 'pm10_grade': 1, 'wind_speed': 3})
    print(f"야외활동 (좋은 날씨): {resp}")

    # 비 오는 날
    ok, resp = evaluate_judge("outdoor_activity", {'precip_prob': 60, 'pm10_grade': 1, 'wind_speed': 3})
    print(f"야외활동 (비): {resp}")

    # 옷차림
    ok, resp = evaluate_judge("clothing", {'temp_high': 25, 'temp_low': 12, 'location': '서울'})
    print(f"옷차림: {resp}")

    # 공기질
    ok, resp = evaluate_judge("air_quality", {'pm10_grade': 4, 'pm25_grade': 3})
    print(f"공기질 (나쁨): {resp}")

    # 비용
    ok, resp = evaluate_judge("cost_trend", {'trend': 'rising'})
    print(f"비용 추세: {resp}")

    print("\n=== 복합 매크로 ===")
    for key, macro in COMPOSITE_MACROS.items():
        print(f"\n{macro['name']}:")
        for action in macro['actions']:
            print(f"  → {action}")
        print(f"  응답: {macro['response']}")
