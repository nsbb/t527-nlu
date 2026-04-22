#!/usr/bin/env python3
"""Iter 9 Regression Test — iter8/iter9 rule 동작 검증.

각 rule이 의도대로 fire하는지 확인.
Future changes가 이 rule들을 깨면 이 테스트가 실패해야 함.

사용: python3 scripts/regression_test_iter9.py
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from model_cnn_multihead import HEAD_I2L
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess


# 각 entry: (input, expected_field, expected_value, description)
# 하나라도 match 안 되면 실패
TESTS = [
    # Iter 8: 알람/모닝콜 rule
    ("알람 설정해줘", "fn", "schedule_manage", "iter8 alarm rule"),
    ("모닝콜 취소해줘", "fn", "schedule_manage", "iter8 alarm rule (모닝콜)"),
    ("조명 타이머 해줘", "fn", "light_control", "iter8 safeguard (device 있음)"),

    # Iter 8: OOD keywords
    ("네비게이션 켜줘", "fn", "unknown", "iter8 OOD (네비게이션)"),
    ("비행기 예약", "fn", "unknown", "iter8 OOD (비행기)"),

    # Iter 8: unknown → query 복구
    ("서울 날씨 어때", "fn", "weather_query", "iter8 weather 복구"),
    ("국제 뉴스", "fn", "news_query", "iter8 news 복구"),
    ("근처 내과", "fn", "medical_query", "iter8 medical 복구"),

    # Iter 9: 어순 rule
    ("거실 불 좀 켜줘", "exec_type", "control_then_confirm", "iter9 어순 rule"),
    ("침실 조명 좀 꺼줘", "exec_type", "control_then_confirm", "iter9 어순 rule"),

    # Iter 9: curtain rules
    ("블라인드 올려", "param_direction", "up", "iter9 curtain 올려→up"),
    ("커튼 올려", "param_direction", "up", "iter9 curtain 올려→up"),
    ("블라인드 내려줘", "param_direction", "close", "iter9 블라인드 내려→close"),
    ("안방 블라인드", "param_direction", "stop", "iter9 블라인드 no-action→stop"),

    # Iter 9: heat CTC + none → on
    ("바닥 난방", "param_direction", "on", "iter9 heat 기본 on"),
    ("보일러 작동", "param_direction", "on", "iter9 heat 기본 on"),

    # Iter 9: home_info rules
    ("음량 확인", "fn", "home_info", "iter9 음량 → home_info"),
    ("알림 뭐야", "fn", "home_info", "iter9 알림 → home_info"),

    # Iter 9: home_info capability query 예외
    # ("화면 밝기 어떻게 조절할 수 있어", "fn", "system_meta", "iter9 capability 예외"),

    # Iter 9: 전화 entity
    ("전화해줘", "fn", "unknown", "iter9 전화 entity-less OOD"),
    ("관리사무소 전화해줘", "fn", "home_info", "iter9 관리실 전화 in-domain"),

    # Preprocess regression
    ("미세문지 어때", "fn", "weather_query", "preprocess 미세먼지"),
    ("남방 올려쥬", "fn", "heat_control", "preprocess 남방→난방"),
    ("에어컨 이십삼도", "fn", "ac_control", "preprocess 한글숫자"),
    ("스물다섯도로 맞춰줘", "fn", "heat_control", "preprocess 한글숫자 스물다섯"),
    ("거실에 불 좀 켜줘", "exec_type", "control_then_confirm", "iter9 조사 허용"),
    ("불 좀 키줘", "fn", "light_control", "iter9 키줘→켜줘 preprocess"),

    # Reflection (2026-04-22) 추가 개선
    ("방이 덥네", "fn", "ac_control", "reflection: 덥→ac 교정"),
    ("너무 덥다", "fn", "ac_control", "reflection: 덥→ac"),
    ("방 안이 춥다", "fn", "heat_control", "reflection: 춥→heat 확정"),
    ("서울 날씨", "param_direction", "none", "reflection: query dir=none 강제"),

    # Continuous (2026-04-22) 추가
    ("좀 시원하게", "fn", "ac_control", "continuous: 시원→ac"),
    ("블라인드 닫아", "param_direction", "close", "continuous: 블라인드 닫아 open→close"),
    ("공기청정 켜", "fn", "vent_control", "continuous: 공기청정→vent"),
    ("가 스 잠 가", "fn", "gas_control", "continuous: extra-space collapse"),

    # Continuous 2 (10:40+) — more rules
    ("음... 난방", "fn", "heat_control", "continuous: filler 제거"),
    ("통행 시간", "fn", "traffic_query", "continuous: 통행→traffic"),
    ("등산", "fn", "unknown", "continuous: 등산 단독은 unknown"),
    ("등산 괜찮아", "fn", "weather_query", "continuous: 등산 괜찮→weather judge"),
    ("작년보다 추워?", "fn", "weather_query", "continuous: 작년+추워→weather"),
    ("예약 확인해", "fn", "schedule_manage", "continuous: 예약 확인→schedule"),
    ("영어로 뭐야", "fn", "unknown", "continuous: 영어로 OOD"),
    ("커튼 내려줘", "param_direction", "down", "continuous: 커튼 내려→down"),
    ("난방 왜 안 나와", "fn", "heat_control", "continuous: 난방→heat 확정"),
    ("거실이 너무 환해", "fn", "light_control", "continuous: 환해→light"),
    ("에어컨 해줘", "param_direction", "on", "continuous: 해줘→on default"),
]


def main():
    print("Loading ensemble + rules...")
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                 providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    print("Loaded.\n")

    passed = 0
    failed = 0
    fail_details = []

    for text, field, expected, desc in TESTS:
        pp = preprocess(text)
        preds = predict_with_rules(pp, sess, tok)
        actual = preds.get(field, 'N/A')

        if actual == expected:
            passed += 1
            print(f"  ✓ [{desc}]")
            print(f"    \"{text}\" → {field}={actual}")
        else:
            failed += 1
            print(f"  ✗ [{desc}]")
            print(f"    \"{text}\" → {field}={actual}, expected {expected}")
            fail_details.append((desc, text, field, expected, actual))

    print(f"\n결과: {passed} passed, {failed} failed ({len(TESTS)} total)")
    if failed > 0:
        print("\n실패 케이스:")
        for desc, text, field, exp, act in fail_details:
            print(f"  [{desc}] \"{text}\": {field} expected={exp} actual={act}")
        sys.exit(1)
    print("✅ All iter8/iter9 rules working as expected.")


if __name__ == '__main__':
    main()
