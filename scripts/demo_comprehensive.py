#!/usr/bin/env python3
"""종합 데모 — NLU 배포 파이프라인 전체 기능 시연.

사용:
    python3 scripts/demo_comprehensive.py

시연 내용:
  1. 기본 제어 (조명/에어컨/난방)
  2. STT 오류 내성 (preprocess)
  3. 후처리 rule (알람/OOD/unknown 복구)
  4. DST 멀티턴 (slot filling, 재교정)
  5. Value 추출 (온도/시간/볼륨)
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort
import numpy as np
import json, re
from transformers import AutoTokenizer
from model_cnn_multihead import HEAD_I2L
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess
from dialogue_state_tracker import DialogueStateTracker


def run_single(sess, tok, text):
    """단일 발화 처리 (preprocess + ensemble + rules)"""
    pp = preprocess(text)
    preds = predict_with_rules(pp, sess, tok)
    return pp, preds


def print_result(text, pp, preds, dst_result=None):
    pp_marker = f' → "{pp}"' if pp != text else ''
    print(f'  입력: "{text}"{pp_marker}')
    print(f'  NLU:  fn={preds["fn"]:<18} exec={preds["exec_type"]:<22} dir={preds["param_direction"]}')
    if dst_result:
        value = dst_result.get('value')
        val_str = f"  value={value}" if value else ""
        changed = ' [DST resolved]' if (dst_result.get('fn') != preds['fn'] or
                                        dst_result.get('exec_type') != preds['exec_type'] or
                                        dst_result.get('param_direction') != preds['param_direction']) else ''
        print(f'  DST:  fn={dst_result.get("fn", preds["fn"]):<18} exec={dst_result.get("exec_type", preds["exec_type"]):<22} dir={dst_result.get("param_direction", preds["param_direction"])}{changed}')
        print(f'        room={dst_result.get("room", "none")}{val_str}')


def main():
    print("종합 NLU 데모 로딩 중...")
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                 providers=['CPUExecutionProvider'])
    tok = AutoTokenizer.from_pretrained('tokenizer/')
    print("로딩 완료.\n")

    # ============================================================
    # 1. 기본 제어
    # ============================================================
    print("=" * 60)
    print("1. 기본 기기 제어")
    print("=" * 60)
    for text in [
        "거실 불 켜줘",
        "안방 에어컨 23도로 맞춰줘",
        "현관 도어락 열어줘",
        "가스 밸브 잠가줘",
        "엘리베이터 불러줘",
    ]:
        pp, preds = run_single(sess, tok, text)
        print_result(text, pp, preds)
        print()

    # ============================================================
    # 2. STT 오류 내성
    # ============================================================
    print("=" * 60)
    print("2. STT 오류 내성 (preprocess 사전)")
    print("=" * 60)
    for text in [
        "미세문지 어때",
        "남방 올려쥬",
        "엘레베이터 불러줘",
        "까스 잠가쥬",
        "침실조명 꺼쥬",
        "오늘날씨어때",
        "지금몇시야",
    ]:
        pp, preds = run_single(sess, tok, text)
        print_result(text, pp, preds)
        print()

    # ============================================================
    # 3. 후처리 rule (iter8/9)
    # ============================================================
    print("=" * 60)
    print("3. 후처리 rule (iter8/9 개선)")
    print("=" * 60)
    for text in [
        "알람 설정해줘",          # → schedule_manage
        "서울 날씨 어때",          # unknown → weather_query 복구
        "병원 추천해줘",          # unknown → medical_query 복구
        "네비게이션 켜줘",        # in-domain keyword → unknown
        "전화해줘",                # OOD (entity 없음)
        "관리사무소 전화번호",    # in-domain (entity 있음)
        "거실 불 좀 켜줘",         # 어순 rule → CTC
        "블라인드 올려줘",         # curtain open → up 수정
        "바닥 난방",               # heat CTC + none → on
    ]:
        pp, preds = run_single(sess, tok, text)
        print_result(text, pp, preds)
        print()

    # ============================================================
    # 4. DST 멀티턴 대화
    # ============================================================
    print("=" * 60)
    print("4. DST 멀티턴 (context 추적)")
    print("=" * 60)

    def room_extract(text):
        """간단 room extract"""
        rooms = {'거실': 'living', '안방': 'bedroom_main', '침실': 'bedroom_sub',
                 '주방': 'kitchen', '작은방': 'bedroom_sub', '아이방': 'bedroom_sub'}
        for kr, en in rooms.items():
            if kr in text:
                return en
        return 'none'

    scenarios = [
        ("room follow-up", [
            "거실 불 켜줘",
            "안방도",
        ]),
        ("device follow-up", [
            "거실 에어컨 켜줘",
            "난방도",
        ]),
        ("correction", [
            "거실 불 켜줘",
            "아니 꺼줘",
        ]),
        ("slot filling (iter9)", [
            "거실 난방 25도로 맞춰줘",
            "더 올려줘",
            "조금 내려줘",
        ]),
        ("confirm", [
            "안방 에어컨 켜줄까요?",
            "응",
        ]),
    ]

    for name, turns in scenarios:
        print(f"--- {name} ---")
        dst = DialogueStateTracker(timeout=10)
        for text in turns:
            pp, preds = run_single(sess, tok, text)
            room = room_extract(pp)
            resolved = dst.update(preds, room=room, text=pp)
            print_result(text, pp, preds, resolved)
        print()

    # ============================================================
    # 5. Latency
    # ============================================================
    print("=" * 60)
    print("5. 성능 측정")
    print("=" * 60)
    import time, random
    test = json.load(open('data/test_suite.json'))
    random.seed(42)
    sample = random.sample(test, 100)
    t0 = time.perf_counter()
    for item in sample:
        run_single(sess, tok, item['utterance'])
    dt = (time.perf_counter() - t0) / 100 * 1000
    print(f"  평균 추론 latency: {dt:.2f}ms/query (100 샘플)")

    # TS 정확도
    print(f"\n  Test Suite 성능:")
    c = n = 0
    for item in test:
        pp, preds = run_single(sess, tok, item['utterance'])
        if (preds['fn'] == item['fn'] and
            preds['exec_type'] == item['exec'] and
            preds['param_direction'] == item['dir']):
            c += 1
        n += 1
    print(f"  combo accuracy: {c/n*100:.2f}% ({c}/{n})")


if __name__ == '__main__':
    main()
