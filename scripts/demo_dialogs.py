#!/usr/bin/env python3
"""실제 사용 시나리오 Demo — End-to-End 대화 예시
preprocess → ensemble inference → DST → response
각 단계가 실제 동작하는 모습 보여줌
"""
import os, sys, json, re, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from preprocess import preprocess
from dialogue_state_tracker import DialogueStateTracker
from sap_inference_v2 import generate_response as full_generate_response, extract_room, extract_value
from transformers import AutoTokenizer

import onnxruntime as ort


def run_dialog(title, turns):
    """하나의 대화 시나리오 실행"""
    print(f"\n{'═'*70}")
    print(f"  시나리오: {title}")
    print('═'*70)

    tok = AutoTokenizer.from_pretrained('tokenizer/')
    sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                providers=['CPUExecutionProvider'])
    dst = DialogueStateTracker(timeout=30)

    for turn_idx, user_text in enumerate(turns, 1):
        print(f"\n┌─ Turn {turn_idx}")
        print(f"│ 사용자: \"{user_text}\"")

        # 1. preprocess
        clean = preprocess(user_text)
        if clean != user_text:
            print(f"│ [전처리] \"{user_text}\" → \"{clean}\"")

        # 2. 추론
        start = time.time()
        tk = tok(clean, padding='max_length', truncation=True, max_length=32, return_tensors='np')
        outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        elapsed = (time.time() - start) * 1000

        nlu = {
            'fn': HEAD_I2L['fn'][outs[0][0].argmax()],
            'exec_type': HEAD_I2L['exec_type'][outs[1][0].argmax()],
            'param_direction': HEAD_I2L['param_direction'][outs[2][0].argmax()],
            'param_type': HEAD_I2L['param_type'][outs[3][0].argmax()],
            'judge': HEAD_I2L['judge'][outs[4][0].argmax()],
        }

        # param_type 규칙 보정
        if nlu['param_direction'] in ('open', 'close', 'stop'): nlu['param_type'] = 'none'
        if nlu['judge'] != 'none': nlu['param_type'] = 'none'
        if nlu['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'): nlu['param_type'] = 'none'

        # 3. Rule-based slots
        room = extract_room(clean)
        value = extract_value(clean)

        # 4. DST (judge/param_type은 DST에서 처리 안 하므로 nlu에서 유지)
        resolved = dst.update(nlu, room, clean)
        resolved['judge'] = nlu['judge']
        resolved['param_type'] = nlu['param_type']

        # 5. Action 표시
        print(f"│ [NLU]   fn={nlu['fn']} exec={nlu['exec_type']} dir={nlu['param_direction']}" +
              (f" param={nlu['param_type']}" if nlu['param_type'] != 'none' else "") +
              (f" judge={nlu['judge']}" if nlu['judge'] != 'none' else ""))
        if room != 'none' or value:
            print(f"│ [Rule]  room={room}" + (f" value={value[1]}{value[0]}" if value else ""))

        # DST 효과 표시
        dst_changed = (resolved['fn'] != nlu['fn'] or
                       resolved['exec_type'] != nlu['exec_type'] or
                       resolved['param_direction'] != nlu['param_direction'] or
                       resolved['room'] != room)
        if dst_changed:
            print(f"│ [DST]   fn={resolved['fn']} exec={resolved['exec_type']} "
                  f"dir={resolved['param_direction']} room={resolved['room']}")

        # 6. 응답 생성 (sap_inference_v2의 generate_response 사용)
        # resolved를 full preds 형태로 확장
        full_preds = dict(resolved)
        full_preds['judge'] = nlu['judge']
        full_preds['param_type'] = nlu['param_type']
        response = full_generate_response(full_preds, clean)
        print(f"│ 어시스턴트: \"{response}\"")
        print(f"└─ ({elapsed:.1f}ms)")


def generate_simple_response(r, value):
    fn = r['fn']; exec_t = r['exec_type']; direction = r['param_direction']; room = r['room']

    room_kr = {'living':'거실 ','kitchen':'주방 ','bedroom_main':'안방 ','bedroom_sub':'침실 ',
               'all':'전체 ','none':''}.get(room, '')

    action_map = {'on': '켰습니다', 'off': '껐습니다', 'open': '열었습니다', 'close': '닫았습니다',
                  'up': '올렸습니다', 'down': '내렸습니다', 'set': '설정했습니다', 'stop': '중단했습니다'}

    if fn == 'unknown':
        return '해당 요청은 서버에서 처리하겠습니다.'

    if exec_t == 'control_then_confirm':
        action = action_map.get(direction, '처리했습니다')
        if value and value[0] == 'temperature':
            action = f'{value[1]}도로 설정합니다'
        elif value and value[0] == 'time':
            action = f'{value[1]}분 후 실행하도록 설정합니다'

        device_map = {
            'light_control': f'{room_kr}조명을',
            'heat_control': f'{room_kr}난방을',
            'ac_control': f'{room_kr}에어컨을',
            'vent_control': '환기시스템을',
            'gas_control': '가스 밸브를',
            'door_control': '현관문을',
            'curtain_control': f'{room_kr}커튼을',
            'elevator_call': '엘리베이터를 호출합니다',
            'security_mode': '외출 모드로 전환합니다',
            'schedule_manage': f'예약을',
        }
        device = device_map.get(fn, '기기를')
        if fn in ('elevator_call', 'security_mode'):
            return f'네, {device}.'
        return f'네, {device} {action}.'

    if exec_t == 'query_then_respond':
        templates = {
            'heat_control': '현재 실내 온도는 23도이며 난방 설정은 25도입니다.',
            'ac_control': '현재 에어컨은 자동 모드로 작동 중입니다.',
            'weather_query': '오늘 서울 날씨는 맑고 최고 22도입니다.',
            'news_query': '오늘 주요 뉴스를 안내해드릴까요?',
            'energy_query': '이번 달 전기 사용량은 240kWh입니다.',
            'gas_control': '현재 가스 밸브는 잠겨있습니다.',
            'door_control': '현재 현관문은 잠겨있습니다.',
        }
        return templates.get(fn, '정보를 확인합니다.')

    if exec_t == 'query_then_judge':
        templates = {
            'outdoor_activity': '기온과 대기질 모두 양호합니다. 외출하기 무리 없어요.',
            'clothing': '오늘은 따뜻해서 얇은 긴팔이 적당합니다.',
            'air_quality': '미세먼지 양호 — 환기하기 좋은 날씨입니다.',
        }
        return templates.get(r['judge'], '판단해드릴게요.')

    if exec_t == 'direct_respond':
        return '정보를 확인합니다.'

    if exec_t == 'clarify':
        return '어떤 공간이나 기기를 말씀하시는 건가요?'

    return '처리하겠습니다.'


# ============================================================
# 시나리오
# ============================================================
SCENARIOS = [
    ("기본 기기 제어", [
        "거실 에어컨 켜줘",
        "23도로 맞춰줘",
        "안방 난방도 켜줘",
        "전체 끄기",
    ]),
    ("멀티턴 DST - 방 상속", [
        "거실 불 켜줘",
        "안방도",
        "그리고 주방도",
    ]),
    ("멀티턴 DST - 기기 변경", [
        "거실 에어컨 켜줘",
        "난방도",
        "환기도",
    ]),
    ("멀티턴 DST - 교정", [
        "안방 에어컨 켜줘",
        "아니 꺼줘",
    ]),
    ("STT 오류 복원", [
        "에어콘 켜쥬",
        "뉴슈 알려줘",
        "도어렉 열어",
        "오늘날씨어때",
    ]),
    ("값 지정 - 온도/시간", [
        "에어컨 이십삼도로",
        "30분 후 꺼줘",
        "난방 25도로 맞춰줘",
    ]),
    ("정보 질의", [
        "오늘 날씨 어때",
        "실내 온도 몇 도",
        "이번 달 전기세 얼마",
        "교통 상황 어때",
    ]),
    ("판단형 질의", [
        "지금 외출해도 돼",
        "뭐 입고 나갈까",
        "창문 열어도 괜찮아",
    ]),
    ("시스템 메타", [
        "너 이름 뭐야",
        "어떻게 불러",
        "무슨 기능 있어",
    ]),
    ("모호한 짧은 발화", [
        "조명",
        "덥다",
        "추워",
    ]),
]


if __name__ == '__main__':
    print("🤖 NLU Demo — End-to-End 대화 시나리오")
    print(f"   모델: v28+v46 Ensemble ONNX")
    print(f"   파이프라인: preprocess → inference → DST → response")

    for title, turns in SCENARIOS:
        run_dialog(title, turns)

    print(f"\n{'═'*70}")
    print(f"  총 {len(SCENARIOS)}개 시나리오, {sum(len(t) for _, t in SCENARIOS)}개 턴 시연 완료")
    print('═'*70)
