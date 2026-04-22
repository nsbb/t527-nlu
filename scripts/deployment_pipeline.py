#!/usr/bin/env python3
"""Deployment Pipeline — 배포용 end-to-end NLU 인터페이스.

Ensemble ONNX + preprocess + post-proc rules + DST 전체 통합.
Android JNI 포팅 시 이 파이프라인 구조를 따르면 됨.

사용:
    pipeline = DeploymentPipeline()
    result = pipeline.process("거실 불 켜줘")
    print(result)  # {fn, exec_type, param_direction, room, value, confidence, response}

    # 멀티턴
    result2 = pipeline.process("안방도")  # DST가 이전 fn 상속

    # 세션 초기화
    pipeline.reset_dst()
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import onnxruntime as ort
import numpy as np
import json, re
from transformers import AutoTokenizer
from model_cnn_multihead import HEAD_I2L
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess
from dialogue_state_tracker import DialogueStateTracker


# Room extraction (simple regex-based)
ROOM_MAP = {
    '거실': 'living', '리빙': 'living',
    '안방': 'bedroom_main', '주침실': 'bedroom_main',
    '침실': 'bedroom_sub', '작은방': 'bedroom_sub', '아이방': 'bedroom_sub',
    '서재': 'bedroom_sub',
    '주방': 'kitchen', '부엌': 'kitchen',
    '현관': 'external', '외부': 'external',
    '욕실': 'none', '화장실': 'none',
    '전체': 'all', '모든': 'all',
}


def extract_room(text):
    for kr, en in ROOM_MAP.items():
        if kr in text:
            return en
    return 'none'


def extract_rooms(text):
    """여러 room 감지 — 'X와/이랑/하고/과 Y' 같은 패턴"""
    rooms = []
    seen = set()
    for kr, en in ROOM_MAP.items():
        if kr in text and en not in seen:
            rooms.append(en)
            seen.add(en)
    return rooms if rooms else ['none']


# Response templates (간소화 — 실제는 sap_inference_v2.py의 RESPONSE_TEMPLATES 사용 권장)
ROOM_KR = {'living': '거실 ', 'kitchen': '주방 ', 'bedroom_main': '안방 ',
           'bedroom_sub': '침실 ', 'all': '전체 ', 'none': '', 'external': ''}

ACTION_MAP = {
    'on': '켰습니다', 'off': '껐습니다', 'open': '열었습니다', 'close': '닫았습니다',
    'up': '올렸습니다', 'down': '내렸습니다', 'set': '설정했습니다', 'stop': '중단했습니다',
}


def generate_simple_response(preds, room, value=None, raw_text=''):
    fn = preds['fn']
    exec_t = preds['exec_type']
    direction = preds['param_direction']

    # Emergency 응답 (우선순위 최상)
    if re.search(r'가스\s*냄새|연기\s*(?:나|난|올)|불\s*(?:났|붙)|침입|도둑', raw_text):
        return '⚠️ 비상 상황 감지. 119/112에 연락하세요. 외출 모드로 전환합니다.'

    if fn == 'unknown':
        return '해당 요청은 서버에서 처리합니다.'

    if exec_t == 'control_then_confirm':
        action = ACTION_MAP.get(direction, '설정했습니다')
        # value 기반 응답 — dir이 명확한 action (on/off/open/close)일 땐 dir 우선
        if value and direction in ('set', 'up', 'down', 'none'):
            vtype, vnum = value
            if vtype == 'temperature':
                action = f'{vnum}도로 설정했습니다'
            elif vtype == 'minute':
                action = f'{vnum}분으로 설정했습니다'
            elif vtype == 'hour':
                action = f'{vnum}시간으로 설정했습니다'
            elif vtype == 'percent':
                action = f'{vnum}%로 설정했습니다'
            elif vtype == 'level':
                action = f'{vnum}단계로 설정했습니다'
        elif value and direction in ('on', 'off'):
            # 타이머 설정: "30분 후 난방 꺼줘" → "30분 뒤에 난방을 끕니다"
            vtype, vnum = value
            if vtype in ('minute', 'hour', 'second'):
                unit_kr = {'minute': '분', 'hour': '시간', 'second': '초'}[vtype]
                action = f'{vnum}{unit_kr} 뒤에 ' + ACTION_MAP.get(direction, '설정합니다')

        # Special cases: elevator uses 호출
        if fn == 'elevator_call' and direction == 'on':
            return '네, 엘리베이터를 호출했습니다.'
        # Schedule: "7시" 같은 시간은 별도 처리
        if fn == 'schedule_manage' and value and value[0] == 'hour':
            return f'네, {value[1]}시 예약을 설정했습니다.'

        room_kr = ROOM_KR.get(room, '')
        # 종성 있는 단어는 '을', 없으면 '를'
        fn_kr = {'light_control': ('조명', '을'), 'heat_control': ('난방', '을'),
                  'ac_control': ('에어컨', '을'), 'vent_control': ('환기 시스템', '을'),
                  'gas_control': ('가스 밸브', '를'), 'door_control': ('도어락', '을'),
                  'curtain_control': ('전동커튼', '을'), 'elevator_call': ('엘리베이터', '를'),
                  'security_mode': ('외출모드', '를'), 'schedule_manage': ('예약', '을')}.get(fn, ('기기', '를'))
        return f'네, {room_kr}{fn_kr[0]}{fn_kr[1]} {action}.'

    if exec_t == 'query_then_respond':
        responses = {
            'weather_query': '오늘 날씨는 맑고 최고 22도입니다.',
            'news_query': '오늘 주요 뉴스를 브리핑합니다.',
            'traffic_query': '현재 교통상황을 확인합니다.',
            'energy_query': '에너지 사용량을 확인합니다.',
            'market_query': '시세 정보를 확인합니다.',
            'medical_query': '근처 병원 정보를 안내합니다.',
        }
        return responses.get(fn, '상태를 확인합니다.')

    if exec_t == 'clarify':
        return '어떤 공간의 기기를 제어할지 말씀해주세요.'

    return '처리합니다.'


class DeploymentPipeline:
    """NLU 배포용 파이프라인.

    preprocess → ensemble → rules → DST → response
    """

    def __init__(self, onnx_path='checkpoints/nlu_v28_v46_ensemble.onnx',
                 tokenizer_path='tokenizer/', timeout=10):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dst = DialogueStateTracker(timeout=timeout)

    def reset_dst(self):
        """DST 세션 초기화"""
        self.dst.reset()

    def process_compound(self, text, use_dst=True):
        """복합 명령 처리 — "A 하고 B" 분할 후 각각 process.

        Returns:
            dict with 'actions': list of process() results, 'is_compound': bool
        """
        # 연결어로 분할 시도
        parts = re.split(r'\s+(?:하고|그리고|그러고|이랑)\s+', text)
        # "동사고 B" 패턴 (동사 + 고 + 공백 + 다음 명령)
        if len(parts) == 1:
            # 동사 어미 + "고 " 찾기 (끄고/꺼고/켜고/닫고/잠그고/열고/올리고/내리고)
            m = re.search(r'^(.+?(?:꺼|끄|켜|닫|잠그|잠가|열|올리|내리))고\s+(.+)$', text)
            if m:
                parts = [m.group(1), m.group(2)]

        if len(parts) < 2:
            # 단일 명령
            r = self.process(text, use_dst=use_dst)
            return {'actions': [r], 'is_compound': False}

        # 복합 명령 — 각 부분 처리
        actions = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            r = self.process(p, use_dst=use_dst)
            actions.append(r)
        return {'actions': actions, 'is_compound': True}

    def process(self, text, use_dst=True):
        """단일 발화 처리 → 구조화된 결과 반환.

        Returns:
            dict with keys:
              - raw: 원본 텍스트
              - preprocessed: preprocess된 텍스트
              - fn, exec_type, param_direction: NLU head outputs
              - room, value: rule-extracted slots
              - dst_applied: DST가 바꿨는지
              - response: 응답 템플릿
        """
        # 1. Preprocess
        pp = preprocess(text)

        # 2. NLU inference (ensemble + rules)
        nlu = predict_with_rules(pp, self.sess, self.tok)

        # 3. Room extract (primary + multi)
        rooms = extract_rooms(pp)
        room = rooms[0]  # primary for DST/response

        # 4. DST
        if use_dst:
            resolved = self.dst.update(nlu, room=room, text=pp)
            dst_applied = (resolved['fn'] != nlu['fn'] or
                           resolved['exec_type'] != nlu['exec_type'] or
                           resolved['param_direction'] != nlu['param_direction'])
            final = resolved
        else:
            dst_applied = False
            final = {**nlu, 'room': room, 'value': None}

        # 5. Response
        response = generate_simple_response(
            {'fn': final['fn'], 'exec_type': final['exec_type'],
             'param_direction': final['param_direction']},
            final.get('room', 'none'),
            final.get('value'),
            raw_text=pp)

        return {
            'raw': text,
            'preprocessed': pp,
            'fn': final['fn'],
            'exec_type': final['exec_type'],
            'param_direction': final['param_direction'],
            'room': final.get('room', 'none'),
            'rooms': rooms,          # 다중 room 리스트 (iter9)
            'value': final.get('value'),
            'dst_applied': dst_applied,
            'response': response,
        }


def main():
    """Sample usage."""
    print("Loading Deployment Pipeline...")
    pipeline = DeploymentPipeline()
    print("Ready.\n")

    # Single queries (reset DST between each to show standalone behavior)
    print("=" * 60)
    print("Single Query Demo (DST reset between each)")
    print("=" * 60)
    for text in [
        "거실 불 켜줘",
        "안방 에어컨 23도로 맞춰줘",
        "알람 설정해줘",
        "서울 날씨 어때",
        "네비게이션 켜줘",
    ]:
        pipeline.reset_dst()
        r = pipeline.process(text)
        print(f'  "{text}" → fn={r["fn"]}, room={r["room"]}')
        print(f'    응답: {r["response"]}')
        print()

    # Multi-turn
    print("=" * 60)
    print("Multi-turn Demo (DST)")
    print("=" * 60)
    pipeline.reset_dst()
    for text in [
        "거실 불 켜줘",
        "안방도",        # room follow-up
        "아니 꺼줘",      # correction
    ]:
        r = pipeline.process(text)
        marker = ' [DST]' if r['dst_applied'] else ''
        print(f'  "{text}" → fn={r["fn"]}, room={r["room"]}, dir={r["param_direction"]}{marker}')
        print(f'    응답: {r["response"]}')
        print()

    # Slot filling
    print("=" * 60)
    print("Slot Filling Demo")
    print("=" * 60)
    pipeline.reset_dst()
    for text in [
        "거실 난방 25도로 맞춰줘",
        "더 올려줘",
        "조금 내려줘",
    ]:
        r = pipeline.process(text)
        val_str = f" value={r['value']}" if r['value'] else ""
        print(f'  "{text}" → fn={r["fn"]}, dir={r["param_direction"]}{val_str}')
        print(f'    응답: {r["response"]}')
        print()


if __name__ == '__main__':
    main()
