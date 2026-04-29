#!/usr/bin/env python3
"""Deployment Pipeline v2 — end-to-end NLU + AI기대응답 생성.

v1과 차이:
- v1: generate_simple_response (간소 템플릿)
- v2: generate_response_v2 (르엘 AI기대응답 스타일)

우위점 (vs single intent):
- fn이 unknown이어도 room/device keyword hint로 조립 응답
- 일부 헤드만 맞아도 합리적 응답 (single intent는 이상한 응답 or 무조건 unknown)

사용:
    pipeline = DeploymentPipelineV2()
    result = pipeline.process("거실 불 켜줘")
    # result['response'] = "네, 거실 조명을 켭니다."
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import onnxruntime as ort
import numpy as np
import re
from transformers import AutoTokenizer
from model_cnn_multihead import HEAD_I2L
from ensemble_inference_with_rules import predict_with_rules
from preprocess import preprocess
from dialogue_state_tracker import DialogueStateTracker
from response_generator_v2 import generate_response_v2


# Room extraction (v1과 동일)
ROOM_MAP = {
    '거실': 'living', '리빙': 'living',
    '안방': 'bedroom_main', '주침실': 'bedroom_main', '우리 방': 'bedroom_main',
    '침실': 'bedroom_sub', '작은방': 'bedroom_sub', '아이방': 'bedroom_sub',
    '작은 방': 'bedroom_sub', '아이 방': 'bedroom_sub',
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
    rooms = []
    seen = set()
    for kr, en in ROOM_MAP.items():
        if kr in text and en not in seen:
            rooms.append(en)
            seen.add(en)
    return rooms if rooms else ['none']


CONTROL_FNS = {'light_control', 'ac_control', 'heat_control', 'vent_control',
               'gas_control', 'door_control', 'curtain_control', 'elevator_call',
               'security_mode'}


class HomeState:
    """집 상태(HomeState) — 제어 명령 결과를 누적해 충돌 해결/조회에 활용.

    내부 표현: self._s[(fn, room)] = dict {
        'power': 'on'|'off'|'open'|'closed'|'stop'|None,
        'value': (vtype, vnum) or None,    # temperature/percent/level/enum/minute
        'mode':  '외출'|'재택'|'취침' 등 or None,  # security_mode 전용
    }
    """

    def __init__(self):
        self._s = {}

    def _ensure(self, fn, room):
        key = (fn, room)
        if key not in self._s:
            self._s[key] = {'power': None, 'value': None, 'mode': None}
        return self._s[key]

    def update(self, fn, room, direction, value=None, text=None):
        st = self._ensure(fn, room)
        # 보안 모드는 별도: 모드 명칭을 추출
        if fn == 'security_mode':
            if direction == 'on':
                st['power'] = 'on'
                if text:
                    if '외출' in text or '나간' in text or '나갈' in text or '집 비울' in text:
                        st['mode'] = '외출'
                    elif '재택' in text:
                        st['mode'] = '재택'
                    elif '취침' in text or '잠' in text:
                        st['mode'] = '취침'
                    elif '방범' in text:
                        st['mode'] = '방범'
            elif direction == 'off':
                st['power'] = 'off'; st['mode'] = None
            return

        # 전원/개폐/멈춤
        if direction == 'on':
            st['power'] = 'on'
            if value:  # "에어컨 강풍으로 켜줘" 같은 enum/temperature 동시 발화
                st['value'] = value
        elif direction == 'off':
            st['power'] = 'off'
            st['value'] = None
        elif direction == 'open':
            st['power'] = 'open'
        elif direction == 'close':
            st['power'] = 'closed'
        elif direction == 'stop':
            st['power'] = 'stop'
        elif direction == 'set':
            if value:
                st['value'] = value
                if st['power'] in (None, 'off'):
                    st['power'] = 'on'  # set은 켜짐을 함의
        elif direction == 'up':
            # 강도 표현 (밝게/시원하게/세게) — power 켜짐 유지, value enum 가능
            if st['power'] in (None, 'off'):
                st['power'] = 'on'
            if value and value[0] == 'enum':
                st['value'] = value
        elif direction == 'down':
            if st['power'] in (None, 'off'):
                st['power'] = 'on'
            if value and value[0] == 'enum':
                st['value'] = value

    def is_on(self, fn, room):
        st = self._s.get((fn, room))
        if not st:
            return False
        p = st.get('power')
        return p in ('on', 'open')

    def summary_kr(self):
        """현재 상태를 한국어 요약 문자열로 반환."""
        DEVICE_KR = {
            'light_control': '조명', 'ac_control': '에어컨', 'heat_control': '난방',
            'vent_control': '환기', 'gas_control': '가스밸브', 'door_control': '도어락',
            'curtain_control': '커튼', 'elevator_call': '엘리베이터',
            'security_mode': '보안',
        }
        ROOM_KR = {
            'living': '거실', 'bedroom_main': '안방', 'bedroom_sub': '침실',
            'kitchen': '주방', 'external': '현관', 'all': '전체', 'none': '',
        }
        POWER_KR = {'on': '켜짐', 'off': '꺼짐', 'open': '열림',
                    'closed': '닫힘', 'stop': '정지'}
        ENUM_KR = {'strong': '강', 'medium': '중', 'weak': '약',
                   'max': '최대', 'min': '최소'}
        if not self._s:
            return None
        items = []
        for (fn, room), st in self._s.items():
            dev = DEVICE_KR.get(fn, fn)
            rm = ROOM_KR.get(room, '')
            prefix = f'{rm} ' if rm else ''
            # 보안 모드는 모드명 우선
            if fn == 'security_mode':
                if st.get('power') == 'on' and st.get('mode'):
                    items.append(f'보안 {st["mode"]} 모드')
                elif st.get('power') == 'on':
                    items.append('보안 켜짐')
                elif st.get('power') == 'off':
                    items.append('보안 해제')
                continue
            # 일반 기기: value 있으면 value+켜짐, 없으면 power만
            val = st.get('value')
            power = st.get('power')
            if val and power in ('on', 'open', None):
                vtype, vnum = val
                if vtype == 'temperature':
                    items.append(f'{prefix}{dev} {vnum}도 켜짐')
                elif vtype == 'percent':
                    items.append(f'{prefix}{dev} {vnum}% 켜짐')
                elif vtype == 'level':
                    items.append(f'{prefix}{dev} {vnum}단계 켜짐')
                elif vtype == 'enum':
                    items.append(f'{prefix}{dev} {ENUM_KR.get(vnum, vnum)} 켜짐')
                elif vtype == 'minute':
                    items.append(f'{prefix}{dev} {vnum}분 예약')
                else:
                    items.append(f'{prefix}{dev} {vnum} 켜짐')
            elif power:
                items.append(f'{prefix}{dev} {POWER_KR.get(power, power)}')
        return ', '.join(items) if items else None

    def reset(self):
        self._s.clear()


class DeploymentPipelineV2:
    """NLU v2 배포용 파이프라인 — AI기대응답 생성.

    preprocess → ensemble → rules → DST → response_v2
    """

    def __init__(self, onnx_path='checkpoints/nlu_v28_v72_ensemble.onnx',
                 tokenizer_path='tokenizer/', timeout=10):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dst = DialogueStateTracker(timeout=timeout)
        self.home_state = HomeState()

    def reset_dst(self):
        self.dst.reset()
        self.home_state.reset()

    def _split_compound(self, text):
        parts = re.split(r'\s+(?:하고|그리고|그러고|이랑)\s+', text)
        if len(parts) > 1:
            result = []
            for p in parts:
                result.extend(self._split_compound(p.strip()))
            return result
        m = re.search(r'^(.+?(?:꺼|끄|켜|닫|잠그|잠가|열|올리|내리))고\s+(.+)$', text)
        if m:
            return [m.group(1).strip()] + self._split_compound(m.group(2).strip())
        return [text.strip()] if text.strip() else []

    def process_compound(self, text, use_dst=True):
        parts = self._split_compound(text)
        if len(parts) < 2:
            r = self.process(text, use_dst=use_dst)
            return {'actions': [r], 'is_compound': False}
        actions = []
        for p in parts:
            if not p:
                continue
            r = self.process(p, use_dst=use_dst)
            actions.append(r)
        return {'actions': actions, 'is_compound': True}

    def process(self, text, use_dst=True):
        """단일 발화 처리 → 구조화된 결과 + AI기대응답 문장."""
        # 1. Preprocess
        pp = preprocess(text)

        # 2. NLU inference (ensemble + rules)
        nlu = predict_with_rules(pp, self.sess, self.tok)

        # 3. Room extract
        rooms = extract_rooms(pp)
        room = rooms[0]

        # 4. DST (fn/value slot filling)
        if use_dst:
            resolved = self.dst.update(nlu, room=room, text=pp)
            dst_applied = (resolved['fn'] != nlu['fn'] or
                           resolved['exec_type'] != nlu['exec_type'] or
                           resolved['param_direction'] != nlu['param_direction'])
            final = resolved
        else:
            dst_applied = False
            final = {**nlu, 'room': room, 'value': None}

        fn_final = final['fn']
        dir_final = final['param_direction']
        room_final = final.get('room', 'none')
        val_final = final.get('value')

        # 4.5. HVAC 충돌 재해석 (집상태 기반, 명시적 기기 키워드 없을 때만)
        # 반대 HVAC 켜진 상태 + 암시적 신체감각 발화 → 켜진 기기를 끔
        # (예: AC on + "추워" → heat/on 보다 ac/off가 자연스러움)
        # 명시적 키워드("난방 켜줘"/"에어컨 켜줘") 있으면 사용자 의도 존중 → 자동 상호배제만
        _explicit_heat = re.search(r'난방|보일러|히터|라디에이터|온돌', pp)
        _explicit_ac = re.search(r'에어컨|냉방|에어콘', pp)
        if fn_final == 'heat_control' and dir_final == 'on' and not _explicit_heat:
            if self.home_state.is_on('ac_control', room_final):
                fn_final = 'ac_control'
                dir_final = 'off'
        elif fn_final == 'ac_control' and dir_final == 'on' and not _explicit_ac:
            if self.home_state.is_on('heat_control', room_final):
                fn_final = 'heat_control'
                dir_final = 'off'

        # 5. 집 상태 업데이트 (제어 명령만)
        if fn_final in CONTROL_FNS:
            self.home_state.update(fn_final, room_final, dir_final, val_final, text=pp)
            # 케이스 B: HVAC ON 시 반대 HVAC 자동 OFF (상호배제)
            if dir_final == 'on' and fn_final in ('ac_control', 'heat_control'):
                opposite = 'heat_control' if fn_final == 'ac_control' else 'ac_control'
                if self.home_state.is_on(opposite, room_final):
                    self.home_state.update(opposite, room_final, 'off')

        # 6. Response v2 (AI기대응답 스타일)
        multihead = {
            'fn': fn_final,
            'exec_type': final['exec_type'],
            'param_direction': dir_final,
            'room': room_final,
            'value': val_final,
            'old_value': final.get('old_value'),
            'home_state': self.home_state.summary_kr(),  # 집 상태 조회용
        }
        response = generate_response_v2(multihead, raw_text=pp)

        return {
            'raw': text,
            'preprocessed': pp,
            'fn': fn_final,
            'exec_type': final['exec_type'],
            'param_direction': dir_final,
            'room': room_final,
            'rooms': rooms,
            'value': val_final,
            'dst_applied': dst_applied,
            'response': response,
            'home_state': self.home_state.summary_kr(),
        }


def main():
    print("Loading Deployment Pipeline v2...")
    p = DeploymentPipelineV2()
    print("Ready.\n")

    # 219 시나리오 중 샘플
    tests = [
        '지금 집 상태 어때?',
        '지금 몇 시야?',
        '거실 불 켜줘',
        '안방 에어컨 23도로 맞춰줘',
        '전체 난방 켜 줘',
        '가스 밸브 잠금해',
        '거실 전동커튼 열어줘',
        '오늘 날씨 어때?',
        '강남역까지 얼마나 걸려?',
        '긴급 상황이야',
        '알람 7시',
        '외출모드 실행해 줘',
        '뉴스 틀어줘',
        # Unknown 조립 테스트
        '넷플릭스 틀어줘',
        '거실 뭐 좀 켜',  # unknown + room + verb
    ]
    for t in tests:
        p.reset_dst()
        r = p.process(t, use_dst=False)
        print(f'  "{t}"')
        print(f'    fn={r["fn"]}, exec={r["exec_type"]}, dir={r["param_direction"]}, room={r["room"]}, v={r["value"]}')
        print(f'    → {r["response"]}')
        print()


if __name__ == '__main__':
    main()
