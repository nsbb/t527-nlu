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
    rooms = []
    seen = set()
    for kr, en in ROOM_MAP.items():
        if kr in text and en not in seen:
            rooms.append(en)
            seen.add(en)
    return rooms if rooms else ['none']


class DeploymentPipelineV2:
    """NLU v2 배포용 파이프라인 — AI기대응답 생성.

    preprocess → ensemble → rules → DST → response_v2
    """

    def __init__(self, onnx_path='checkpoints/nlu_v28_v46_ensemble.onnx',
                 tokenizer_path='tokenizer/', timeout=10):
        self.sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dst = DialogueStateTracker(timeout=timeout)

    def reset_dst(self):
        self.dst.reset()

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

        # 5. Response v2 (AI기대응답 스타일)
        multihead = {
            'fn': final['fn'],
            'exec_type': final['exec_type'],
            'param_direction': final['param_direction'],
            'param_type': final.get('param_type', 'none'),
            'judge': final.get('judge', 'none'),
            'room': final.get('room', 'none'),
            'value': final.get('value'),
            'old_value': final.get('old_value'),
        }
        response = generate_response_v2(multihead, raw_text=pp)

        return {
            'raw': text,
            'preprocessed': pp,
            'fn': final['fn'],
            'exec_type': final['exec_type'],
            'param_direction': final['param_direction'],
            'param_type': final.get('param_type', 'none'),
            'judge': final.get('judge', 'none'),
            'room': final.get('room', 'none'),
            'rooms': rooms,
            'value': final.get('value'),
            'dst_applied': dst_applied,
            'response': response,
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
