#!/usr/bin/env python3
"""Dialogue State Tracker — 멀티턴 대화 상태 관리

사용 예:
    dst = DialogueStateTracker(timeout=10)

    # Turn 1: "거실 불 켜줘"
    result = dst.update(nlu_result={'fn':'light_control','exec':'control','dir':'on'},
                        room='living', text="거실 불 켜줘")
    # result = {'fn':'light_control', 'exec':'control', 'dir':'on', 'room':'living'}

    # Turn 2: "안방도" (10초 이내)
    result = dst.update(nlu_result={'fn':'light_control','exec':'clarify','dir':'none'},
                        room='bedroom_main', text="안방도")
    # result = {'fn':'light_control', 'exec':'control', 'dir':'on', 'room':'bedroom_main'}
    # → 이전 턴의 fn+exec+dir 유지, room만 변경
"""
import time
import re


class DialogueStateTracker:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.prev_fn = None
        self.prev_exec = None
        self.prev_dir = None
        self.prev_room = None
        self.prev_time = 0

    def is_active(self):
        """이전 턴이 timeout 내인지"""
        return (time.time() - self.prev_time) < self.timeout

    def update(self, nlu_result, room='none', text=''):
        """NLU 결과 + 대화 상태 결합

        Args:
            nlu_result: {'fn': ..., 'exec_type': ..., 'param_direction': ...}
            room: rule 기반 room 추출 결과
            text: 원본 발화

        Returns:
            resolved: 대화 상태 반영된 최종 결과
        """
        fn = nlu_result.get('fn', 'unknown')
        exec_t = nlu_result.get('exec_type', 'query_then_respond')
        direction = nlu_result.get('param_direction', 'none')

        # 세션 활성 + 불완전 발화 (clarify 또는 room만 있는 경우)
        if self.is_active():
            # "안방도" — device 미지정, room만 있음
            if self._is_followup(text, fn):
                fn = self.prev_fn or fn
                exec_t = self.prev_exec or exec_t
                direction = self.prev_dir or direction

            # "아니 꺼줘" — device는 이전 것, action만 변경
            if self._is_correction(text):
                fn = self.prev_fn or fn

        # 상태 저장
        self.prev_fn = fn
        self.prev_exec = exec_t
        self.prev_dir = direction
        self.prev_room = room if room != 'none' else self.prev_room
        self.prev_time = time.time()

        return {
            'fn': fn,
            'exec_type': exec_t,
            'param_direction': direction,
            'room': room if room != 'none' else (self.prev_room or 'none'),
        }

    def _is_followup(self, text, fn):
        """follow-up 발화인지 (room만 있거나 "~도" 패턴)"""
        if not text:
            return False

        # "안방도", "주방도", "거실도"
        if re.search(r'(거실|안방|주방|침실|작은방|아이방)도$', text.strip()):
            return True

        # 매우 짧고 room만 있는 경우
        rooms = ['거실', '안방', '주방', '침실', '작은방', '아이방']
        text_clean = text.strip()
        if text_clean in rooms:
            return True

        return False

    def _is_correction(self, text):
        """교정 발화인지 ("아니 ~", "아니야 ~")"""
        if not text:
            return False

        return text.strip().startswith('아니') or text.strip().startswith('아냐')


if __name__ == '__main__':
    dst = DialogueStateTracker(timeout=10)

    # 시뮬레이션
    tests = [
        ("거실 불 켜줘", {'fn':'light_control','exec_type':'control_then_confirm','param_direction':'on'}, 'living'),
        ("안방도", {'fn':'light_control','exec_type':'clarify','param_direction':'none'}, 'bedroom_main'),
        ("에어컨도", {'fn':'ac_control','exec_type':'control_then_confirm','param_direction':'on'}, 'none'),
        ("아니 꺼줘", {'fn':'light_control','exec_type':'control_then_confirm','param_direction':'off'}, 'none'),
    ]

    print("=== 멀티턴 DST 시뮬레이션 ===\n")
    for text, nlu, room in tests:
        result = dst.update(nlu, room, text)
        print(f"  사용자: \"{text}\"")
        print(f"  NLU:    fn={nlu['fn']}, exec={nlu['exec_type']}, dir={nlu['param_direction']}")
        print(f"  DST:    fn={result['fn']}, exec={result['exec_type']}, dir={result['param_direction']}, room={result['room']}")
        print()
