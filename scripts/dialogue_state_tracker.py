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
        self.prev_value = None
        self.prev_text = None
        self.prev_time = 0
        self.turn_count = 0

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

        resolved_room = room

        # 세션 활성 + 불완전 발화 (clarify 또는 room만 있는 경우)
        if self.is_active():
            followup_type = self._get_followup_type(text, fn)

            if followup_type == 'room':
                # "안방도", "거실" — room만 변경, fn/exec/dir 이전 것 유지
                fn = self.prev_fn or fn
                exec_t = self.prev_exec or exec_t
                direction = self.prev_dir or direction

            elif followup_type == 'device':
                # "난방도", "에어컨도" — fn은 NLU 예측 유지, exec/dir만 이전 것 상속
                exec_t = self.prev_exec or exec_t
                if direction == 'none':
                    direction = self.prev_dir or direction

            elif followup_type == 'confirm':
                # "응", "해줘" — 이전 턴 전체 반복
                fn = self.prev_fn or fn
                exec_t = self.prev_exec or exec_t
                direction = self.prev_dir or direction

            elif self._is_correction(text):
                # "아니 꺼줘" — device는 이전 것, action만 변경
                fn = self.prev_fn or fn

            elif self._is_there_too(text):
                # "거기도" — 이전 room 재사용 + 이전 fn/exec/dir 유지
                fn = self.prev_fn or fn
                exec_t = self.prev_exec or exec_t
                direction = self.prev_dir or direction
                resolved_room = self.prev_room or room

            elif exec_t == 'clarify' and self.prev_fn:
                # exec_type이 clarify이고 이전 상태가 있으면 보완
                fn = self.prev_fn
                exec_t = self.prev_exec or exec_t
                if direction == 'none':
                    direction = self.prev_dir or direction

        # Room resolution
        if resolved_room == 'none' and self.prev_room:
            resolved_room = self.prev_room

        # 상태 저장
        self.prev_fn = fn
        self.prev_exec = exec_t
        self.prev_dir = direction
        self.prev_room = resolved_room if resolved_room != 'none' else self.prev_room
        self.prev_text = text
        self.prev_time = time.time()
        self.turn_count += 1

        return {
            'fn': fn,
            'exec_type': exec_t,
            'param_direction': direction,
            'room': resolved_room if resolved_room != 'none' else (self.prev_room or 'none'),
        }

    def _get_followup_type(self, text, fn):
        """follow-up 발화 유형 판별

        Returns:
            'room': 방 follow-up ("안방도", "거실")
            'device': 기기 follow-up ("난방도", "에어컨도")
            'confirm': 확인/동의 ("응", "해줘")
            None: follow-up 아님
        """
        if not text:
            return None

        text_clean = text.strip()

        # "안방도", "주방도", "거실도" — room follow-up
        if re.search(r'(거실|안방|주방|침실|작은방|아이방|부엌|큰방)도$', text_clean):
            return 'room'

        # "에어컨도", "난방도" — device follow-up (fn은 유지, exec/dir만 상속)
        if re.search(r'(에어컨|난방|조명|불|환기|커튼|가스|도어락|보일러)도$', text_clean):
            return 'device'

        # 매우 짧고 room만 있는 경우
        rooms = ['거실', '안방', '주방', '침실', '작은방', '아이방', '부엌', '큰방']
        if text_clean in rooms:
            return 'room'

        # 짧은 확인/동의
        if text_clean in ('응', '그래', '맞아', '네', '예', '좋아', '해줘', '그거'):
            return 'confirm'

        return None

    def _is_correction(self, text):
        """교정 발화인지 ("아니 ~", "아니야 ~")"""
        if not text:
            return False

        text_clean = text.strip()
        return text_clean.startswith('아니') or text_clean.startswith('아냐')

    def _is_there_too(self, text):
        """'거기도', '여기도' 패턴"""
        if not text:
            return False
        text_clean = text.strip()
        return text_clean in ('거기도', '여기도', '같은거', '똑같이')


if __name__ == '__main__':
    dst = DialogueStateTracker(timeout=10)

    scenarios = [
        {
            'name': '시나리오 1: 방 추가 (follow-up)',
            'turns': [
                ("거실 불 켜줘", {'fn':'light_control','exec_type':'control_then_confirm','param_direction':'on'}, 'living'),
                ("안방도", {'fn':'light_control','exec_type':'clarify','param_direction':'none'}, 'bedroom_main'),
            ]
        },
        {
            'name': '시나리오 2: 기기 추가',
            'turns': [
                ("거실 에어컨 켜줘", {'fn':'ac_control','exec_type':'control_then_confirm','param_direction':'on'}, 'living'),
                ("난방도", {'fn':'heat_control','exec_type':'control_then_confirm','param_direction':'on'}, 'none'),
            ]
        },
        {
            'name': '시나리오 3: 교정',
            'turns': [
                ("거실 불 켜줘", {'fn':'light_control','exec_type':'control_then_confirm','param_direction':'on'}, 'living'),
                ("아니 꺼줘", {'fn':'light_control','exec_type':'control_then_confirm','param_direction':'off'}, 'none'),
            ]
        },
        {
            'name': '시나리오 4: clarify 후 확인',
            'turns': [
                ("불 켜줘", {'fn':'light_control','exec_type':'clarify','param_direction':'on'}, 'none'),
                ("거실", {'fn':'light_control','exec_type':'clarify','param_direction':'none'}, 'living'),
            ]
        },
        {
            'name': '시나리오 5: 동의 패턴',
            'turns': [
                ("안방 에어컨 켜줄까요?", {'fn':'ac_control','exec_type':'control_then_confirm','param_direction':'on'}, 'bedroom_main'),
                ("응", {'fn':'unknown','exec_type':'direct_respond','param_direction':'none'}, 'none'),
            ]
        },
    ]

    print("=== 멀티턴 DST 시뮬레이션 ===\n")
    for scenario in scenarios:
        print(f"--- {scenario['name']} ---")
        dst.reset()
        for text, nlu, room in scenario['turns']:
            result = dst.update(nlu, room, text)
            print(f"  사용자: \"{text}\"")
            print(f"  NLU:    fn={nlu['fn']}, exec={nlu['exec_type']}, dir={nlu['param_direction']}")
            print(f"  DST:    fn={result['fn']}, exec={result['exec_type']}, dir={result['param_direction']}, room={result['room']}")
        print()
