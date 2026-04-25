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
        self.prev_value = None           # (type, number) 예: ('temperature', 25)
        self.prev_text = None
        self.prev_time = 0
        self.turn_count = 0
        self.history = []                # recent turns: last 5 (iter9 extension)

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
                # "난방도", "에어컨도" — fn은 NLU 예측 유지, exec/dir 이전 턴 우선
                exec_t = self.prev_exec or exec_t
                # prev_dir이 명확한 action이면 (on/off/open/close) 우선
                if self.prev_dir in ('on', 'off', 'open', 'close'):
                    direction = self.prev_dir
                elif direction == 'none':
                    direction = self.prev_dir or direction

            elif followup_type == 'confirm':
                # "응", "해줘" — 이전 턴 전체 반복, 단 확인 질문(query)이었으면 control로 승격
                fn = self.prev_fn or fn
                # "~할까요?" 같은 확인 질문 → confirm으로 바뀌면 실행
                if self.prev_exec == 'query_then_respond' and self.prev_text and re.search(r'까요\?|할까|낼까|될까', self.prev_text or ''):
                    exec_t = 'control_then_confirm'
                    # dir이 none이면 prev_text에서 action verb로 추론
                    if direction == 'none' and (self.prev_dir is None or self.prev_dir == 'none'):
                        if re.search(r'켤|켜|틀', self.prev_text):
                            direction = 'on'
                        elif re.search(r'끌|꺼|끄', self.prev_text):
                            direction = 'off'
                        elif re.search(r'열', self.prev_text):
                            direction = 'open'
                        elif re.search(r'닫|잠', self.prev_text):
                            direction = 'close'
                    else:
                        direction = self.prev_dir or direction
                else:
                    exec_t = self.prev_exec or exec_t
                    direction = self.prev_dir or direction

            elif self._is_correction(text):
                # "아니 꺼줘" — device 이전 것, action만 변경
                # 단 새 device keyword 명시되면 그쪽 우선 ("아니 에어컨 꺼")
                new_device = re.search(r'에어컨|난방|조명|불|환기|가스|도어락|커튼|블라인드', text)
                if not new_device:
                    fn = self.prev_fn or fn
                # direction 추론 — 발화 내 verb 우선
                if direction == 'none':
                    if re.search(r'켜', text):
                        direction = 'on'
                        exec_t = 'control_then_confirm'
                    elif re.search(r'꺼|끄', text):
                        direction = 'off'
                        exec_t = 'control_then_confirm'
                    elif re.search(r'열어', text):
                        direction = 'open'
                        exec_t = 'control_then_confirm'
                    elif re.search(r'닫아|잠가|잠궈', text):
                        direction = 'close'
                        exec_t = 'control_then_confirm'
                    elif new_device and self.prev_dir:
                        # 새 device + verb 없음 → 이전 direction 상속
                        # ("난방 켜" → "아니 에어컨" = 에어컨 켜)
                        direction = self.prev_dir
                        exec_t = 'control_then_confirm'

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

        # Value 추출 (iter9: slot filling)
        current_value = self._extract_value(text)
        inferred_value = None
        old_value = None  # 조정 전 값 (응답 "N도에서 M도로" 용)

        # 짧은 bare 발화 (더/덜/조금/많이) → fn 상속 + direction 추론
        if self.is_active() and self.prev_fn and len(text.strip()) <= 4:
            bare_direction_map = {
                '더': 'up', '더 더': 'up', '많이': 'up',
                '덜': 'down', '조금': 'down', '조금만': 'down', '살짝': 'down',
            }
            if text.strip() in bare_direction_map:
                fn = self.prev_fn
                exec_t = self.prev_exec or exec_t
                direction = bare_direction_map[text.strip()]
                # slot fill value
                if self.prev_value:
                    vtype, vnum = self.prev_value
                    step = 1 if vtype == 'temperature' else 10
                    delta = step if direction == 'up' else -step
                    old_value = self.prev_value
                    inferred_value = (vtype, vnum + delta)

        # continuous: 짧은 bare 동사 (꺼/켜/끄/열어/닫아/취소) → fn 상속
        bare_verb_map = {
            '꺼': 'off', '꺼줘': 'off', '끄': 'off', '끄자': 'off',
            '켜': 'on', '켜줘': 'on',
            '열어': 'open', '열어줘': 'open',
            '닫아': 'close', '닫아줘': 'close', '잠가': 'close',
            '취소': 'off', '취소해': 'off', '취소해줘': 'off',
            '삭제': 'off', '삭제해': 'off',
        }
        if self.is_active() and self.prev_fn and text.strip() in bare_verb_map:
            fn = self.prev_fn
            exec_t = 'control_then_confirm'
            direction = bare_verb_map[text.strip()]

        # continuous: "{room}(조사)? {verb}" 패턴 — device 없으면 prev_fn 상속
        # 예: "주방은 꺼줘" (light 이전 턴) → light_control 상속 (ac_control 오예측 교정)
        if self.is_active() and self.prev_fn:
            has_device_kw = re.search(
                r'조명|불|램프|전등|스탠드|난방|보일러|에어컨|냉방|환기|환풍|공기청정|'
                r'가스|도어|커튼|블라인드|월패드|잠가|잠금|잠그|도어락|열쇠', text)
            room_verb_match = re.search(
                r'(거실|안방|침실|주방|부엌|작은방|아이방|서재|현관|전체|모든)'
                r'(?:은|는|이|가|도|을|를|에|의)?\s*'
                r'(?:좀\s*)?(켜|꺼|끄|열어|닫아|잠가)', text)
            if not has_device_kw and room_verb_match and fn != self.prev_fn:
                # model이 device 없는데 다른 fn을 예측 → prev_fn으로 강제 상속
                fn = self.prev_fn
                exec_t = 'control_then_confirm'
                # 매칭된 동사에서 direction도 재설정 (NLU가 틀린 direction 줄 수 있음)
                _vdir = {'켜': 'on', '꺼': 'off', '끄': 'off', '열어': 'open',
                         '닫아': 'close', '잠가': 'close'}
                _matched_verb = room_verb_match.group(2)
                if _matched_verb in _vdir:
                    direction = _vdir[_matched_verb]

        # continuous: "조금/더/많이 올려/내려" fn 상속 (prev_value 없어도 fn만이라도 승계)
        if self.is_active() and self.prev_fn:
            m = re.search(r'^\s*(더|조금|조금만|살짝|많이)\s+(올려|내려|낮춰|줄여|높여|키워)', text)
            if m and fn in ('system_meta', 'home_info', 'unknown'):
                fn = self.prev_fn
                exec_t = self.prev_exec or exec_t
                if direction == 'none':
                    direction = 'up' if m.group(2) in ('올려', '높여', '키워') else 'down'

        # "더 올려/내려/줄여" 같은 relative 발화 → 이전 value + 1/2 씩 조정
        # 이때 fn도 이전 턴에서 상속 (짧은 relative 발화에서 모델이 fn 헷갈릴 수 있음)
        if self.is_active() and self.prev_value and current_value is None:
            if re.search(r'^\s*(더|조금|조금만|살짝)\s+(올려|내려|낮춰|줄여|높여|키워)', text):
                # 짧고 명확한 relative 발화 — fn 이전 값 승계
                if self.prev_fn:
                    fn = self.prev_fn
                    exec_t = self.prev_exec or exec_t
                if re.search(r'올려|높여|키워', text):
                    direction = 'up'
                elif re.search(r'내려|낮춰|줄여', text):
                    direction = 'down'
                vtype, vnum = self.prev_value
                step = 1 if vtype == 'temperature' else 10
                delta = step if direction == 'up' else -step
                old_value = self.prev_value
                inferred_value = (vtype, vnum + delta)

        # iter10: "N도 더 올려" / "N도만 더 올려" / "N도 올려" relative + explicit number
        if self.is_active() and self.prev_value:
            m = re.search(r'(\d+)\s*도\s*(?:만)?\s*(?:더)?\s*(올려|내려|높여|낮춰|올리|내리)', text)
            if m:
                delta = int(m.group(1))
                is_up = '올려' in m.group(2) or '높여' in m.group(2)
                vtype, vnum = self.prev_value
                if vtype == 'temperature':
                    old_value = self.prev_value
                    inferred_value = (vtype, vnum + (delta if is_up else -delta))
                    # fn 상속
                    if self.prev_fn:
                        fn = self.prev_fn
                        exec_t = self.prev_exec or exec_t
                    direction = 'up' if is_up else 'down'
                    current_value = None  # override explicit extraction

        # continuous: "N도 더" / "N도 덜" / "N도 낮춰" without explicit verb (바로 temperature delta)
        if self.is_active() and self.prev_value and inferred_value is None:
            # "1도 더" / "2도 낮춰" 같은 상대값 (위 규칙이 매치 안 될 때)
            m = re.search(r'^\s*(\d+)\s*도\s*(더|덜|낮춰|높여)?\s*$', text)
            if m:
                delta = int(m.group(1))
                modifier = m.group(2) or '더'
                is_up = modifier in ('더', '높여')
                vtype, vnum = self.prev_value
                if vtype == 'temperature':
                    old_value = self.prev_value
                    inferred_value = (vtype, vnum + (delta if is_up else -delta))
                    if self.prev_fn:
                        fn = self.prev_fn
                        exec_t = self.prev_exec or exec_t
                    direction = 'up' if is_up else 'down'
                    current_value = None

        final_value = current_value or inferred_value or self.prev_value

        # 상태 저장
        self.prev_fn = fn
        self.prev_exec = exec_t
        self.prev_dir = direction
        self.prev_room = resolved_room if resolved_room != 'none' else self.prev_room
        self.prev_value = final_value if direction != 'off' else None
        self.prev_text = text
        self.prev_time = time.time()
        self.turn_count += 1

        # History 저장 (최근 5턴)
        self.history.append({
            'fn': fn, 'exec_type': exec_t, 'param_direction': direction,
            'room': resolved_room, 'text': text, 'value': final_value,
            'time': self.prev_time,
        })
        if len(self.history) > 5:
            self.history.pop(0)

        return {
            'fn': fn,
            'exec_type': exec_t,
            'param_direction': direction,
            'room': resolved_room if resolved_room != 'none' else (self.prev_room or 'none'),
            'value': final_value,
            'old_value': old_value,  # 조정 전 값 (응답 "N도에서 M도로" 용)
        }

    def _extract_value(self, text):
        """텍스트에서 value 추출 (temperature/time/percent/level/enum)"""
        if not text:
            return None
        m = re.search(r'(\d+)\s*도', text)
        if m:
            return ('temperature', int(m.group(1)))
        m = re.search(r'(\d+)\s*(분|초|시)', text)
        if m:
            unit = {'분': 'minute', '초': 'second', '시': 'hour'}[m.group(2)]
            return (unit, int(m.group(1)))
        m = re.search(r'(\d+)\s*%', text)
        if m:
            return ('percent', int(m.group(1)))
        m = re.search(r'(\d+)\s*단계', text)
        if m:
            return ('level', int(m.group(1)))
        # iter9: 자연어 레벨 (강/중/약/최대/최소)
        for kw, lvl in [('최대', 'max'), ('최소', 'min'), ('풀가동', 'max'),
                         ('강하게', 'strong'), ('세게', 'strong'),
                         ('중간', 'medium'), ('보통', 'medium'),
                         ('약하게', 'weak'), ('살짝', 'weak'), ('은은', 'weak')]:
            if kw in text:
                return ('enum', lvl)
        # iter9: 음량/밝기 등 bare number (음량 50)
        m = re.search(r'(음량|볼륨|밝기)\s+(\d+)', text)
        if m:
            return ('percent', int(m.group(2)))
        return None

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

        # "안방도", "주방도", "거실도", "안방도 해줘", "안방에도", "모든 방도" — room follow-up
        if re.search(r'^(거실|안방|주방|침실|작은방|아이방|부엌|큰방|모든\s*방|전체)(?:도|에도)(?:\s*다)?(?:\s*해?줘)?$', text_clean):
            return 'room'

        # "에어컨도", "난방도" — device follow-up (fn은 유지, exec/dir만 상속)
        if re.search(r'(에어컨|난방|조명|불|환기|커튼|가스|도어락|보일러)도$', text_clean):
            return 'device'

        # 매우 짧고 room만 있는 경우
        rooms = ['거실', '안방', '주방', '침실', '작은방', '아이방', '부엌', '큰방']
        if text_clean in rooms:
            return 'room'

        # 짧은 확인/동의
        if text_clean in ('응', '그래', '맞아', '네', '예', '좋아', '해줘', '그거',
                          '알겠어', '알겠습니다', '오케이', 'OK', 'ok', '그렇게', '그렇게 해줘',
                          '그렇게 해', '부탁해', '부탁드려요', '그렇게 해주세요',
                          '부탁', '부탁드려'):  # preprocess가 어미 제거하는 경우
            return 'confirm'

        return None

    def _is_correction(self, text):
        """교정/재설정 발화인지 ("아니 ~", "아니야 ~", "아 역시 ~", "다시 ~", "이제 ~")"""
        if not text:
            return False

        text_clean = text.strip()
        return (text_clean.startswith('아니') or
                text_clean.startswith('아냐') or
                text_clean.startswith('아 역시') or
                text_clean.startswith('다시') or
                text_clean.startswith('아님') or
                text_clean.startswith('이제'))

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
