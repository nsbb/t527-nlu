#!/usr/bin/env python3
"""Semantic Action Parser v2 — CNN 5-Head + Rule-based slots + Unknown handling
사용법:
    python3 scripts/sap_inference_v2.py
    python3 scripts/sap_inference_v2.py "거실 에어컨 23도로 맞춰줘"
"""
import torch, torch.nn.functional as F, json, re, sys, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer

# ============================================================
# Rule-based Slot Extraction
# ============================================================
ROOMS = {
    '거실': 'living', '주방': 'kitchen', '부엌': 'kitchen',
    '안방': 'bedroom_main', '큰방': 'bedroom_main',
    '작은방': 'bedroom_sub', '침실': 'bedroom_sub', '아이방': 'bedroom_sub',
    '전체': 'all', '전부': 'all', '모든': 'all',
}

def extract_room(text):
    for kw, room in ROOMS.items():
        if kw in text:
            return room
    return 'none'

def extract_value(text):
    m = re.search(r'(\d+)\s*도', text)
    if m: return ('temperature', m.group(1))
    m = re.search(r'(\d+)\s*분', text)
    if m: return ('time', m.group(1))
    m = re.search(r'(\d+)\s*%', text)
    if m: return ('percent', m.group(1))
    m = re.search(r'(\d+)\s*단계', text)
    if m: return ('level', m.group(1))
    return None

# 미지원 액션 키워드 (fn은 맞지만 해당 동작을 지원하지 않는 경우)
UNSUPPORTED_ACTIONS = {
    'medical_query': ['예약', '상담', '진료', '증상', '처방'],
    'traffic_query': ['택시', '대리', '카풀', '렌트'],
    'market_query': ['추천', '매수', '매도', '예측', '수익률', '계좌', '마진', '떨어질까'],
    'news_query': ['구독', '매일 아침', '브리핑 해줘', '예약 브리핑'],
    'system_meta': ['비밀번호 바꿔', '비밀번호 추천', '비번 변경', '이름 바꾸'],
    'home_info': ['골프장 예약', '수영장 예약', '헬스장 예약', '예약 취소', '자동 밝기', '자동 절전'],
    'vent_control': ['필터 주문', '고쳐', 'AS', '수리'],
    'ac_control': ['고장', 'AS', '수리'],
    'weather_query': ['다음 달 날씨', '다음달 날씨', '몇 mm', '강수량', '강수 확률', '장기 예보'],
}

def check_unsupported(fn, text):
    """fn은 맞지만 해당 액션이 미지원인지 확인"""
    keywords = UNSUPPORTED_ACTIONS.get(fn, [])
    for kw in keywords:
        if kw in text:
            return True, f"죄송합니다. 해당 기능은 지원하지 않습니다."
    return False, None

# ============================================================
# Response Templates
# ============================================================
RESPONSE_TEMPLATES = {
    # 각 fn별 1~3개 variation — random.choice로 선택됨
    'control_then_confirm': {
        'light_control': ['네, {room}조명을 {action}.',
                           '{room}조명 {action}.',
                           '알겠습니다, {room}조명을 {action}.'],
        'heat_control': ['네, {room}난방을 {action}.',
                          '{room}난방 {action}.',
                          '{room}난방을 {action}.'],
        'ac_control': ['네, {room}에어컨을 {action}.',
                        '{room}에어컨 {action}.',
                        '알겠습니다, {room}에어컨 {action}.'],
        'vent_control': ['네, 환기시스템을 {action}.',
                          '환기 {action}.',
                          '환기 시스템 {action}.'],
        'gas_control': ['네, 가스 밸브를 {action}.',
                         '가스 밸브 {action}.'],
        'door_control': ['네, 도어락을 {action}.',
                          '도어락 {action}.'],
        'curtain_control': ['네, {room}전동커튼을 {action}.',
                             '{room}커튼 {action}.'],
        'elevator_call': ['네, 엘리베이터를 호출합니다.',
                           '엘리베이터 호출할게요.',
                           '엘리베이터 호출했습니다.'],
        'security_mode': ['네, 외출모드로 전환합니다.',
                           '외출모드로 설정했습니다.'],
        'schedule_manage': ['네, 예약이 설정되었습니다.',
                             '예약 완료되었습니다.'],
        'energy_query': ['네, 에너지 설정이 변경되었습니다.',
                          '에너지 설정을 바꿨습니다.'],
        'home_info': ['네, 설정을 변경했습니다.',
                       '설정이 완료되었습니다.'],
    },
    'query_then_respond': {
        'light_control': ['현재 조명 상태를 확인합니다.', '조명 상태를 조회합니다.'],
        'heat_control': ['현재 실내 온도는 23도이며 난방 설정은 25도입니다.',
                          '실내 23도, 난방 설정 25도입니다.'],
        'ac_control': ['현재 에어컨은 자동 모드로 작동 중입니다.',
                        '에어컨은 자동 모드입니다.'],
        'vent_control': ['현재 환기 장치 상태를 확인합니다.', '환기 상태를 조회합니다.'],
        'gas_control': ['현재 가스 밸브는 잠겨있습니다.', '가스 밸브는 잠김 상태입니다.'],
        'door_control': ['현재 도어락 상태를 확인합니다.', '도어락 상태를 조회합니다.'],
        'curtain_control': ['현재 전동커튼 상태를 확인합니다.', '커튼 상태를 조회합니다.'],
        'elevator_call': ['엘리베이터 위치를 확인합니다.', '엘리베이터가 어디 있는지 보겠습니다.'],
        'security_mode': ['외출모드 설정을 확인합니다.', '외출모드 상태를 조회합니다.'],
        'schedule_manage': ['현재 예약 설정을 확인합니다.', '예약 상태를 조회합니다.'],
        'weather_query': ['오늘 서울 날씨는 맑고 최고 22도입니다.',
                           '서울 날씨는 맑고 기온은 22도예요.'],
        'news_query': ['오늘 주요 뉴스를 브리핑합니다.', '오늘 뉴스 들려드릴게요.'],
        'traffic_query': ['현재 교통상황을 확인합니다.', '교통 상황을 조회합니다.'],
        'energy_query': ['에너지 사용량을 확인합니다.', '에너지 사용량을 보겠습니다.'],
        'home_info': ['정보를 확인합니다.', '정보를 조회합니다.'],
        'market_query': ['시세 정보를 확인합니다.', '시세를 조회합니다.'],
        'medical_query': ['근처 병원 정보를 안내합니다.', '근처 병원을 찾아드립니다.'],
        'vehicle_manage': ['차량 정보를 확인합니다.', '차량 정보를 조회합니다.'],
        'system_meta': ['시스템 정보를 확인합니다.', '시스템 정보를 조회합니다.'],
    },
    'query_then_judge': {
        'outdoor_activity': '기온과 대기질 모두 양호합니다. 외출하기에 무리가 없습니다.',
        'clothing': '오늘 기온은 22도로 따뜻합니다. 얇은 긴팔이 적당합니다.',
        'air_quality': '현재 미세먼지 양호 수준으로 창문을 열어 환기하기 적절합니다.',
        'cost_trend': '최근 안정적인 수준입니다.',
    },
    'direct_respond': {
        'system_meta': {
            'capability': '네, 저는 조명 등 집안 기기 상태를 조회하고 조절 및 예약할 수 있어요.',
            'creator': '저는 HDC랩스에서 개발한 AI모델입니다.',
            'name': "제가 필요할 때 '하이 원더'라고 호출해주세요.",
            'usage': '월패드 화면에서 기능을 확인하실 수 있습니다.',
        },
        'home_info': {
            'time': '네, 지금은 오후 2시 30분입니다.',
            'default': '정보를 확인합니다.',
        },
    },
    'clarify': ['어떤 공간의 기기를 제어할지 말씀해주세요.',
                 '어느 방의 기기를 말하시는 건가요?',
                 '공간명을 포함해서 다시 말씀해주세요.'],
}

ROOM_KR = {'living':'거실 ','kitchen':'주방 ','bedroom_main':'안방 ','bedroom_sub':'침실 ',
           'all':'전체 ','none':'','external':''}

ACTION_MAP = {
    'on': '켰습니다', 'off': '껐습니다', 'open': '열었습니다', 'close': '닫았습니다',
    'up': '올렸습니다', 'down': '내렸습니다', 'set': '설정했습니다', 'stop': '중단했습니다',
}


def generate_response(preds, text):
    fn = preds['fn']
    exec_t = preds['exec_type']
    direction = preds['param_direction']
    judge = preds['judge']
    room = extract_room(text)
    room_kr = ROOM_KR.get(room, '')
    value = extract_value(text)

    # Unknown → 서버로
    if fn == 'unknown':
        return '해당 요청은 서버에서 처리합니다.'

    # 미지원 액션 체크
    is_unsupported, unsupported_msg = check_unsupported(fn, text)
    if is_unsupported:
        return unsupported_msg

    import random

    def pick(v, default=''):
        """list면 random, 문자열이면 그대로."""
        if isinstance(v, list):
            return random.choice(v) if v else default
        return v if v else default

    if exec_t == 'control_then_confirm':
        tmpl_raw = RESPONSE_TEMPLATES['control_then_confirm'].get(fn, ['네, 처리했습니다.'])
        template = pick(tmpl_raw, '네, 처리했습니다.')
        action = ACTION_MAP.get(direction, '설정했습니다')
        # value가 있으면 "X도로/X분으로" 형식으로 간결하게 (중복 방지)
        if value and value[0] == 'temperature':
            action = f'{value[1]}도로 설정했습니다'
        elif value and value[0] == 'time':
            action = f'{value[1]}분 예약을 설정했습니다'
        elif value and value[0] == 'percent':
            action = f'{value[1]}%로 설정했습니다'
        elif value and value[0] == 'level':
            action = f'{value[1]}단계로 설정했습니다'
        elif preds['param_type'] == 'mode':
            for mode in ['제습','송풍','자동','냉방','외출','재택','취침','수면']:
                if mode in text:
                    action = f'{mode} 모드로 설정했습니다'
                    break
        return template.format(room=room_kr, action=action)

    elif exec_t == 'query_then_respond':
        return pick(RESPONSE_TEMPLATES['query_then_respond'].get(fn), '정보를 확인합니다.')

    elif exec_t == 'query_then_judge':
        return pick(RESPONSE_TEMPLATES['query_then_judge'].get(judge), '판단 결과를 확인합니다.')

    elif exec_t == 'direct_respond':
        templates = RESPONSE_TEMPLATES['direct_respond'].get(fn, {})
        if isinstance(templates, dict):
            lo = text.lower()
            if any(w in lo for w in ['할 수 있','기능','뭐 할']): return templates.get('capability','정보를 확인합니다.')
            if any(w in lo for w in ['만들','개발']): return templates.get('creator','정보를 확인합니다.')
            if any(w in lo for w in ['이름','뭐라고 불']): return templates.get('name','정보를 확인합니다.')
            if any(w in lo for w in ['사용법','어떻게 써']): return templates.get('usage','정보를 확인합니다.')
            if any(w in lo for w in ['몇 시','시간']): return templates.get('time','정보를 확인합니다.')
            return templates.get('default','정보를 확인합니다.')
        return str(templates)

    elif exec_t == 'clarify':
        return pick(RESPONSE_TEMPLATES['clarify'], '다시 말씀해주세요.')

    return '처리합니다.'


# ============================================================
# Pipeline
# ============================================================
class SAPv2Pipeline:
    def __init__(self):
        print("모델 로딩 중...")
        sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        pw = sbert.embeddings.word_embeddings.weight.detach()
        del sbert

        self.model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
        ckpt = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['state'])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained('tokenizer/')

        # DST 초기화
        from dialogue_state_tracker import DialogueStateTracker
        self.dst = DialogueStateTracker(timeout=10)
        params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"로딩 완료 — 5 heads, {params/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
        print(f"  epoch {ckpt['epoch']}, combo {ckpt['combo']:.1f}%")

    def predict(self, text):
        from preprocess import preprocess
        text = preprocess(text)
        if not text:
            return {h: 'none' for h in HEAD_NAMES}, 0.0

        tk = self.tokenizer(text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(tk['input_ids'])

        preds = {h: HEAD_I2L[h][logits[h].argmax(1).item()] for h in HEAD_NAMES}
        confidence = F.softmax(logits['fn'], dim=1).max().item()

        # confidence fallback — 저신뢰 발화는 unknown으로
        # iter8 참고: 3-tier (0.3/0.5) 실험 → TS -0.04%p, UX 이득 불확실 → 2-tier 유지
        if confidence < 0.5 and preds['fn'] != 'unknown':
            preds['fn'] = 'unknown'
            preds['exec_type'] = 'direct_respond'

        # param_type 규칙 보정
        if preds['param_direction'] in ('open', 'close', 'stop'):
            preds['param_type'] = 'none'
        if preds['judge'] != 'none':
            preds['param_type'] = 'none'
        if preds['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
            preds['param_type'] = 'none'

        # dir 규칙 보정 (v28 학습 오류 교정, 2026-04-21 발견)
        # "밝게" → up (v28이 down으로 잘못 학습한 케이스 다수)
        if re.search(r'밝게', text) and preds['param_direction'] == 'down':
            preds['param_direction'] = 'up'
            preds['param_type'] = 'brightness'
        # "어둡게" → down (교차 검증용)
        if re.search(r'어둡게', text) and preds['param_direction'] in ('up', 'on'):
            preds['param_direction'] = 'down'
            preds['param_type'] = 'brightness'

        # 알람/모닝콜 → schedule_manage (iter8+iter9 refinement)
        # device keyword 없을 때만, dir은 TS 불일치 때문에 모델 예측 유지
        has_device = re.search(r'조명|불|램프|난방|에어컨|환기|가스|도어|커튼|공기청정|블라인드', text)
        if not has_device and re.search(r'알람|모닝콜', text) and preds['fn'] in ('system_meta', 'home_info', 'unknown'):
            preds['fn'] = 'schedule_manage'

        # Out-of-domain keywords → unknown (iter8, v46 unknown_to_known 오류 완화)
        # 주의: "전화" (관리실 전화), "카드" (출입카드), "와이파이" 는 in-domain 일 수 있어 제외
        OOD_KEYWORDS = ['네비게이션', '비행기', '크루즈', '수면 기록', '길 안내']
        if any(kw in text for kw in OOD_KEYWORDS):
            preds['fn'] = 'unknown'
            preds['exec_type'] = 'direct_respond'
            preds['param_direction'] = 'none'
            preds['param_type'] = 'none'

        # iter9: "전화" entity 없으면 OOD (관리사무소/경비/이웃/놓친 제외)
        if '전화' in text and preds['fn'] == 'home_info':
            entity_markers = ['관리사무소', '관리실', '경비', '이웃', '주민', '같은 동',
                               '다른 집', '분리수거', '공동', '놓친']
            if not any(kw in text for kw in entity_markers):
                preds['fn'] = 'unknown'
                preds['exec_type'] = 'direct_respond'
                preds['param_direction'] = 'none'
                preds['param_type'] = 'none'

        # iter9: "{room} {device} 좀 {verb}" 어순은 CTC (clarify는 "{room} 좀 {device} {verb}" 어순)
        # 조사(에/은/의) 허용
        if preds['exec_type'] == 'clarify' and preds['fn'] == 'light_control':
            if re.search(r'(거실|안방|침실|주방|부엌|작은방|아이방|서재|현관)(?:에|은|의)?\s+(불|조명|등|라이트)\s+좀\s+(켜|꺼|끄)', text):
                preds['exec_type'] = 'control_then_confirm'
                if preds['param_direction'] == 'none':
                    preds['param_direction'] = 'on' if re.search(r'켜', text) else 'off'

        # iter9: curtain_control 올려 → up (pred=open 포함), 블라인드 내려 → close
        if preds['fn'] == 'curtain_control' and '올려' in text and preds['param_direction'] in ('stop', 'none', 'open'):
            preds['param_direction'] = 'up'
        if preds['fn'] == 'curtain_control' and '블라인드' in text and '내려' in text:
            if preds['param_direction'] in ('down', 'none', 'open'):
                preds['param_direction'] = 'close'
        # 블라인드만 있고 action 없음 → stop
        if preds['fn'] == 'curtain_control' and '블라인드' in text and not re.search(r'올려|내려|열어|닫아|멈춰|스톱', text):
            if preds['param_direction'] == 'open':
                preds['param_direction'] = 'stop'

        # iter9: heat_control CTC + dir=none → on (기본 동작은 켜기)
        if preds['fn'] == 'heat_control' and preds['exec_type'] == 'control_then_confirm' and preds['param_direction'] == 'none':
            preds['param_direction'] = 'on'

        # iter9: 공기청정/공기 정화 → vent_control
        if re.search(r'공기청정|공기\s*정화|공기\s*청정', text):
            if preds['fn'] in ('weather_query', 'unknown', 'home_info'):
                preds['fn'] = 'vent_control'
                if preds['exec_type'] == 'direct_respond':
                    preds['exec_type'] = 'control_then_confirm'
            if preds['fn'] == 'vent_control' and preds['param_direction'] == 'none':
                if '켜' in text or '가동' in text or '작동' in text:
                    preds['param_direction'] = 'on'
                elif '꺼' in text or '끄' in text:
                    preds['param_direction'] = 'off'

        # reflection: 덥다/더워/덥네 → ac_control
        if preds['fn'] == 'heat_control' and re.search(r'덥다|더워|덥네|더운', text):
            if not re.search(r'난방|보일러|온돌', text):
                preds['fn'] = 'ac_control'
                if preds['param_direction'] in ('none', 'up'):
                    preds['param_direction'] = 'on'

        # reflection: 춥다/추워 → heat_control
        if preds['fn'] == 'ac_control' and re.search(r'춥다|추워|추운', text):
            if not re.search(r'에어컨|냉방', text):
                preds['fn'] = 'heat_control'
                if preds['param_direction'] == 'none':
                    preds['param_direction'] = 'on'

        # reflection: 시원하게 → ac_control
        if preds['fn'] == 'heat_control' and re.search(r'시원', text):
            if not re.search(r'난방|보일러', text):
                preds['fn'] = 'ac_control'
                if preds['param_direction'] == 'none':
                    preds['param_direction'] = 'on'

        # reflection: 블라인드 닫아 → close
        if preds['fn'] == 'curtain_control' and re.search(r'닫아|닫기', text):
            if preds['param_direction'] == 'open':
                preds['param_direction'] = 'close'

        # reflection: query fn + query exec + spurious dir → none
        if preds['fn'] in ('weather_query', 'news_query', 'traffic_query',
                            'market_query', 'medical_query') and preds['exec_type'] == 'query_then_respond':
            if preds['param_direction'] != 'none':
                preds['param_direction'] = 'none'

        # continuous: 커튼 내려 → down (블라인드는 close)
        if preds['fn'] == 'curtain_control' and '커튼' in text and '내려' in text and '블라인드' not in text:
            if preds['param_direction'] in ('stop', 'none', 'open'):
                preds['param_direction'] = 'down'

        # continuous: 현관 → door_control (curtain 오예측)
        if preds['fn'] == 'curtain_control' and '현관' in text:
            preds['fn'] = 'door_control'
            if preds['param_direction'] == 'stop':
                if '닫' in text or '잠' in text:
                    preds['param_direction'] = 'close'
                elif '열' in text:
                    preds['param_direction'] = 'open'

        # continuous: 예약 확인 → schedule_manage
        if preds['fn'] == 'home_info' and re.search(r'예약\s*확인|예약\s*정보', text):
            preds['fn'] = 'schedule_manage'

        # continuous: energy_query + 추워/더워 weather 문맥
        if preds['fn'] == 'energy_query' and re.search(r'추워|더워|덥|춥', text):
            if re.search(r'작년|올해|이번 해|지난 해|과거', text) or '?' in text:
                preds['fn'] = 'weather_query'
                preds['exec_type'] = 'query_then_respond'
                preds['param_direction'] = 'none'

        # continuous: system_meta OOD 예외
        if preds['fn'] == 'system_meta':
            if re.search(r'와이파이\s*비번|영어로\s*뭐|업데이트$|^일정$', text):
                preds['fn'] = 'unknown'
                preds['exec_type'] = 'direct_respond'
                preds['param_direction'] = 'none'

        # continuous: ac/vent + 해줘 → on
        if preds['fn'] in ('ac_control', 'vent_control') and preds['exec_type'] == 'control_then_confirm' and preds['param_direction'] == 'none':
            if re.search(r'해줘|해\s*줘|틀어|가동|작동', text):
                preds['param_direction'] = 'on'

        # continuous: 통행/교통 → traffic_query
        if preds['fn'] == 'unknown':
            if re.search(r'통행|교통|소요\s*시간|얼마나\s*걸려|몇\s*분\s*걸려', text):
                preds['fn'] = 'traffic_query'
                preds['exec_type'] = 'query_then_respond'

        # continuous: judgment (타도 돼/나가도 돼 등) → weather_query
        if re.search(r'타도\s*돼\?|나가도\s*돼|세차해도|운동해도|소풍', text):
            if preds['fn'] in ('market_query', 'traffic_query', 'unknown'):
                preds['fn'] = 'weather_query'
                preds['exec_type'] = 'query_then_judge'
                preds['judge'] = 'outdoor_activity'

        # continuous: 난방 keyword → heat force
        if '난방' in text and preds['fn'] == 'light_control':
            preds['fn'] = 'heat_control'

        # continuous: 환해/밝 → light (vent 오예측)
        if preds['fn'] == 'vent_control' and re.search(r'환해|환하|밝다|밝아', text):
            preds['fn'] = 'light_control'

        # continuous: 현관→door, 예약 확인→schedule
        if preds['fn'] == 'curtain_control' and '현관' in text:
            preds['fn'] = 'door_control'
        if preds['fn'] == 'home_info' and re.search(r'예약\s*확인|예약\s*정보', text):
            preds['fn'] = 'schedule_manage'

        # continuous: OOD single words
        if text.strip() in ('등산', '카드', '녹화', '토토', '경마', '선풍기', '음식 주문', '택배 조회'):
            preds['fn'] = 'unknown'
            preds['exec_type'] = 'direct_respond'
            preds['param_direction'] = 'none'

        # continuous: 비상 상황 → security_mode
        if re.search(r'가스\s*냄새|연기\s*나|불\s*났|침입', text):
            preds['fn'] = 'security_mode'
            preds['exec_type'] = 'control_then_confirm'
            preds['param_direction'] = 'on'

        # continuous: clarify→CTC (room + adverb + 불)
        if preds['exec_type'] == 'clarify' and preds['fn'] == 'light_control':
            if re.search(r'(거실|안방|침실|주방|부엌|작은방|아이방)\s+(?:지금|혹시|야)\s+(불|조명|등)\s+(켜|꺼)', text):
                preds['exec_type'] = 'control_then_confirm'

        # iter9: 화면/월패드/알림/음량 → home_info (capability query 제외)
        capability_q = re.search(r'어떻게|할\s*수\s*있', text)
        if preds['fn'] == 'system_meta' and not capability_q:
            if re.search(r'화면\s*밝기|월패드\s*밝기|음량', text):
                preds['fn'] = 'home_info'
            elif re.search(r'알림', text) and not re.search(r'사용량|긴급|에너지', text):
                preds['fn'] = 'home_info'

        # 외부 쿼리 keyword → 해당 fn (iter8, v46 known_to_unknown 9건 완화)
        if preds['fn'] == 'unknown':
            if re.search(r'날씨|기온|비\s*와|더울까|추울까|맑|흐림', text):
                preds['fn'] = 'weather_query'
                preds['exec_type'] = 'query_then_respond'
            elif re.search(r'뉴스|브리핑|속보', text):
                preds['fn'] = 'news_query'
                preds['exec_type'] = 'query_then_respond'
            elif re.search(r'병원|의원|약국|신경외과|내과|외과|안과|치과|한의원|한방|약\s*처', text):
                preds['fn'] = 'medical_query'
                preds['exec_type'] = 'query_then_respond'


        return preds, confidence

    def run(self, text):
        preds, confidence = self.predict(text)

        # Room/Value from rules
        room = extract_room(text)
        value = extract_value(text)

        # DST (멀티턴 대화 상태 추적)
        if hasattr(self, 'dst'):
            resolved = self.dst.update(preds, room, text)
            preds['fn'] = resolved['fn']
            preds['exec_type'] = resolved['exec_type']
            preds['param_direction'] = resolved['param_direction']
            room = resolved['room']

        response = generate_response(preds, text)

        # 요약
        parts = [f"fn={preds['fn']}", f"exec={preds['exec_type']}"]
        if preds['param_direction'] != 'none': parts.append(f"dir={preds['param_direction']}")
        if room != 'none': parts.append(f"room={room}")
        if preds['param_type'] != 'none': parts.append(f"param={preds['param_type']}")
        if preds['judge'] != 'none': parts.append(f"judge={preds['judge']}")
        if value: parts.append(f"value={value[1]}{value[0]}")
        parts.append(f"conf={confidence:.2f}")

        return {
            'input': text,
            'preds': preds,
            'room': room,
            'value': value,
            'confidence': confidence,
            'summary': ', '.join(parts),
            'response': response,
        }


if __name__ == '__main__':
    sap = SAPv2Pipeline()

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        result = sap.run(text)
        print(f"\n입력: {result['input']}")
        print(f"Action: {result['summary']}")
        print(f"응답: {result['response']}")
    else:
        print("\n=== SAP v2 대화형 모드 (종료: q) ===\n")
        while True:
            text = input("사용자: ").strip()
            if text.lower() in ('q', 'quit', 'exit', '종료'):
                break
            if not text:
                continue
            result = sap.run(text)
            print(f"  → Action: {result['summary']}")
            print(f"  → 응답: {result['response']}")
            print()
