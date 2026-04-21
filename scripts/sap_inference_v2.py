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
    'weather_query': ['다음 달 날씨', '몇 mm', '강수량'],
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
    'control_then_confirm': {
        'light_control': '네, {room}조명을 {action}.',
        'heat_control': '네, {room}난방을 {action}.',
        'ac_control': '네, {room}에어컨을 {action}.',
        'vent_control': '네, 환기시스템을 {action}.',
        'gas_control': '네, 가스 밸브를 {action}.',
        'door_control': '네, 도어락을 {action}.',
        'curtain_control': '네, {room}전동커튼을 {action}.',
        'elevator_call': '네, 엘리베이터를 호출합니다.',
        'security_mode': '네, 외출모드로 전환합니다.',
        'schedule_manage': '네, 예약이 설정되었습니다.',
        'energy_query': '네, 에너지 설정이 변경되었습니다.',
        'home_info': '네, 설정을 변경했습니다.',
    },
    'query_then_respond': {
        'light_control': '현재 조명 상태를 확인합니다.',
        'heat_control': '현재 실내 온도는 23도이며 난방 설정은 25도입니다.',
        'ac_control': '현재 에어컨은 자동 모드로 작동 중입니다.',
        'vent_control': '현재 환기 장치 상태를 확인합니다.',
        'gas_control': '현재 가스 밸브는 잠겨있습니다.',
        'door_control': '현재 도어락 상태를 확인합니다.',
        'curtain_control': '현재 전동커튼 상태를 확인합니다.',
        'elevator_call': '엘리베이터 위치를 확인합니다.',
        'security_mode': '외출모드 설정을 확인합니다.',
        'schedule_manage': '현재 예약 설정을 확인합니다.',
        'weather_query': '오늘 서울 날씨는 맑고 최고 22도입니다.',
        'news_query': '오늘 주요 뉴스를 브리핑합니다.',
        'traffic_query': '현재 교통상황을 확인합니다.',
        'energy_query': '에너지 사용량을 확인합니다.',
        'home_info': '정보를 확인합니다.',
        'market_query': '시세 정보를 확인합니다.',
        'medical_query': '근처 병원 정보를 안내합니다.',
        'vehicle_manage': '차량 정보를 확인합니다.',
        'system_meta': '시스템 정보를 확인합니다.',
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
    'clarify': '어떤 공간의 기기를 제어할지 말씀해주세요.',
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

    if exec_t == 'control_then_confirm':
        template = RESPONSE_TEMPLATES['control_then_confirm'].get(fn, '네, 처리했습니다.')
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
        return RESPONSE_TEMPLATES['query_then_respond'].get(fn, '정보를 확인합니다.')

    elif exec_t == 'query_then_judge':
        return RESPONSE_TEMPLATES['query_then_judge'].get(judge, '판단 결과를 확인합니다.')

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
        return RESPONSE_TEMPLATES['clarify']

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
