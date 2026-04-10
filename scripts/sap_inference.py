#!/usr/bin/env python3
"""Semantic Action Parser — 대화형 추론
사용법:
    python3 scripts/sap_inference.py
    python3 scripts/sap_inference.py "거실 에어컨 23도로 맞춰줘"
"""
import torch, torch.nn as nn, json, re, sys, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')

from model_sap import *
from transformers import AutoModel, AutoTokenizer

# ============================================================
# Action Executor
# ============================================================
RESPONSE_TEMPLATES = {
    'control_then_confirm': {
        'light_control': '네, {room}{device}을 {action}.',
        'heat_control': '네, {room}난방을 {action}.',
        'ac_control': '네, {room}에어컨을 {action}.',
        'vent_control': '네, {room}환기시스템을 {action}.',
        'gas_control': '네, 가스 밸브를 {action}.',
        'door_control': '네, {room}문을 열었습니다.',
        'curtain_control': '네, {room}전동커튼을 {action}.',
        'elevator_call': '네, 엘리베이터를 호출합니다.',
        'security_mode': '네, {mode} 모드로 전환합니다.',
        'schedule_manage': '네, 예약이 설정되었습니다.',
        'energy_query': '네, 에너지 설정이 변경되었습니다.',
    },
    'query_then_respond': {
        'heat_control': '현재 실내 온도는 23도이며 난방 설정은 25도로 되어 있습니다.',
        'ac_control': '현재 에어컨은 자동 모드로 작동 중이며 설정 온도는 22도입니다.',
        'light_control': '현재 조명 상태를 확인합니다.',
        'vent_control': '현재 환기 장치는 동작 중입니다.',
        'gas_control': '현재 가스 밸브는 잠겨있습니다.',
        'door_control': '현재 도어락 상태를 확인합니다.',
        'curtain_control': '현재 전동커튼 상태를 확인합니다.',
        'elevator_call': '엘리베이터 위치를 확인합니다.',
        'weather_query': '오늘 서울 날씨는 맑고 최고 22도입니다.',
        'news_query': '오늘 주요 뉴스 3건을 브리핑합니다.',
        'traffic_query': '현재 교통상황을 반영하여 약 30분 소요됩니다.',
        'energy_query': '최근 3개월 간 에너지 사용량이 감소하고 있습니다.',
        'info_query': '정보를 조회합니다.',
        'schedule_manage': '현재 예약 설정을 확인합니다.',
        'security_mode': '외출모드 설정을 확인합니다.',
    },
    'query_then_judge': {
        'outdoor_activity': '기온과 대기질을 확인합니다. {result}',
        'clothing': '오늘 기온을 확인합니다. {clothing_advice}',
        'air_quality': '현재 미세먼지 수준을 확인합니다. {result}',
        'cost_trend': '최근 가격 추세를 확인합니다. {result}',
    },
    'direct_respond': {
        'info_query': {
            'default': '해당 기능은 지원하지 않습니다.',
            'capability': '네, 저는 사용자 요청에 따라 조명 등 집안 기기 상태를 조회하고 조절 및 예약할 수 있어요.',
            'creator': '저는 HDC랩스에서 개발한 AI모델입니다.',
            'name': "제가 필요할 때 '하이 원더'라고 호출해주세요.",
            'usage': '월패드 화면에서 기능을 확인하실 수 있습니다.',
            'time': '네, 지금은 오후 2시 30분입니다.',
        },
    },
    'clarify': '어떤 공간의 {device}을 제어할지 말씀해주세요.',
}

ROOM_KR = {'living':'거실 ','kitchen':'주방 ','bedroom_main':'안방 ','bedroom_sub':'침실 ',
           'all':'전체 ','external':'','none':'','ambiguous':''}
DEVICE_KR = {'light_control':'조명','heat_control':'난방','ac_control':'에어컨',
             'vent_control':'환기','curtain_control':'커튼','gas_control':'가스',
             'door_control':'문','elevator_call':'엘리베이터','security_mode':'보안',
             'schedule_manage':'예약','weather_query':'날씨','news_query':'뉴스',
             'traffic_query':'교통','energy_query':'에너지','info_query':'정보'}

def generate_response(preds, text):
    exec_t = preds['exec_type']
    fn = preds['fn']
    room = ROOM_KR.get(preds['room'], '')
    device = DEVICE_KR.get(fn, '')
    judge = preds.get('judge', 'none')
    param = preds.get('param_type', 'none')

    if exec_t == 'control_then_confirm':
        template = RESPONSE_TEMPLATES['control_then_confirm'].get(fn, '네, 처리했습니다.')
        # direction으로 action 결정
        direction = preds.get('param_direction', 'on')
        action_map = {
            'on': '켰습니다', 'off': '껐습니다', 'open': '열었습니다', 'close': '닫았습니다',
            'up': '올렸습니다', 'down': '내렸습니다', 'set': '설정합니다', 'stop': '중단했습니다',
        }
        action = action_map.get(direction, '설정했습니다')
        if param == 'mode':
            for mode in ['제습','송풍','자동','냉방','외출','재택']:
                if mode in text:
                    action = f'{mode} 모드로 설정합니다'
                    break
        if param == 'temperature':
            m = re.search(r'(\d+)\s*도', text)
            if m:
                action = f'온도를 {m.group(1)}도로 설정합니다'
            elif direction == 'up':
                action = '온도를 올렸습니다'
            elif direction == 'down':
                action = '온도를 내렸습니다'
        if param == 'brightness':
            if direction == 'up': action = '밝기를 올렸습니다'
            elif direction == 'down': action = '밝기를 낮췄습니다'
        return template.format(room=room, device=device, action=action, mode=preds.get('param_type',''))

    elif exec_t == 'query_then_respond':
        return RESPONSE_TEMPLATES['query_then_respond'].get(fn, '정보를 확인합니다.')

    elif exec_t == 'query_then_judge':
        if judge == 'outdoor_activity':
            return '기온과 대기질 모두 양호합니다. 외출하기에 무리가 없습니다.'
        elif judge == 'clothing':
            return '오늘 기온은 22도로 따뜻합니다. 얇은 긴팔이 적당합니다.'
        elif judge == 'air_quality':
            return '현재 미세먼지 양호 수준으로 창문을 열어 환기하기 적절합니다.'
        elif judge == 'cost_trend':
            return '최근 안정적인 수준입니다.'
        return '판단 결과를 확인합니다.'

    elif exec_t == 'direct_respond':
        direct_map = RESPONSE_TEMPLATES['direct_respond'].get(fn, {})
        if isinstance(direct_map, dict):
            lo = text.lower()
            if any(w in lo for w in ['할 수 있','기능','뭐 할']): return direct_map.get('capability', direct_map.get('default',''))
            if any(w in lo for w in ['만들','개발']): return direct_map.get('creator', direct_map.get('default',''))
            if any(w in lo for w in ['이름','뭐라고 불']): return direct_map.get('name', direct_map.get('default',''))
            if any(w in lo for w in ['사용법','어떻게 써']): return direct_map.get('usage', direct_map.get('default',''))
            if any(w in lo for w in ['몇 시','시간']): return direct_map.get('time', direct_map.get('default',''))
            return direct_map.get('default', '해당 기능은 지원하지 않습니다.')
        return str(direct_map)

    elif exec_t == 'clarify':
        return RESPONSE_TEMPLATES['clarify'].format(device=device)

    return '처리합니다.'


# ============================================================
# Pipeline
# ============================================================
class SAPPipeline:
    def __init__(self):
        print("모델 로딩 중...")
        sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        self.pw = sbert.embeddings.word_embeddings.weight.detach()
        del sbert

        self.model = SemanticActionParser(self.pw, d_model=256, num_layers=3, dropout=0.1)
        ckpt = torch.load('checkpoints/sap_best.pt', map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt['state'])
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained('tokenizer/')
        self.last_fn = 'NONE'
        self.last_time = 0
        print(f"로딩 완료 — 7 heads, {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")

    def predict(self, text):
        text = ''.join(c if c.isprintable() or c == ' ' else ' ' for c in text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return {h: 'none' for h in HEAD_NAMES}

        # 멀티턴: 10초 내면 이전 fn 컨텍스트 사용
        now = time.time()
        ctx_token = self.last_fn if (now - self.last_time) < 10 else 'NONE'
        ctx_id = CTX_L2I.get(ctx_token, 0)

        tk = self.tokenizer(text, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(tk['input_ids'], torch.tensor([ctx_id]))
            preds = {h: HEAD_I2L[h][logits[h].argmax(1).item()] for h in HEAD_NAMES}

        # 세션 업데이트
        fn_upper = preds['fn'].upper()
        self.last_fn = fn_upper if fn_upper in CTX_L2I else 'NONE'
        self.last_time = now

        return preds

    def run(self, text):
        preds = self.predict(text)
        response = generate_response(preds, text)

        # 요약 출력
        parts = [f"fn={preds['fn']}", f"exec={preds['exec_type']}"]
        if preds.get('param_direction', 'none') != 'none': parts.append(f"dir={preds['param_direction']}")
        if preds['room'] not in ('none', ''): parts.append(f"room={preds['room']}")
        if preds['param_type'] != 'none': parts.append(f"param={preds['param_type']}")
        if preds['judge'] != 'none': parts.append(f"judge={preds['judge']}")
        if preds['api'] not in ('none', ''): parts.append(f"api={preds['api']}")
        if preds['multi_action'] == 'composite': parts.append("multi=composite")

        return {
            'input': text,
            'action_struct': preds,
            'summary': ', '.join(parts),
            'response': response,
        }


if __name__ == '__main__':
    sap = SAPPipeline()

    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        result = sap.run(text)
        print(f"\n입력: {result['input']}")
        print(f"Action: {result['summary']}")
        print(f"응답: {result['response']}")
    else:
        print("\n=== Semantic Action Parser 대화형 모드 (종료: q) ===\n")
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
