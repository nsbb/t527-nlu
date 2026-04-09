#!/usr/bin/env python3
"""NLU 추론 + 응답 생성 파이프라인
사용법:
    python scripts/nlu_inference.py "난방 켜줘"
    python scripts/nlu_inference.py    # 대화형 모드
"""
import torch, torch.nn as nn, json, re, sys, os

# 프로젝트 루트로 이동
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

from transformers import AutoModel, AutoTokenizer

# ============================================================
# 모델
# ============================================================
class PretrainedCNN(nn.Module):
    def __init__(self, pw, nc=94, ml=32, cd=256, ks=[3,5,7,3], dr=0.1):
        super().__init__()
        vs, ed = pw.shape
        self.embedding = nn.Embedding.from_pretrained(pw, freeze=True, padding_idx=0)
        self.proj = nn.Linear(ed, cd)
        layers = []
        for k in ks:
            layers += [nn.Conv1d(cd, cd, k, padding=k//2), nn.ReLU(), nn.BatchNorm1d(cd), nn.Dropout(dr)]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(cd, nc)
        self.max_len = ml

    def forward(self, x):
        sl = min(x.shape[1], self.max_len)
        x = self.proj(self.embedding(x[:, :sl].long())).permute(0, 2, 1)
        return self.fc(self.convs(x).mean(dim=2))

# ============================================================
# Slot 파서 (룰 기반)
# ============================================================
ROOMS = ['거실', '안방', '침실', '주방', '서재', '드레스룸', '아이방', '큰방', '작은방', '다용도실', '발코니', '화장실', '복도']
AC_MODES = {'제습': '제습', '송풍': '송풍', '자동': '자동', '냉방': '냉방', '난방': '난방', '자동모드': '자동', '팬온리': '송풍'}
WIND_LEVELS = {'세게': '강풍', '강하게': '강풍', '약하게': '약풍', '줄여': '중풍', '올려': '강풍'}

def parse_slots(text):
    slots = {}
    # 방
    for r in ROOMS:
        if r in text:
            slots['room'] = r
            break
    if 'room' not in slots:
        if '전체' in text:
            slots['room'] = '전체'
        else:
            slots['room'] = ''

    # 온도
    m = re.search(r'(\d+)\s*도', text)
    if m:
        slots['temp'] = m.group(1)

    # 에어컨 모드
    for keyword, mode in AC_MODES.items():
        if keyword in text:
            slots['mode'] = mode
            break

    # 풍량
    for keyword, level in WIND_LEVELS.items():
        if keyword in text:
            slots['level'] = level
            break

    # 시간
    m = re.search(r'(\d+)\s*시', text)
    if m:
        slots['hour'] = m.group(1)
    m = re.search(r'(\d+)\s*분', text)
    if m:
        slots['minute'] = m.group(1)

    # 위치/지역
    m = re.search(r'(서울|부산|대구|인천|광주|대전|울산|세종|제주|경기|강원|충북|충남|전북|전남|경북|경남)', text)
    if m:
        slots['location'] = m.group(1)

    return slots

# ============================================================
# 응답 생성
# ============================================================
def load_response_config():
    with open('data/intent_response_config.json', encoding='utf-8') as f:
        return json.load(f)

def generate_response(intent, text, config):
    c = config.get(intent)
    if not c:
        return f"[{intent}] 응답 설정 없음"

    slots = parse_slots(text)
    resp_type = c['type']

    # === 고정 응답 ===
    if resp_type == 'fixed':
        return c['response']

    # === 기기 제어 ===
    elif resp_type == 'device':
        device = c.get('device', '')
        action = c.get('action', '')
        template = c.get('response_template', '')

        # 슬롯 채우기
        room = slots.get('room', '')
        room_prefix = f"{room} " if room else ""

        fill = {
            'room': room_prefix.strip(),
            'temp': slots.get('temp', ''),
            'mode': slots.get('mode', ''),
            'level': slots.get('level', ''),
            'time': slots.get('hour', ''),
        }
        try:
            result = template.format(**fill)
            return re.sub(r'\s+', ' ', result).strip()  # 중복 공백 제거
        except (KeyError, IndexError):
            return template.replace('{room}', room_prefix.strip())

    # === API 조회 ===
    elif resp_type == 'api':
        api = c.get('api', '')
        template = c.get('response_template', '')

        # 실제 API 호출 대신 시뮬레이션
        sim_data = {
            'time': f"네, 지금은 14시 30분입니다.",
            'weather': f"오늘 {slots.get('location', '서울')} 날씨는 맑고 최고 22도입니다.",
            'air_quality': f"현재 미세먼지는 보통 수준입니다.",
            'news': "오늘 주요 뉴스 3건을 브리핑합니다.",
            'traffic': f"현재 교통상황을 반영하여 약 30분 소요됩니다.",
            'bus': "000번 버스가 5분 후 도착 예정입니다.",
            'stock': "현재 코스피는 2,620포인트입니다.",
            'fuel': "전국 평균 휘발유는 1,650원입니다.",
            'hospital': "반경 1km 내 병원 3곳이 있습니다.",
            'energy': "최근 3개월 간 에너지 사용량이 감소하고 있습니다.",
            'complex_info': "관리사무소 연락처는 000-0000-0000입니다.",
            'notification': "새로운 알림이 없습니다.",
            'notice': "새로 등록된 단지소식이 없습니다.",
        }

        if api in sim_data:
            return sim_data[api]
        elif template:
            return template
        else:
            return f"[{intent}] {api} API 호출"

    return f"[{intent}] 처리 불가"

# ============================================================
# NLU 파이프라인
# ============================================================
class NLUPipeline:
    def __init__(self, model_path='checkpoints/cnn_sbert_v6_best.pt'):
        print("모델 로딩 중...")
        # 임베딩
        sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        self.emb_weights = sbert.embeddings.word_embeddings.weight.detach()
        del sbert

        # 모델
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        self.i2l = {int(k): v for k, v in ckpt['i2l'].items()}
        self.model = PretrainedCNN(self.emb_weights, nc=len(self.i2l))
        self.model.load_state_dict(ckpt['state'])
        self.model.eval()

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained('tokenizer/')

        # 응답 설정
        self.config = load_response_config()
        print(f"로딩 완료 — {len(self.i2l)}개 intent")

    def predict(self, text):
        # 특수문자/공백 정리
        text = text.strip()
        if not text:
            return 'manual_capability', 0.0
        tokens = self.tokenizer(text, padding="max_length", truncation=True,
                                max_length=32, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(tokens["input_ids"])
            probs = torch.softmax(logits, dim=1)
            confidence, pred_id = probs.max(dim=1)

        intent = self.i2l[pred_id.item()]
        conf = confidence.item()
        return intent, conf

    def run(self, text):
        intent, conf = self.predict(text)
        response = generate_response(intent, text, self.config)
        return {
            'input': text,
            'intent': intent,
            'confidence': round(conf, 4),
            'response': response,
        }

# ============================================================
# 실행
# ============================================================
if __name__ == '__main__':
    nlu = NLUPipeline()

    if len(sys.argv) > 1:
        # 커맨드라인 모드
        text = ' '.join(sys.argv[1:])
        result = nlu.run(text)
        print(f"\n입력: {result['input']}")
        print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
        print(f"응답: {result['response']}")
    else:
        # 대화형 모드
        print("\n=== NLU 대화형 모드 (종료: q) ===\n")
        while True:
            text = input("사용자: ").strip()
            if text.lower() in ('q', 'quit', 'exit', '종료'):
                break
            if not text:
                continue

            result = nlu.run(text)
            print(f"  → Intent: {result['intent']} ({result['confidence']:.1%})")
            print(f"  → 응답: {result['response']}")
            print()
