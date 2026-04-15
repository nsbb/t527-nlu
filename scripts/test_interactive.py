#!/usr/bin/env python3
"""NLU 대화형 테스트
사용법:
    python3 scripts/test_interactive.py              # v7 clean (기본)
    python3 scripts/test_interactive.py v9            # v9
    python3 scripts/test_interactive.py "거실 불 켜줘" # 단일 테스트
"""
import torch, torch.nn.functional as F, json, re, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer

# 모델 선택
version = 'v7_clean'
single_text = None
for arg in sys.argv[1:]:
    if arg in ('v7', 'v7_clean'):
        version = 'v7_clean'
    elif arg == 'v9':
        version = 'v9'
    else:
        single_text = arg

CKPT = {
    'v7_clean': 'checkpoints/cnn_multihead_v7_clean.pt',
    'v9': 'checkpoints/cnn_multihead_v9.pt',
}

# 로드
print(f"모델 로딩 ({version})...")
sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
pw = sbert.embeddings.word_embeddings.weight.detach()
del sbert

model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
ckpt = torch.load(CKPT[version], map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['state'])
model.eval()

tok = AutoTokenizer.from_pretrained('tokenizer/')
print(f"로딩 완료 — epoch {ckpt['epoch']}, combo {ckpt['combo']:.1f}%\n")

# Room 추출
ROOMS = {'거실':'living','주방':'kitchen','부엌':'kitchen','안방':'bedroom_main',
         '큰방':'bedroom_main','작은방':'bedroom_sub','침실':'bedroom_sub',
         '아이방':'bedroom_sub','전체':'all','전부':'all'}

# 미지원 액션
UNSUPPORTED = {
    'medical_query': ['예약','상담','진료','증상','처방'],
    'traffic_query': ['택시','대리','카풀','렌트'],
    'market_query': ['추천','매수','매도','예측','수익률','계좌','마진','떨어질까'],
    'news_query': ['구독','매일 아침','브리핑 해줘','예약 브리핑'],
    'system_meta': ['비밀번호 바꿔','비밀번호 추천','비번 변경','이름 바꾸'],
    'home_info': ['골프장 예약','수영장 예약','자동 절전'],
    'vent_control': ['필터 주문','고쳐','AS','수리'],
    'ac_control': ['고장','AS','수리'],
}

def predict(text):
    text = ''.join(c if c.isprintable() or c == ' ' else ' ' for c in text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return None, None, 0

    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        logits = model(tk['input_ids'])

    preds = {h: HEAD_I2L[h][logits[h].argmax(1).item()] for h in HEAD_NAMES}
    probs = F.softmax(logits['fn'], dim=1)[0]
    conf = probs.max().item()

    # Top 3
    top3_idx = probs.topk(3).indices.tolist()
    top3 = [(HEAD_I2L['fn'][i], probs[i].item()) for i in top3_idx]

    return preds, top3, conf


def format_result(text, preds, top3, conf):
    fn = preds['fn']
    ex = preds['exec_type']
    dr = preds['param_direction']
    pt = preds['param_type']
    jd = preds['judge']

    # Room
    room = 'none'
    for kw, rm in ROOMS.items():
        if kw in text:
            room = rm
            break

    # Value
    value = None
    m = re.search(r'(\d+)\s*도', text)
    if m: value = f"{m.group(1)}도"
    m2 = re.search(r'(\d+)\s*분', text)
    if m2: value = f"{m2.group(1)}분"

    # 미지원 체크
    unsupported = False
    for kw in UNSUPPORTED.get(fn, []):
        if kw in text:
            unsupported = True
            break

    # 출력
    lines = []
    lines.append(f"  fn:        {fn} ({conf*100:.0f}%)")
    lines.append(f"  exec:      {ex}")
    if dr != 'none': lines.append(f"  direction: {dr}")
    if pt != 'none': lines.append(f"  param:     {pt}")
    if jd != 'none': lines.append(f"  judge:     {jd}")
    if room != 'none': lines.append(f"  room:      {room}")
    if value: lines.append(f"  value:     {value}")

    # Top 3
    lines.append(f"  top3:      {', '.join(f'{n}({p*100:.0f}%)' for n, p in top3)}")

    # 응답
    if fn == 'unknown':
        lines.append(f"  → 서버로 전송")
    elif unsupported:
        lines.append(f"  → 죄송합니다. 해당 기능은 지원하지 않습니다.")
    elif ex == 'clarify':
        lines.append(f"  → 어떤 공간의 기기를 제어할지 말씀해주세요.")
    else:
        lines.append(f"  → [정상 처리]")

    return '\n'.join(lines)


if single_text:
    preds, top3, conf = predict(single_text)
    print(f"입력: {single_text}")
    print(format_result(single_text, preds, top3, conf))
else:
    print("=== NLU 대화형 테스트 ===")
    print(f"모델: {version}")
    print("종료: q / quit / 종료\n")

    while True:
        try:
            text = input("사용자: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ('q', 'quit', 'exit', '종료'):
            break
        if not text:
            continue

        preds, top3, conf = predict(text)
        if preds:
            print(format_result(text, preds, top3, conf))
        print()
