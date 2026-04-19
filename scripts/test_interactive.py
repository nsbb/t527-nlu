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
version = 'v25'
single_text = None
for arg in sys.argv[1:]:
    if arg in ('v7', 'v7_clean'):
        version = 'v7_clean'
    elif arg == 'v9':
        version = 'v9'
    elif arg == 'v21':
        version = 'v25'
    else:
        single_text = arg

CKPT = {
    'v7_clean': 'checkpoints/cnn_multihead_v7_clean.pt',
    'v9': 'checkpoints/cnn_multihead_v9.pt',
    'v25': 'checkpoints/cnn_multihead_v28.pt',
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
    from preprocess import preprocess
    original = text
    text = preprocess(text)
    if not text:
        return None, None, None

    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        logits = model(tk['input_ids'])

    preds = {}
    all_probs = {}
    for h in HEAD_NAMES:
        probs = F.softmax(logits[h], dim=1)[0]
        pred_idx = probs.argmax().item()
        preds[h] = HEAD_I2L[h][pred_idx]
        # top 3 per head
        top_k = min(3, len(probs))
        top_idx = probs.topk(top_k).indices.tolist()
        all_probs[h] = [(HEAD_I2L[h][i], probs[i].item()) for i in top_idx]

    # param_type 규칙 보정
    if preds['param_direction'] in ('open', 'close', 'stop'):
        preds['param_type'] = 'none'
    if preds['judge'] != 'none':
        preds['param_type'] = 'none'
    if preds['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
        preds['param_type'] = 'none'

    preprocessed = text if text != original else None
    return preds, all_probs, preprocessed


def format_result(text, preds, all_probs):
    fn = preds['fn']

    # Room (rule)
    room = 'none'
    for kw, rm in ROOMS.items():
        if kw in text:
            room = rm
            break

    # Value (rule)
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

    # 출력 — 5 head 전부
    lines = []
    lines.append(f"  ┌─ Multi-Head 결과 ─────────────────────────────")
    for h in HEAD_NAMES:
        top = all_probs[h]
        pred_val = preds[h]
        conf = top[0][1]
        top_str = ' | '.join(f'{n}({p*100:.0f}%)' for n, p in top)
        lines.append(f"  │ {h:17s}: {pred_val:25s} [{top_str}]")

    lines.append(f"  ├─ Rule 추출 ──────────────────────────────────")
    if room != 'none': lines.append(f"  │ room:              {room}")
    if value:          lines.append(f"  │ value:             {value}")
    if room == 'none' and not value:
                       lines.append(f"  │ (없음)")

    lines.append(f"  ├─ 판정 ──────────────────────────────────────")
    if fn == 'unknown':
        lines.append(f"  │ → 🔀 서버로 전송")
    elif unsupported:
        lines.append(f"  │ → ❌ 미지원: 해당 기능은 지원하지 않습니다.")
    elif preds['exec_type'] == 'clarify':
        lines.append(f"  │ → ❓ 어떤 공간의 기기를 제어할지 말씀해주세요.")
    else:
        lines.append(f"  │ → ✅ 정상 처리")
    lines.append(f"  └────────────────────────────────────────────")

    return '\n'.join(lines)


if single_text:
    preds, all_probs, preprocessed = predict(single_text)
    print(f"입력: {single_text}")
    if preprocessed:
        print(f"전처리: {preprocessed}")
    print(format_result(single_text, preds, all_probs))
else:
    print("=== NLU 대화형 테스트 (5-Head Multi-Head + STT 전처리) ===")
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

        preds, all_probs, preprocessed = predict(text)
        if preprocessed:
            print(f"  [전처리] {text} → {preprocessed}")
        if preds:
            print(format_result(text, preds, all_probs))
        print()
