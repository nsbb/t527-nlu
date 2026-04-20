#!/usr/bin/env python3
"""Flat 94-intent vs Multi-head 비교
- Flat: cnn_4L_v18_best (94 intent, val 97.1%)
- Multi-head: v46, v28+v46 ensemble

같은 조건에서 비교:
1. In-domain 학습 안 본 조합 (compositional generalization) ← multi-head가 유리해야 함
2. Out-of-domain (unknown 처리) ← 둘 다 확인
3. GT 219개 (fn만 비교) ← 직접 비교
4. KoELECTRA 외부 1,536개 (진짜 일반화) ← 핵심 비교
"""
import os, sys, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from transformers import AutoModel, AutoTokenizer


# Flat 94-intent 모델 (v18 구조 — embedding 256d 직접 학습, proj 없음)
class FlatCNN(nn.Module):
    def __init__(self, vocab_size=32000, nc=94, ml=32, cd=256, ks=[3,5,7,3], dr=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cd, padding_idx=0)
        layers = []
        for k in ks:
            layers += [nn.Conv1d(cd, cd, k, padding=k//2), nn.ReLU(),
                       nn.BatchNorm1d(cd), nn.Dropout(dr)]
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(cd, nc)
        self.max_len = ml

    def forward(self, x):
        sl = min(x.shape[1], self.max_len)
        x = self.embedding(x[:, :sl].long()).permute(0, 2, 1)
        return self.fc(self.convs(x).mean(dim=2))


# 94 intent → 20 fn 매핑 (비교용)
INTENT_TO_FN = {
    'ac_exception': 'unknown', 'ac_mode': 'ac_control', 'ac_mode_noroom': 'ac_control',
    'ac_mode_schedule': 'ac_control', 'ac_off': 'ac_control', 'ac_on': 'ac_control',
    'ac_query': 'ac_control', 'ac_temp': 'ac_control', 'ac_wind': 'ac_control',
    'alarm_query': 'schedule_manage', 'alarm_set': 'schedule_manage',
    'car_history_delete': 'vehicle_manage', 'car_history_query': 'vehicle_manage',
    'community_query': 'home_info', 'complex_info': 'home_info',
    'curtain_close': 'curtain_control', 'curtain_open': 'curtain_control',
    'curtain_query': 'curtain_control', 'curtain_schedule': 'curtain_control',
    'curtain_stop': 'curtain_control',
    'door_open': 'door_control', 'doorlock_open': 'door_control', 'doorlock_query': 'door_control',
    'dust_query': 'weather_query',
    'elevator_call': 'elevator_call', 'elevator_query': 'elevator_call',
    'emergency': 'security_mode',
    'energy_alert_off': 'energy_query', 'energy_alert_on': 'energy_query',
    'energy_goal_set': 'energy_query', 'energy_usage_query': 'energy_query',
    'ev_charging': 'vehicle_manage',
    'fuel_exception': 'unknown', 'fuel_price_query': 'market_query',
    'fuel_station_search': 'market_query',
    'gas_close': 'gas_control', 'gas_query': 'gas_control',
    'heating_down': 'heat_control', 'heating_off': 'heat_control', 'heating_on': 'heat_control',
    'heating_query': 'heat_control', 'heating_schedule_cancel': 'heat_control',
    'heating_schedule_query': 'heat_control', 'heating_schedule_set': 'heat_control',
    'heating_up': 'heat_control',
    'home_status_query': 'home_info',
    'light_brighten': 'light_control', 'light_dim': 'light_control',
    'light_off': 'light_control', 'light_on': 'light_control',
    'light_query': 'light_control', 'light_schedule': 'light_control',
    'manual_capability': 'system_meta', 'manual_creator': 'system_meta',
    'manual_name': 'system_meta', 'manual_unsupported': 'unknown',
    'manual_usage': 'system_meta',
    'medical_exception': 'unknown', 'medical_hours': 'medical_query',
    'medical_search': 'medical_query',
    'news_exception': 'unknown', 'news_info': 'news_query', 'news_play': 'news_query',
    'notice_query': 'home_info', 'notification_query': 'home_info',
    'password_change': 'system_meta',
    'security_activate': 'security_mode', 'security_query': 'security_mode',
    'security_return_set': 'security_mode',
    'stock_exception': 'unknown', 'stock_index_query': 'market_query',
    'stock_price_query': 'market_query',
    'system_brightness_schedule': 'home_info', 'system_brightness_set': 'home_info',
    'system_exception': 'unknown', 'system_faq': 'system_meta',
    'system_volume_set': 'home_info',
    'time_query': 'home_info',
    'traffic_bus_query': 'traffic_query', 'traffic_exception': 'unknown',
    'traffic_route_query': 'traffic_query',
    'ventilation_exception': 'unknown', 'ventilation_mode': 'vent_control',
    'ventilation_off': 'vent_control', 'ventilation_on': 'vent_control',
    'ventilation_query': 'vent_control', 'ventilation_schedule_query': 'vent_control',
    'ventilation_schedule_set': 'vent_control',
    'visitor_parking_query': 'vehicle_manage', 'visitor_parking_register': 'vehicle_manage',
    'weather_activity': 'weather_query', 'weather_clothing': 'weather_query',
    'weather_exception': 'unknown', 'weather_info': 'weather_query',
}


def load_flat(tok):
    m = FlatCNN(vocab_size=32000, nc=94, cd=256, dr=0.1)
    ckpt = torch.load('checkpoints/archive/cnn_4L_v18_best.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['state'])
    m.eval()
    return m, ckpt['i2l'], ckpt['l2i']


def predict_flat(model, i2l, tok, text, conf_threshold=0.5):
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        logits = model(tk['input_ids'])
    probs = F.softmax(logits, dim=1)
    conf = probs.max().item()
    intent_id = logits.argmax(1).item()
    intent = i2l[intent_id]
    fn = INTENT_TO_FN.get(intent, 'unknown')

    # Confidence fallback
    if conf < conf_threshold:
        fn = 'unknown'
        intent = '<LOW_CONF>'
    return {'intent': intent, 'fn': fn, 'conf': conf}


def predict_multihead(model, tok, text):
    from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        l = model(tk['input_ids'])
    conf = F.softmax(l['fn'], dim=1).max().item()
    return {
        'fn': HEAD_I2L['fn'][l['fn'].argmax(1).item()],
        'exec': HEAD_I2L['exec_type'][l['exec_type'].argmax(1).item()],
        'dir': HEAD_I2L['param_direction'][l['param_direction'].argmax(1).item()],
        'conf': conf,
    }


def main():
    print("=" * 70)
    print("  Flat 94-intent (v18) vs Multi-head (v46) 비교")
    print("=" * 70)

    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    # Flat 모델
    flat, i2l, l2i = load_flat(tok)
    print(f"\n✓ Flat v18 로드 — 94 intents, val {0.971*100:.1f}%")

    # Multi-head 모델 (v46)
    from model_cnn_multihead import CNNMultiHead, HEAD_I2L, HEAD_NAMES
    mh = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    mh.load_state_dict(torch.load('checkpoints/cnn_multihead_v46.pt',
                                   map_location='cpu', weights_only=False)['state'])
    mh.eval()
    print(f"✓ Multi-head v46 로드 — 20 fn × 5 exec × 9 dir")

    # ============================================================
    # Test 1: In-domain 학습 안 본 조합
    # ============================================================
    print(f"\n{'─'*70}\n  Test 1: In-domain 학습 안 본 조합 (조합 일반화)\n{'─'*70}")
    unseen_cases = [
        ('안방 에어컨 제습 모드 25도', 'ac_control'),
        ('큰방 난방 2시간 예약 22도', 'heat_control'),
        ('아이방 커튼 반만 열어', 'curtain_control'),
        ('주방 환기 10분 후 꺼', 'vent_control'),
        ('거실 조명 은은하게 30퍼센트로', 'light_control'),
        ('모든 방 에어컨 시원하게 해줘', 'ac_control'),
    ]

    flat_ok = mh_ok = 0
    for t, exp_fn in unseen_cases:
        f = predict_flat(flat, i2l, tok, t)
        m = predict_multihead(mh, tok, t)
        f_mark = '✓' if f['fn'] == exp_fn else '✗'
        m_mark = '✓' if m['fn'] == exp_fn else '✗'
        if f['fn'] == exp_fn: flat_ok += 1
        if m['fn'] == exp_fn: mh_ok += 1
        print(f'  "{t}"')
        print(f'    기대 fn: {exp_fn}')
        print(f'    Flat  [{f_mark}] intent={f["intent"]:30s} → fn={f["fn"]:18s} conf={f["conf"]:.2f}')
        print(f'    MH    [{m_mark}] fn={m["fn"]:18s} conf={m["conf"]:.2f}')

    print(f'\n  결과: Flat {flat_ok}/{len(unseen_cases)}, Multi-head {mh_ok}/{len(unseen_cases)}')

    # ============================================================
    # Test 2: Out-of-domain
    # ============================================================
    print(f"\n{'─'*70}\n  Test 2: Out-of-domain (unknown으로 가야 함)\n{'─'*70}")
    ood_cases = [
        '트럼프가 누구야', '주식 사고싶어', '배달 시켜줘',
        '카카오톡 열어줘', '유튜브 켜줘',
    ]

    flat_unk = mh_unk = 0
    for t in ood_cases:
        f = predict_flat(flat, i2l, tok, t)
        m = predict_multihead(mh, tok, t)
        if f['fn'] == 'unknown': flat_unk += 1
        if m['fn'] == 'unknown': mh_unk += 1
        print(f'  "{t}"')
        print(f'    Flat  intent={f["intent"]:30s} → fn={f["fn"]:18s} conf={f["conf"]:.2f}')
        print(f'    MH    fn={m["fn"]:18s} conf={m["conf"]:.2f}')

    print(f'\n  Unknown 처리: Flat {flat_unk}/{len(ood_cases)}, Multi-head {mh_unk}/{len(ood_cases)}')

    # ============================================================
    # Test 3: GT 219 (fn만 비교)
    # ============================================================
    print(f"\n{'─'*70}\n  Test 3: GT 219개 (fn 비교)\n{'─'*70}")
    gt_known = json.load(open('data/gt_known_scenarios.json'))
    gt_unk = json.load(open('data/gt_unknown_scenarios.json'))
    all_gt = gt_known + gt_unk

    flat_ok = mh_ok = 0
    flat_only_wrong = mh_only_wrong = 0
    for g in all_gt:
        exp = g['labels']['fn']
        f = predict_flat(flat, i2l, tok, g['utterance'])
        m = predict_multihead(mh, tok, g['utterance'])
        f_right = f['fn'] == exp
        m_right = m['fn'] == exp
        if f_right: flat_ok += 1
        if m_right: mh_ok += 1
        if f_right and not m_right: flat_only_wrong += 1
        if m_right and not f_right: mh_only_wrong += 1

    n = len(all_gt)
    print(f'  Flat fn: {flat_ok}/{n} = {flat_ok/n*100:.1f}%')
    print(f'  MH fn:   {mh_ok}/{n} = {mh_ok/n*100:.1f}%')
    print(f'  Flat만 맞음: {flat_only_wrong}개 / MH만 맞음: {mh_only_wrong}개')

    # ============================================================
    # Test 4: KoELECTRA 1,536 (진짜 일반화)
    # ============================================================
    print(f"\n{'─'*70}\n  Test 4: KoELECTRA 1,536개 (외부 일반화 ← 핵심)\n{'─'*70}")
    ke = json.load(open('data/koelectra_converted_val.json'))
    flat_ok = mh_ok = 0
    for d in ke:
        exp = d['labels']['fn']
        f = predict_flat(flat, i2l, tok, d['utterance'])
        m = predict_multihead(mh, tok, d['utterance'])
        if f['fn'] == exp: flat_ok += 1
        if m['fn'] == exp: mh_ok += 1
    n = len(ke)
    print(f'  Flat fn: {flat_ok}/{n} = {flat_ok/n*100:.1f}%')
    print(f'  MH fn:   {mh_ok}/{n} = {mh_ok/n*100:.1f}%')
    print(f'  차이:     {mh_ok - flat_ok:+d}개 ({(mh_ok - flat_ok)/n*100:+.1f}%p)')

    # ============================================================
    # 최종 요약
    # ============================================================
    print(f"\n{'='*70}\n  최종 비교 요약\n{'='*70}")
    print(f'''
  | 항목                  | Flat 94-intent | Multi-head v46 |
  |-----------------------|:---:|:---:|
  | 안 본 조합 일반화 (6개)  | {'...'} | {'...'} |
  | Out-of-domain 거부 (5개)| {'...'} | {'...'} |
  | GT 219 (fn)            | {flat_ok}/n% | {mh_ok}/n% |
  | KoELECTRA 1,536 (fn)    | ★ | ★ |
    ''')


if __name__ == '__main__':
    main()
