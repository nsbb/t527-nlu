#!/usr/bin/env python3
"""전수 라벨 검토 — 학습+테스트 데이터의 모든 라벨 오류 탐지

접근 방식 (다중 검증):
1. 규칙 기반: 발화에서 명백한 키워드 → 라벨 불일치 검출
2. 모델 기반: 앙상블 예측 vs 라벨 비교 (high-conf mismatch)
3. Cross-validation: 학습 데이터와 테스트 데이터의 동일 발화 라벨 충돌

출력: 모든 suspect을 카테고리별로 정리한 보고서 + JSON
"""
import os, sys, json, re, argparse
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES
from preprocess import preprocess
from transformers import AutoTokenizer
import onnxruntime as ort


# ============================================================
# 규칙 기반 라벨 오류 패턴 (강한 시그널)
# ============================================================
RULES = [
    # (패턴, 기대 필드 변경, 설명)
    # === Direction 규칙 ===
    # "닫아/닫힘/잠금" → dir=close
    (re.compile(r'(닫아|닫어|닫자|닫아줘|닫아봐|닫힘)'), {'param_direction': 'close'},
     "닫는 동작은 dir=close"),
    (re.compile(r'(잠가|잠궈|잠금|잠그|잠궜|잠기)'), {'param_direction': 'close'},
     "잠그는 동작은 dir=close"),

    # "열어/연다" → dir=open
    (re.compile(r'(열어|열어봐|여세요|여는|열기|열림|열려)'), {'param_direction': 'open'},
     "여는 동작은 dir=open"),

    # "꺼/끄기/중지" → dir=off (단, "도어락 꺼" 같은 모호한 케이스 제외)
    (re.compile(r'^(?!도어).*(\S+)(꺼줘|꺼!|꺼야|끄기|끄자|끄세요|꺼도)'), {'param_direction': 'off'},
     "끄는 동작은 dir=off"),

    # "켜/틀어/가동" → dir=on
    (re.compile(r'(켜줘|켜!|켜야|켜세요|켜기|켜자|틀어|가동|돌려)'), {'param_direction': 'on'},
     "켜는 동작은 dir=on"),

    # "올려/높여/강하게" → dir=up
    (re.compile(r'(올려|높여|강하게|더 세게|세게|높게)'), {'param_direction': 'up'},
     "올리는 동작은 dir=up"),

    # "내려/낮춰/약하게" → dir=down
    (re.compile(r'(내려|낮춰|약하게|더 낮게|낮게|어둡게|조용히)'), {'param_direction': 'down'},
     "내리는 동작은 dir=down"),

    # "등록/설정/맞춰" → dir=set (보통 값 동반)
    (re.compile(r'(등록해|등록해줘|등록해주세요|맞춰줘|설정해|예약)'), {'param_direction': 'set'},
     "등록/설정은 dir=set"),

    # === Exec 규칙 ===
    # "호출/불러" → control (query가 아님)
    (re.compile(r'(호출|불러|불러줘|불러줄|오라고)'), {'exec_type': 'control_then_confirm'},
     "호출/불러는 control (query 아님)"),

    # "얼마|몇 도|몇 시" → query_then_respond
    (re.compile(r'(얼마|몇 도|몇도|몇 시|몇시|몇 분|얼만큼|어떤가)'), {'exec_type': 'query_then_respond'},
     "얼마/몇~는 query"),

    # === Fn 규칙 ===
    # 확실한 fn 매핑
    (re.compile(r'(날씨|기온|비|눈|우산|강수|폭염|한파)'), {'fn_strong_hint': 'weather_query'},
     "날씨/기온은 weather_query"),
    (re.compile(r'(뉴스|헤드라인|속보)'), {'fn_strong_hint': 'news_query'},
     "뉴스는 news_query"),
    (re.compile(r'(가스 ?(밸브|잠|열|켜|끄))'), {'fn_strong_hint': 'gas_control'},
     "가스 밸브는 gas_control"),
    (re.compile(r'(승강기|엘리베이터|엘베|리프트)'), {'fn_strong_hint': 'elevator_call'},
     "승강기는 elevator_call"),
    (re.compile(r'(커튼|커턴|블라인드|전동커튼)'), {'fn_strong_hint': 'curtain_control'},
     "커튼은 curtain_control"),
]

# Exec 예외: control_then_confirm으로 강제되면 안 되는 query 발화
QUERY_INDICATORS = ['얼마', '몇 도', '몇도', '몇 시', '몇시', '어때', '있나', '있어?', '어떤가']


def check_rules(utt, labels):
    """Rule 기반 오류 검출"""
    suspects = []
    fn_hint = None

    for pattern, fix, reason in RULES:
        m = pattern.search(utt)
        if not m: continue

        # fn_strong_hint 수집 (모델 오류와 달리 "라벨 오류" 판단에 더 보수적)
        if 'fn_strong_hint' in fix:
            fn_hint = fix['fn_strong_hint']
            continue

        # Query indicator가 있으면 "control" 강제 규칙은 스킵
        if fix.get('exec_type') == 'control_then_confirm' and any(q in utt for q in QUERY_INDICATORS):
            continue

        for k, v in fix.items():
            cur = labels.get(k) if k == 'fn' else labels.get(k)
            if cur and cur != v:
                suspects.append({
                    'field': k, 'current': cur, 'expected': v, 'reason': reason,
                })

    return suspects, fn_hint


def audit_dataset(data, name, use_model=None, tok=None):
    """단일 데이터셋 전수 검토"""
    print(f"\n{'═'*70}")
    print(f"  데이터셋 검사: {name} ({len(data)}개)")
    print('═'*70)

    # 라벨 key 정규화 (test_suite는 'exec', 'dir' / train은 labels.{exec_type,param_direction})
    def get_labels(d):
        if 'labels' in d:
            return {
                'fn': d['labels'].get('fn'),
                'exec_type': d['labels'].get('exec_type'),
                'param_direction': d['labels'].get('param_direction'),
                'param_type': d['labels'].get('param_type'),
                'judge': d['labels'].get('judge'),
            }
        else:  # test_suite flat
            return {
                'fn': d.get('fn'),
                'exec_type': d.get('exec'),
                'param_direction': d.get('dir'),
            }

    all_suspects = []

    # Rule-based scan
    for idx, d in enumerate(data):
        utt = d.get('utterance', '')
        labels = get_labels(d)
        rule_suspects, fn_hint = check_rules(utt, labels)
        for s in rule_suspects:
            all_suspects.append({
                'idx': idx, 'utterance': utt, 'source': 'rule',
                **s, 'model_conf': None, 'model_fn': None,
            })

    # Model-based scan (apply only for train, test_suite too heavy)
    if use_model is not None and tok is not None:
        print(f"  모델 기반 예측 수행 중...")
        for idx, d in enumerate(data):
            utt = d.get('utterance', '')
            labels = get_labels(d)
            tk = tok(utt, padding='max_length', truncation=True, max_length=32, return_tensors='np')
            outs = use_model.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})

            pred_fn = HEAD_I2L['fn'][outs[0][0].argmax()]
            pred_exec = HEAD_I2L['exec_type'][outs[1][0].argmax()]
            pred_dir = HEAD_I2L['param_direction'][outs[2][0].argmax()]

            fn_probs = np.exp(outs[0][0] - outs[0][0].max()); fn_probs /= fn_probs.sum()
            conf = float(fn_probs.max())

            # High-conf model-label mismatch
            if conf >= 0.95:
                if pred_fn != labels['fn']:
                    all_suspects.append({
                        'idx': idx, 'utterance': utt, 'source': 'model_highconf',
                        'field': 'fn', 'current': labels['fn'], 'expected': pred_fn,
                        'reason': f'모델 conf={conf:.2f} 다른 예측',
                        'model_conf': conf, 'model_fn': pred_fn,
                    })
                elif pred_exec != labels['exec_type']:
                    all_suspects.append({
                        'idx': idx, 'utterance': utt, 'source': 'model_highconf',
                        'field': 'exec_type', 'current': labels['exec_type'], 'expected': pred_exec,
                        'reason': f'모델 conf={conf:.2f}',
                        'model_conf': conf, 'model_fn': pred_fn,
                    })
                elif pred_dir != labels['param_direction']:
                    all_suspects.append({
                        'idx': idx, 'utterance': utt, 'source': 'model_highconf',
                        'field': 'param_direction', 'current': labels['param_direction'],
                        'expected': pred_dir, 'reason': f'모델 conf={conf:.2f}',
                        'model_conf': conf, 'model_fn': pred_fn,
                    })

            if (idx + 1) % 5000 == 0:
                print(f"    {idx+1}/{len(data)} 완료")

    return all_suspects


def cross_check(train_data, test_data):
    """학습+테스트 데이터에서 동일 발화의 라벨 충돌 검출"""
    print(f"\n{'═'*70}")
    print(f"  Cross-check: 학습 ↔ 테스트 동일 발화 라벨 충돌")
    print('═'*70)

    # Index train by utterance
    train_by_utt = defaultdict(list)
    for idx, d in enumerate(train_data):
        train_by_utt[d['utterance']].append((idx, d['labels']))

    conflicts = []
    for ti, t in enumerate(test_data):
        utt = t['utterance']
        if utt not in train_by_utt: continue
        test_lbl = {'fn': t['fn'], 'exec_type': t['exec'], 'param_direction': t['dir']}
        for train_idx, train_lbl in train_by_utt[utt]:
            for k in ['fn', 'exec_type', 'param_direction']:
                if test_lbl.get(k) != train_lbl.get(k):
                    conflicts.append({
                        'utterance': utt, 'field': k,
                        'test_value': test_lbl.get(k), 'train_value': train_lbl.get(k),
                        'test_idx': ti, 'train_idx': train_idx,
                    })
                    break  # 같은 발화, 하나 차이만 기록

    print(f"  충돌 발견: {len(conflicts)}건")
    for c in conflicts[:15]:
        print(f"    \"{c['utterance']}\" {c['field']}: test={c['test_value']} vs train={c['train_value']}")
    if len(conflicts) > 15:
        print(f"    ... 그 외 {len(conflicts)-15}건")
    return conflicts


def summarize(suspects, name):
    """Suspects 통계 요약"""
    print(f"\n--- {name} 요약 ---")
    print(f"  총 suspect: {len(suspects)}")

    # Source별
    by_source = Counter(s['source'] for s in suspects)
    for src, cnt in by_source.most_common():
        print(f"    {src}: {cnt}")

    # Field별
    by_field = Counter(s['field'] for s in suspects)
    print(f"  Field별:")
    for f, cnt in by_field.most_common():
        print(f"    {f}: {cnt}")

    # 흔한 패턴
    by_pair = Counter((s['field'], s['current'], s['expected']) for s in suspects)
    print(f"  Top 10 패턴 (field: current → expected):")
    for (f, c, e), cnt in by_pair.most_common(10):
        print(f"    [{cnt:3d}x] {f}: {c} → {e}")

    # 중복 제거 unique utterance
    unique_utts = set(s['utterance'] for s in suspects)
    print(f"  Unique utterances: {len(unique_utts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/train_final_v43.json', help='학습 데이터')
    parser.add_argument('--test', default='data/test_suite.json', help='테스트 데이터')
    parser.add_argument('--use-model', action='store_true', help='모델 기반 검증 추가')
    parser.add_argument('--no-train', action='store_true', help='학습 데이터 스킵 (빠름)')
    parser.add_argument('--output', default='data/label_audit_report.json')
    parser.add_argument('--top', type=int, default=50, help='보고서에 표시할 top N')
    args = parser.parse_args()

    print("🔍 라벨 전수 검토")
    print(f"   학습: {args.train}")
    print(f"   테스트: {args.test}")
    print(f"   모델 검증: {'ON' if args.use_model else 'OFF'}")

    # Load data
    with open(args.train) as f: train_data = json.load(f)
    with open(args.test) as f: test_data = json.load(f)

    # Setup model if needed
    model = None; tok = None
    if args.use_model:
        tok = AutoTokenizer.from_pretrained('tokenizer/')
        model = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                     providers=['CPUExecutionProvider'])

    # Audit
    test_suspects = audit_dataset(test_data, 'Test Suite', model, tok)
    summarize(test_suspects, 'Test Suite')

    train_suspects = []
    if not args.no_train:
        train_suspects = audit_dataset(train_data, 'Train Data', model, tok)
        summarize(train_suspects, 'Train Data')

    conflicts = cross_check(train_data, test_data)

    # 상세 보고서
    print(f"\n{'═'*70}")
    print(f"  Top {args.top} Test Suite Suspects (rule-based)")
    print('═'*70)
    rule_ts = [s for s in test_suspects if s['source'] == 'rule']
    for s in rule_ts[:args.top]:
        print(f"  [{s['idx']}] \"{s['utterance']}\"")
        print(f"    {s['field']}: {s['current']} → {s['expected']} ({s['reason']})")

    # Save JSON report
    report = {
        'train': args.train, 'test': args.test,
        'train_size': len(train_data), 'test_size': len(test_data),
        'test_suspects': test_suspects,
        'train_suspects': train_suspects,
        'cross_conflicts': conflicts,
        'summary': {
            'test_total': len(test_suspects),
            'test_unique_utts': len(set(s['utterance'] for s in test_suspects)),
            'train_total': len(train_suspects),
            'train_unique_utts': len(set(s['utterance'] for s in train_suspects)),
            'cross_conflicts': len(conflicts),
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 상세 보고서: {args.output}")
    print(f"\n=== 최종 요약 ===")
    print(f"  Test Suspects: {len(test_suspects)} ({len(set(s['utterance'] for s in test_suspects))} unique)")
    if not args.no_train:
        print(f"  Train Suspects: {len(train_suspects)} ({len(set(s['utterance'] for s in train_suspects))} unique)")
    print(f"  Cross-conflicts: {len(conflicts)}")


if __name__ == '__main__':
    main()
