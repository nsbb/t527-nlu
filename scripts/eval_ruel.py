#!/usr/bin/env python3
"""르엘 219개 시나리오 + test set 평가"""
import torch, json, csv, os, sys, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


def load_model():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert

    model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
    ckpt = torch.load('checkpoints/cnn_multihead_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('tokenizer/')
    print(f"Model loaded — epoch {ckpt['epoch']}, combo {ckpt['combo']:.1f}%")
    return model, tokenizer


def predict(model, tokenizer, text):
    text = ''.join(c if c.isprintable() or c == ' ' else ' ' for c in text)
    text = re.sub(r'\s+', ' ', text).strip()
    tk = tokenizer(text, padding='max_length', truncation=True, max_length=32, return_tensors='pt')
    with torch.no_grad():
        logits = model(tk['input_ids'])
    return {h: HEAD_I2L[h][logits[h].argmax(1).item()] for h in HEAD_NAMES}


def eval_test_set(model, tokenizer):
    """val/test set 평가"""
    with open('data/test_final.json', encoding='utf-8') as f:
        test_data = json.load(f)

    head_correct = {h: 0 for h in HEAD_NAMES}
    all_correct = 0
    total = len(test_data)

    for d in test_data:
        preds = predict(model, tokenizer, d['utterance'])
        match_all = True
        for h in HEAD_NAMES:
            gt = d['labels'].get(h, 'none')
            if preds[h] == gt:
                head_correct[h] += 1
            else:
                match_all = False
        if match_all:
            all_correct += 1

    print(f"\n=== Test Set ({total}개) ===")
    for h in HEAD_NAMES:
        acc = head_correct[h] / total * 100
        print(f"  {h:20s}: {acc:.1f}%")
    print(f"  {'combo':20s}: {all_correct/total*100:.1f}%")


def eval_ruel(model, tokenizer):
    """르엘 시나리오 평가 (Excel GT 기준)"""
    import openpyxl
    wb = openpyxl.load_workbook('semantic_action_parser_head_combos.xlsx', read_only=True)
    ws = wb.active

    correct_fn = 0
    correct_exec = 0
    correct_both = 0
    total = 0
    errors = []

    for row in ws.iter_rows(min_row=2, values_only=True):
        fn_gt = row[5]
        exec_gt = row[6]
        utt = row[4]
        if not fn_gt or not utt:
            continue

        preds = predict(model, tokenizer, utt.strip())
        total += 1

        fn_ok = preds['fn'] == fn_gt
        exec_ok = preds['exec_type'] == exec_gt

        if fn_ok:
            correct_fn += 1
        if exec_ok:
            correct_exec += 1
        if fn_ok and exec_ok:
            correct_both += 1

        if not fn_ok or not exec_ok:
            errors.append({
                'utterance': utt.strip(),
                'gt_fn': fn_gt, 'pred_fn': preds['fn'],
                'gt_exec': exec_gt, 'pred_exec': preds['exec_type'],
                'fn_ok': '✓' if fn_ok else '✗',
                'exec_ok': '✓' if exec_ok else '✗',
            })

    print(f"\n=== 르엘 Excel GT ({total}개) ===")
    print(f"  fn:        {correct_fn}/{total} = {correct_fn/total*100:.1f}%")
    print(f"  exec_type: {correct_exec}/{total} = {correct_exec/total*100:.1f}%")
    print(f"  fn+exec:   {correct_both}/{total} = {correct_both/total*100:.1f}%")

    if errors:
        print(f"\n  오류 {len(errors)}개:")
        # fn별 오류 집계
        fn_errors = Counter()
        for e in errors:
            if e['fn_ok'] == '✗':
                fn_errors[f"{e['gt_fn']}→{e['pred_fn']}"] += 1

        print(f"\n  fn 오류 패턴:")
        for pattern, cnt in fn_errors.most_common(15):
            print(f"    {pattern}: {cnt}건")

        print(f"\n  상세 (처음 20개):")
        for e in errors[:20]:
            print(f"    [{e['fn_ok']}{e['exec_ok']}] \"{e['utterance'][:30]}\" "
                  f"fn:{e['gt_fn']}→{e['pred_fn']} exec:{e['gt_exec']}→{e['pred_exec']}")


def eval_edge_cases(model, tokenizer):
    """간접 표현 / 어미 변형 / STT 오류 테스트"""
    cases = [
        # 간접 표현
        ("찜통이야", "ac_control", "control_then_confirm"),
        ("좀 춥다", "heat_control", "control_then_confirm"),
        ("어두워", "light_control", "control_then_confirm"),
        ("시베리아야", "heat_control", "control_then_confirm"),
        ("동굴이야", "light_control", "control_then_confirm"),
        ("공기가 탁해", "vent_control", "control_then_confirm"),
        ("나 나간다", "security_mode", "control_then_confirm"),
        ("답답해", "vent_control", "control_then_confirm"),
        # 어미 변형
        ("지금 온도 어떠니", "heat_control", "query_then_respond"),
        ("에어컨 켜줄래", "ac_control", "control_then_confirm"),
        ("불 좀 꺼봐", "light_control", "control_then_confirm"),
        # STT 오류
        ("남방 커줘", "heat_control", "control_then_confirm"),
        ("에어콘 틀어", "ac_control", "control_then_confirm"),
        ("엘레베이터 불러", "elevator_call", "control_then_confirm"),
        # 판단형
        ("세차해도 돼?", "weather_query", "query_then_judge"),
        ("뭐 입고 나가?", "weather_query", "query_then_judge"),
        ("창문 열어도 괜찮아?", "weather_query", "query_then_judge"),
        # 직접 응답
        ("너 뭐 할 수 있어?", "system_meta", "direct_respond"),
        ("누가 만들었어?", "system_meta", "direct_respond"),
        ("지금 몇 시야?", "home_info", "direct_respond"),
        # 복합
        ("거실 에어컨 제습 23도", "ac_control", "control_then_confirm"),
        ("안방 난방 25도로 맞춰줘", "heat_control", "control_then_confirm"),
    ]

    print(f"\n=== Edge Cases ({len(cases)}개) ===")
    correct = 0
    for utt, gt_fn, gt_exec in cases:
        preds = predict(model, tokenizer, utt)
        fn_ok = preds['fn'] == gt_fn
        exec_ok = preds['exec_type'] == gt_exec
        both = fn_ok and exec_ok
        if both:
            correct += 1
        mark = '✓' if both else '✗'
        detail = ''
        if not fn_ok:
            detail += f" fn:{gt_fn}→{preds['fn']}"
        if not exec_ok:
            detail += f" exec:{gt_exec}→{preds['exec_type']}"
        print(f"  [{mark}] \"{utt}\" dir={preds['param_direction']}{detail}")

    print(f"\n  Edge case 정확도: {correct}/{len(cases)} = {correct/len(cases)*100:.1f}%")


if __name__ == '__main__':
    model, tokenizer = load_model()
    eval_test_set(model, tokenizer)
    eval_ruel(model, tokenizer)
    eval_edge_cases(model, tokenizer)
