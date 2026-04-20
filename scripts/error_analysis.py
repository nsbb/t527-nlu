#!/usr/bin/env python3
"""Error Analysis Tool — 자동 오류 카테고리 분류 및 패턴 탐지

사용법:
    python3 scripts/error_analysis.py                         # v46 기본
    python3 scripts/error_analysis.py ensemble                # ensemble ONNX
    python3 scripts/error_analysis.py v28                     # v28 모델
    python3 scripts/error_analysis.py v28 --export errors.csv # CSV 출력
"""
import os, sys, json, re, argparse
from collections import Counter, defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES, CNNMultiHead
from preprocess import preprocess


def load_model(model_name='v46'):
    """Load model by name (v28, v34, v46, ensemble)"""
    if model_name == 'ensemble':
        import onnxruntime as ort
        sess = ort.InferenceSession('checkpoints/nlu_v28_v46_ensemble.onnx',
                                    providers=['CPUExecutionProvider'])
        return sess, 'onnx'
    else:
        import torch
        from transformers import AutoModel
        sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        pw = sbert.embeddings.word_embeddings.weight.detach()
        del sbert
        model = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
        ckpt_path = f'checkpoints/cnn_multihead_{model_name}.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state'])
        model.eval()
        return model, 'torch'


def predict(model, model_type, tok, text, use_preprocess=True):
    if use_preprocess:
        text = preprocess(text)
    tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='np' if model_type == 'onnx' else 'pt')

    if model_type == 'onnx':
        outs = model.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
        logits = [outs[0][0], outs[1][0], outs[2][0], outs[3][0], outs[4][0]]
        fn_probs = np.exp(logits[0] - logits[0].max()); fn_probs /= fn_probs.sum()
        conf = float(fn_probs.max())
        return {
            'fn': HEAD_I2L['fn'][logits[0].argmax()],
            'exec_type': HEAD_I2L['exec_type'][logits[1].argmax()],
            'param_direction': HEAD_I2L['param_direction'][logits[2].argmax()],
            'param_type': HEAD_I2L['param_type'][logits[3].argmax()],
            'judge': HEAD_I2L['judge'][logits[4].argmax()],
        }, conf, text
    else:
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            l = model(tk['input_ids'])
        conf = F.softmax(l['fn'], dim=1).max().item()
        return {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}, conf, text


def categorize_error(utt, expected, predicted):
    """오류 카테고리 자동 분류"""
    categories = []

    # fn 오류
    if predicted['fn'] != expected['fn']:
        if predicted['fn'] == 'unknown':
            categories.append('fn:known_to_unknown')
        elif expected['fn'] == 'unknown':
            categories.append('fn:unknown_to_known')
        else:
            # fn 쌍
            pair = f"fn:{expected['fn']}→{predicted['fn']}"
            categories.append(pair)

        # 특수 패턴
        if expected['fn'] == 'schedule_manage' and predicted['fn'] == 'system_meta':
            categories.append('pattern:schedule_vs_system')
        if expected['fn'] == 'home_info' and predicted['fn'] == 'system_meta':
            categories.append('pattern:home_vs_system')

    # exec 오류
    if predicted['exec_type'] != expected['exec']:
        if predicted['exec_type'] == 'control_then_confirm' and expected['exec'] == 'query_then_respond':
            categories.append('exec:query_as_control')
        elif predicted['exec_type'] == 'control_then_confirm' and expected['exec'] == 'direct_respond':
            categories.append('exec:direct_as_control')
        else:
            categories.append(f"exec:{expected['exec']}→{predicted['exec_type']}")

    # dir 오류
    if predicted['param_direction'] != expected['dir']:
        e_dir = expected['dir']; p_dir = predicted['param_direction']
        if (e_dir, p_dir) in [('down', 'up'), ('up', 'down')]:
            categories.append('dir:opposite_direction')
        elif (e_dir, p_dir) in [('on', 'off'), ('off', 'on')]:
            categories.append('dir:opposite_onoff')
        elif (e_dir, p_dir) in [('open', 'close'), ('close', 'open')]:
            categories.append('dir:opposite_openclose')
        elif e_dir != 'none' and p_dir == 'none':
            categories.append('dir:missing')
        elif e_dir == 'none' and p_dir != 'none':
            categories.append('dir:spurious')
        else:
            categories.append(f"dir:{e_dir}→{p_dir}")

    # 길이
    n_words = len(utt.split())
    if n_words <= 2:
        categories.append('length:very_short')
    elif n_words <= 4:
        categories.append('length:short')

    # STT 관련
    stt_indicators = ['쥬', '줘요', '쥬세요', '어때', '뭐지', '뭘까', '해야돼']
    if any(ind in utt for ind in stt_indicators):
        categories.append('style:colloquial')

    # 숫자 포함
    if re.search(r'\d+', utt):
        categories.append('has_number')
    if re.search(r'(이십|삼십|사십|스물|서른|마흔)', utt):
        categories.append('has_korean_number')

    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?', default='v46', help='Model name (v28, v46, ensemble)')
    parser.add_argument('--export', help='Export errors to CSV file')
    parser.add_argument('--suite', default='data/test_suite.json', help='Test suite file')
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Suite: {args.suite}")

    model, mtype = load_model(args.model)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('tokenizer/')

    suite = json.load(open(args.suite))
    print(f"Cases: {len(suite)}")

    errors = []
    category_counts = Counter()
    confidence_buckets = defaultdict(lambda: {'correct': 0, 'incorrect': 0})

    for t in suite:
        preds, conf, clean_text = predict(model, mtype, tok, t['utterance'])

        f = preds['fn'] == t['fn']
        e = preds['exec_type'] == t['exec']
        d = preds['param_direction'] == t['dir']
        correct = f and e and d

        # Confidence bucket
        bucket = 'high' if conf >= 0.9 else 'mid' if conf >= 0.7 else 'low'
        confidence_buckets[bucket]['correct' if correct else 'incorrect'] += 1

        if not correct:
            expected = {'fn': t['fn'], 'exec': t['exec'], 'dir': t['dir']}
            cats = categorize_error(t['utterance'], expected, preds)
            for c in cats:
                category_counts[c] += 1

            errors.append({
                'utterance': t['utterance'],
                'preprocessed': clean_text,
                'fn_exp': t['fn'], 'fn_pred': preds['fn'],
                'exec_exp': t['exec'], 'exec_pred': preds['exec_type'],
                'dir_exp': t['dir'], 'dir_pred': preds['param_direction'],
                'conf': conf,
                'categories': cats,
            })

    n = len(suite)
    print(f"\n=== 전체 결과 ===")
    print(f"  정답: {n - len(errors)}/{n} = {(n-len(errors))/n*100:.1f}%")
    print(f"  오류: {len(errors)}")

    print(f"\n=== Confidence 분포 ===")
    for bucket in ['high', 'mid', 'low']:
        b = confidence_buckets[bucket]
        total = b['correct'] + b['incorrect']
        if total > 0:
            acc = b['correct'] / total * 100
            print(f"  {bucket} ({total}): 정확도 {acc:.1f}%, {b['incorrect']}개 오류")

    print(f"\n=== Top 15 오류 카테고리 ===")
    for cat, count in category_counts.most_common(15):
        print(f"  [{count:3d}x] {cat}")

    # High-conf errors (가장 위험한 오류)
    high_conf_errors = [e for e in errors if e['conf'] >= 0.9]
    print(f"\n=== High-confidence 오류 ({len(high_conf_errors)}건, 가장 위험) ===")
    for e in sorted(high_conf_errors, key=lambda x: -x['conf'])[:10]:
        det = []
        if e['fn_exp'] != e['fn_pred']: det.append(f"fn:{e['fn_exp']}→{e['fn_pred']}")
        if e['exec_exp'] != e['exec_pred']: det.append(f"exec:{e['exec_exp']}→{e['exec_pred']}")
        if e['dir_exp'] != e['dir_pred']: det.append(f"dir:{e['dir_exp']}→{e['dir_pred']}")
        print(f"  (conf {e['conf']:.2f}) \"{e['utterance']}\" {' '.join(det)}")

    # CSV export
    if args.export:
        import csv
        with open(args.export, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['utterance', 'preprocessed', 'fn_exp', 'fn_pred',
                           'exec_exp', 'exec_pred', 'dir_exp', 'dir_pred',
                           'conf', 'categories'])
            for e in errors:
                writer.writerow([e['utterance'], e['preprocessed'],
                               e['fn_exp'], e['fn_pred'],
                               e['exec_exp'], e['exec_pred'],
                               e['dir_exp'], e['dir_pred'],
                               f"{e['conf']:.3f}",
                               ';'.join(e['categories'])])
        print(f"\n오류 CSV 출력: {args.export}")


if __name__ == '__main__':
    main()
