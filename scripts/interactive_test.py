#!/usr/bin/env python3
"""대화형 테스트 — 발화 입력하면 3개 모델 결과 비교

사용법:
    python3 scripts/interactive_test.py         # 대화형
    python3 scripts/interactive_test.py "거실 에어컨 켜줘"  # 단일
    python3 scripts/interactive_test.py --batch file.txt    # 배치 (한 줄 1 발화)
"""
import os, sys, json, re, argparse, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import HEAD_I2L, HEAD_NAMES, CNNMultiHead
from preprocess import preprocess
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class ModelComparison:
    def __init__(self, models_to_load=('v28', 'v46', 'ensemble')):
        sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        self.pw = sbert.embeddings.word_embeddings.weight.detach()
        del sbert
        self.tok = AutoTokenizer.from_pretrained('tokenizer/')

        self.models = {}
        for name in models_to_load:
            if name == 'ensemble':
                try:
                    import onnxruntime as ort
                    self.models['ensemble'] = {
                        'type': 'onnx',
                        'session': ort.InferenceSession(
                            'checkpoints/nlu_v28_v46_ensemble.onnx',
                            providers=['CPUExecutionProvider'])
                    }
                except Exception as e:
                    print(f"앙상블 로드 실패: {e}")
                continue
            try:
                m = CNNMultiHead(self.pw, d_model=256, max_len=32, dropout=0.15)
                ckpt = torch.load(f'checkpoints/cnn_multihead_{name}.pt',
                                  map_location='cpu', weights_only=False)
                m.load_state_dict(ckpt['state'])
                m.eval()
                self.models[name] = {'type': 'torch', 'model': m}
            except Exception as e:
                print(f"{name} 로드 실패: {e}")

    def predict(self, model_name, text):
        m = self.models[model_name]
        tk = self.tok(text, padding='max_length', truncation=True,
                      max_length=32, return_tensors='np' if m['type'] == 'onnx' else 'pt')
        if m['type'] == 'onnx':
            outs = m['session'].run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
            logits = {
                'fn': outs[0][0], 'exec_type': outs[1][0], 'param_direction': outs[2][0],
                'param_type': outs[3][0], 'judge': outs[4][0],
            }
            fn_probs = np.exp(logits['fn'] - logits['fn'].max())
            fn_probs /= fn_probs.sum()
            conf = float(fn_probs.max())
            preds = {h: HEAD_I2L[h][logits[h].argmax()] for h in HEAD_NAMES}
        else:
            with torch.no_grad():
                l = m['model'](tk['input_ids'])
            conf = F.softmax(l['fn'], dim=1).max().item()
            preds = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}

        # 후처리 규칙
        if preds['param_direction'] in ('open', 'close', 'stop'):
            preds['param_type'] = 'none'
        if preds['judge'] != 'none':
            preds['param_type'] = 'none'
        if preds['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
            preds['param_type'] = 'none'

        # Confidence fallback
        conf_fallback = False
        if conf < 0.5 and preds['fn'] != 'unknown':
            preds['fn'] = 'unknown'
            preds['exec_type'] = 'direct_respond'
            conf_fallback = True

        return preds, conf, conf_fallback

    def extract_slots(self, text):
        """룰 기반 slot 추출"""
        ROOMS = {'거실': 'living', '주방': 'kitchen', '부엌': 'kitchen',
                 '안방': 'bedroom_main', '큰방': 'bedroom_main',
                 '작은방': 'bedroom_sub', '침실': 'bedroom_sub', '아이방': 'bedroom_sub',
                 '전체': 'all', '전부': 'all', '모든': 'all'}
        room = 'none'
        for kw, r in ROOMS.items():
            if kw in text: room = r; break

        value = None
        for pat, name in [(r'(\d+)\s*도', 'temperature'), (r'(\d+)\s*분', 'time'),
                          (r'(\d+)\s*%', 'percent'), (r'(\d+)\s*단계', 'level')]:
            m = re.search(pat, text)
            if m:
                value = (name, m.group(1)); break
        return room, value


def print_result(text, comp):
    """발화 결과 포맷팅"""
    clean = preprocess(text)

    print(f"\n{'═'*70}")
    print(f"  입력: \"{text}\"")
    if clean != text:
        print(f"  전처리: \"{clean}\"")
    room, value = comp.extract_slots(clean)
    if room != 'none' or value:
        print(f"  룰: room={room}" + (f" value={value[1]}{value[0]}" if value else ""))
    print('─'*70)

    for name in comp.models:
        start = time.time()
        preds, conf, fallback = comp.predict(name, clean)
        elapsed = (time.time() - start) * 1000

        fb = " [⚠conf<0.5]" if fallback else ""
        extras = []
        if preds['param_type'] != 'none':
            extras.append(f"param={preds['param_type']}")
        if preds['judge'] != 'none':
            extras.append(f"judge={preds['judge']}")
        extra_str = f" ({' '.join(extras)})" if extras else ""

        print(f"  {name:10s} fn={preds['fn']:18s} exec={preds['exec_type']:22s} "
              f"dir={preds['param_direction']:6s} conf={conf:.2f}{fb}{extra_str} [{elapsed:.1f}ms]")

    # 불일치 체크
    preds_list = [comp.predict(name, clean)[0] for name in comp.models]
    for h in ['fn', 'exec_type', 'param_direction']:
        vals = set(p[h] for p in preds_list)
        if len(vals) > 1:
            print(f"  ⚠ {h} 불일치: {vals}")


def interactive_mode(comp):
    print("\n🤖 NLU 대화형 테스트")
    print("   모델 로드 완료: " + ", ".join(comp.models.keys()))
    print("   종료: q, quit, exit")
    print("   명령:\n     .t — 최근 발화 다시 예측\n     .s TEXT — Silent 모드 (ensemble만)\n")

    last_text = None
    while True:
        try:
            text = input("사용자: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ('q', 'quit', 'exit', '종료'):
            break
        if not text:
            continue
        if text == '.t' and last_text:
            text = last_text
        elif text.startswith('.s '):
            # Silent — ensemble only
            t = text[3:]
            clean = preprocess(t)
            preds, conf, _ = comp.predict('ensemble', clean)
            print(f"  fn={preds['fn']} exec={preds['exec_type']} dir={preds['param_direction']} conf={conf:.2f}")
            continue

        last_text = text
        print_result(text, comp)


def batch_mode(comp, file_path):
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"\n📝 배치 테스트: {len(lines)}개 발화\n")
    for text in lines:
        print_result(text, comp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('text', nargs='?', help='단일 발화')
    parser.add_argument('--batch', help='파일에서 배치 테스트 (한 줄 1 발화)')
    parser.add_argument('--models', default='v28,v46,ensemble',
                        help='비교할 모델 (쉼표 구분): v28, v46, ensemble, v34, v68')
    args = parser.parse_args()

    models = tuple(args.models.split(','))
    print(f"모델 로딩... ({', '.join(models)})")
    comp = ModelComparison(models)

    if args.batch:
        batch_mode(comp, args.batch)
    elif args.text:
        print_result(args.text, comp)
    else:
        interactive_mode(comp)


if __name__ == '__main__':
    main()
