#!/usr/bin/env python3
"""Retrieval-Augmented Hybrid NLU
ROADMAP Option B — 라벨 품질 의존 감소

입력 → ko-sbert embedding → GT pool cosine similarity 검색
  ├─ max_sim > HIGH_TH: GT 라벨 직접 사용 (★ 핵심)
  ├─ LOW_TH < sim < HIGH_TH: Multi-head 분류 fallback
  └─ sim < LOW_TH: unknown (서버로)

장점:
- GT pool 라벨이 맞으면 100% 정확 (재학습 불필요)
- OOD 거부 자연스러움 (유사 GT 없음 → unknown)
- 새 시나리오 추가는 pool에만 넣으면 됨
- 설명 가능 ("이 GT와 유사해서 분류함")

실험:
- HIGH_TH (0.80, 0.85, 0.90) 비교
- 커버리지 vs 정확도 트레이드오프
- Test Suite + KoELECTRA 모두 평가
"""
import os, sys, json, re, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from transformers import AutoModel, AutoTokenizer
from model_cnn_multihead import HEAD_I2L, HEAD_NAMES, CNNMultiHead
from preprocess import preprocess


# ============================================================
# Sentence embedding (mean pooling on ko-sbert)
# ============================================================
class SentenceEncoder:
    def __init__(self, device='cpu'):
        self.tok = AutoTokenizer.from_pretrained('jhgan/ko-sbert-sts')
        self.model = AutoModel.from_pretrained('jhgan/ko-sbert-sts').to(device).eval()
        self.device = device

    def encode(self, texts, batch_size=64):
        """배치로 임베딩. [N, 768] numpy 반환"""
        if isinstance(texts, str):
            texts = [texts]
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=128, return_tensors='pt').to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            # Mean pooling with attention mask
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)  # L2 normalize for cosine
            embs.append(emb.cpu().numpy())
        return np.concatenate(embs, axis=0)


# ============================================================
# Retrieval index
# ============================================================
class GTRetriever:
    def __init__(self, encoder, gt_data):
        """
        gt_data: list of {utterance, labels: {fn, exec_type, param_direction, param_type, judge}}
        """
        self.gt_data = gt_data
        utts = [preprocess(d['utterance']) for d in gt_data]
        print(f"GT 임베딩 계산 중 ({len(utts)}개)...")
        t0 = time.time()
        self.embs = encoder.encode(utts)  # [N, 768], L2 normalized
        print(f"  완료 ({time.time()-t0:.1f}s, {self.embs.shape})")

    def search(self, query_emb, k=5):
        """query_emb: [768], returns top-k (score, gt_idx)"""
        # cosine = dot product (since L2 normalized)
        sims = self.embs @ query_emb  # [N]
        top_idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), int(i)) for i in top_idx]

    def label_of(self, idx):
        return self.gt_data[idx]['labels']


# ============================================================
# Hybrid predictor
# ============================================================
class HybridPredictor:
    def __init__(self, encoder, retriever, model_v46=None, tok_v46=None,
                 high_threshold=0.85, low_threshold=0.5):
        self.encoder = encoder
        self.retriever = retriever
        self.model = model_v46
        self.tok = tok_v46
        self.high = high_threshold
        self.low = low_threshold

    def predict(self, text, return_detail=False):
        clean = preprocess(text)
        q = self.encoder.encode([clean])[0]
        topk = self.retriever.search(q, k=3)
        top_sim, top_idx = topk[0]

        if top_sim >= self.high:
            # Retrieval — GT 라벨 직접 사용
            labels = self.retriever.label_of(top_idx)
            mode = 'retrieval'
        elif top_sim >= self.low:
            # Fallback — Multi-head 분류
            if self.model is not None:
                labels = self._predict_model(clean)
                mode = 'model'
            else:
                labels = self.retriever.label_of(top_idx)
                mode = 'retrieval_mid'
        else:
            # Unknown
            labels = {'fn': 'unknown', 'exec_type': 'direct_respond',
                      'param_direction': 'none', 'param_type': 'none', 'judge': 'none'}
            mode = 'unknown'

        if return_detail:
            return {
                'labels': labels, 'mode': mode, 'top_sim': top_sim,
                'top_gt': self.retriever.gt_data[top_idx]['utterance'],
                'topk': topk,
            }
        return labels

    def _predict_model(self, text):
        tk = self.tok(text, padding='max_length', truncation=True,
                      max_length=32, return_tensors='pt')
        with torch.no_grad():
            l = self.model(tk['input_ids'])
        preds = {h: HEAD_I2L[h][l[h].argmax(1).item()] for h in HEAD_NAMES}
        if preds['param_direction'] in ('open', 'close', 'stop'):
            preds['param_type'] = 'none'
        if preds['judge'] != 'none':
            preds['param_type'] = 'none'
        if preds['exec_type'] in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
            preds['param_type'] = 'none'
        return preds


# ============================================================
# Evaluation
# ============================================================
def eval_on_suite(predictor, suite, label_key='fn'):
    """Test Suite 또는 GT scenarios 평가"""
    fn_ok = exec_ok = dir_ok = all_ok = 0
    mode_count = {'retrieval': 0, 'model': 0, 'retrieval_mid': 0, 'unknown': 0}
    mode_correct = {'retrieval': 0, 'model': 0, 'retrieval_mid': 0, 'unknown': 0}

    for t in suite:
        res = predictor.predict(t['utterance'], return_detail=True)
        mode_count[res['mode']] += 1
        p = res['labels']

        # Test Suite는 flat {fn, exec, dir}, GT는 nested {labels: {...}}
        if 'labels' in t:
            exp = t['labels']
            exp_fn = exp.get('fn')
            exp_exec = exp.get('exec_type')
            exp_dir = exp.get('param_direction')
        else:
            exp_fn = t.get('fn')
            exp_exec = t.get('exec')
            exp_dir = t.get('dir')

        f = p['fn'] == exp_fn
        e = p['exec_type'] == exp_exec
        d = p['param_direction'] == exp_dir

        if f: fn_ok += 1
        if e: exec_ok += 1
        if d: dir_ok += 1
        if f and e and d:
            all_ok += 1
            mode_correct[res['mode']] += 1

    n = len(suite)
    return {
        'fn': fn_ok / n * 100,
        'exec': exec_ok / n * 100,
        'dir': dir_ok / n * 100,
        'combo': all_ok / n * 100,
        'total': n,
        'mode_count': mode_count,
        'mode_correct': mode_correct,
    }


def eval_koelectra_fn(predictor, ke):
    """KoELECTRA는 fn만 평가"""
    fn_ok = 0
    mode_count = {'retrieval': 0, 'model': 0, 'retrieval_mid': 0, 'unknown': 0}
    for d in ke:
        res = predictor.predict(d['utterance'], return_detail=True)
        mode_count[res['mode']] += 1
        if res['labels']['fn'] == d['labels']['fn']:
            fn_ok += 1
    n = len(ke)
    return {'fn': fn_ok / n * 100, 'total': n, 'mode_count': mode_count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--high', type=float, default=0.85, help='Retrieval threshold')
    parser.add_argument('--low', type=float, default=0.5, help='Unknown threshold')
    parser.add_argument('--no-model', action='store_true', help='Retrieval only (no fallback)')
    parser.add_argument('--threshold-sweep', action='store_true', help='Sweep thresholds')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # GT pool — known + unknown (unknown도 retrieval에 포함, 미지원 발화 거부)
    gt_known = json.load(open('data/gt_known_scenarios.json'))
    gt_unknown = json.load(open('data/gt_unknown_scenarios.json'))
    gt_all = gt_known + gt_unknown
    print(f"GT pool: {len(gt_known)} known + {len(gt_unknown)} unknown = {len(gt_all)}\n")

    # Sentence encoder
    encoder = SentenceEncoder(device)
    retriever = GTRetriever(encoder, gt_all)

    # Fallback model (v46)
    model_v46, tok_v46 = None, None
    if not args.no_model:
        print("\nv46 로드 중...")
        sbert_full = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
        pw = sbert_full.embeddings.word_embeddings.weight.detach()
        del sbert_full
        tok_v46 = AutoTokenizer.from_pretrained('tokenizer/')
        model_v46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.15)
        ckpt = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
        model_v46.load_state_dict(ckpt['state']); model_v46.eval()
        print("  ✓ v46 로드됨")

    # Eval datasets
    test_suite = json.load(open('data/test_suite.json'))
    ke_val = json.load(open('data/koelectra_converted_val.json'))

    if args.threshold_sweep:
        print(f"\n{'='*70}")
        print(f"  Threshold Sweep — HIGH (retrieval → direct) 탐색")
        print('='*70)

        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        print(f"\n{'HIGH':>6} {'TS fn':>7} {'TS exec':>8} {'TS dir':>8} {'TS combo':>10} "
              f"{'KE fn':>7} {'TS retr%':>9} {'TS model%':>10}")
        print('-' * 75)

        for ht in thresholds:
            predictor = HybridPredictor(encoder, retriever, model_v46, tok_v46,
                                         high_threshold=ht, low_threshold=args.low)
            ts_res = eval_on_suite(predictor, test_suite)
            ke_res = eval_koelectra_fn(predictor, ke_val)

            mc = ts_res['mode_count']
            retr_pct = (mc['retrieval'] + mc['retrieval_mid']) / ts_res['total'] * 100
            model_pct = mc['model'] / ts_res['total'] * 100

            print(f"{ht:>6.2f} {ts_res['fn']:>7.2f} {ts_res['exec']:>8.2f} "
                  f"{ts_res['dir']:>8.2f} {ts_res['combo']:>10.2f} "
                  f"{ke_res['fn']:>7.2f} {retr_pct:>9.1f}% {model_pct:>10.1f}%")
    else:
        # Single config
        predictor = HybridPredictor(encoder, retriever, model_v46, tok_v46,
                                     high_threshold=args.high, low_threshold=args.low)

        print(f"\n{'='*70}")
        print(f"  Retrieval Hybrid 평가 (HIGH={args.high}, LOW={args.low})")
        print('='*70)

        print(f"\n[1/2] Test Suite ({len(test_suite)}개)...")
        t0 = time.time()
        ts_res = eval_on_suite(predictor, test_suite)
        print(f"  완료 ({time.time()-t0:.1f}s)")
        print(f"  fn:    {ts_res['fn']:.2f}%")
        print(f"  exec:  {ts_res['exec']:.2f}%")
        print(f"  dir:   {ts_res['dir']:.2f}%")
        print(f"  combo: {ts_res['combo']:.2f}%")
        print(f"  모드별: {ts_res['mode_count']}")
        print(f"  정답률 per 모드: retrieval={ts_res['mode_correct']['retrieval']}/{ts_res['mode_count']['retrieval']}, "
              f"model={ts_res['mode_correct']['model']}/{ts_res['mode_count']['model']}, "
              f"retrieval_mid={ts_res['mode_correct']['retrieval_mid']}/{ts_res['mode_count']['retrieval_mid']}, "
              f"unknown={ts_res['mode_correct']['unknown']}/{ts_res['mode_count']['unknown']}")

        print(f"\n[2/2] KoELECTRA ({len(ke_val)}개)...")
        t0 = time.time()
        ke_res = eval_koelectra_fn(predictor, ke_val)
        print(f"  완료 ({time.time()-t0:.1f}s)")
        print(f"  fn:    {ke_res['fn']:.2f}%")
        print(f"  모드별: {ke_res['mode_count']}")


if __name__ == '__main__':
    main()
