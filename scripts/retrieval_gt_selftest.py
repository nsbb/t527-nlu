#!/usr/bin/env python3
"""Retrieval 자기복제 테스트 — GT 219개에 대해:
- Leave-one-out: 한 GT 제외하고 나머지로 pool 구성 → 제외한 GT를 쿼리로
- 같은 발화 자체 검색 시 top_sim=1.0 자연스럽지만
- 비슷한 GT가 있으면 그걸 매칭하는지 (의미적 관련성 측정)
"""
import os, sys, json, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from retrieval_hybrid import SentenceEncoder, GTRetriever
from preprocess import preprocess
import torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceEncoder(device)

    gt_known = json.load(open('data/gt_known_scenarios.json'))
    gt_unknown = json.load(open('data/gt_unknown_scenarios.json'))
    gt_all = gt_known + gt_unknown
    print(f"GT total: {len(gt_all)}\n")

    # 전체 임베딩 한번만 계산
    utts = [preprocess(d['utterance']) for d in gt_all]
    embs = encoder.encode(utts)
    print(f"임베딩 [{embs.shape}] 완료\n")

    # Leave-one-out: 각 GT에 대해 자기 제외하고 top-1 찾기
    n = len(gt_all)
    fn_match = exec_match = dir_match = all_match = 0
    self_sim_hist = []
    non_self_top_sim = []
    examples = []

    for i in range(n):
        query = embs[i]
        # Self 제외한 top-1 찾기
        sims = embs @ query
        sims[i] = -1  # 자기 자신 제외
        top_idx = int(np.argmax(sims))
        top_sim = float(sims[top_idx])
        non_self_top_sim.append(top_sim)

        exp = gt_all[i]['labels']
        got = gt_all[top_idx]['labels']

        f = exp.get('fn') == got.get('fn')
        e = exp.get('exec_type') == got.get('exec_type')
        d = exp.get('param_direction') == got.get('param_direction')

        if f: fn_match += 1
        if e: exec_match += 1
        if d: dir_match += 1
        if f and e and d:
            all_match += 1
        else:
            if len(examples) < 15:
                diffs = []
                if not f: diffs.append(f"fn:{exp['fn']}→{got['fn']}")
                if not e: diffs.append(f"exec:{exp['exec_type']}→{got['exec_type']}")
                if not d: diffs.append(f"dir:{exp['param_direction']}→{got['param_direction']}")
                examples.append({
                    'query': gt_all[i]['utterance'],
                    'matched': gt_all[top_idx]['utterance'],
                    'sim': top_sim,
                    'diffs': diffs,
                })

    print(f"=== Leave-One-Out Retrieval (GT 자기복제 테스트) ===")
    print(f"  fn:    {fn_match}/{n} = {fn_match/n*100:.1f}%")
    print(f"  exec:  {exec_match}/{n} = {exec_match/n*100:.1f}%")
    print(f"  dir:   {dir_match}/{n} = {dir_match/n*100:.1f}%")
    print(f"  combo: {all_match}/{n} = {all_match/n*100:.1f}%")

    sims_arr = np.array(non_self_top_sim)
    print(f"\n=== Top-1 유사도 분포 ===")
    print(f"  mean: {sims_arr.mean():.3f}")
    print(f"  min:  {sims_arr.min():.3f}")
    print(f"  max:  {sims_arr.max():.3f}")
    print(f"  >0.9:  {(sims_arr > 0.9).sum()}/{n}")
    print(f"  >0.85: {(sims_arr > 0.85).sum()}/{n}")
    print(f"  >0.8:  {(sims_arr > 0.8).sum()}/{n}")
    print(f"  >0.7:  {(sims_arr > 0.7).sum()}/{n}")

    print(f"\n=== 실패 예시 (최대 15개) ===")
    for e in examples:
        print(f"  query:   \"{e['query']}\"")
        print(f"  matched: \"{e['matched']}\" (sim={e['sim']:.3f})")
        print(f"  diffs:   {', '.join(e['diffs'])}")
        print()


if __name__ == '__main__':
    main()
