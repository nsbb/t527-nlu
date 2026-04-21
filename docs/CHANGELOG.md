# NLU Changelog — 버전별 업데이트 내역

## 요약표 (최신 → 과거)

| 버전 | 날짜 | 기법 | TS combo | KE fn | 결과 |
|------|------|------|:---:|:---:|------|
| iter5 | 04-21 PM | ONNX FP16 최적화 + 배포 문서 | 93.59% | 97.79% | 크기 50%↓, CPU 속도 37x↓ (GPU 전용) |
| v70 | 04-21 PM | 대규모 1,180 라벨 수정 후 retrain | 90.04% | 97.33% | 부분 수정으로 데이터 일관성 파괴 |
| iter3 | 04-21 PM | v28b (patch retrain) + rules | 92.74% | 75.33% | 밝게만 교정, 전체 -3.6%p regression |
| iter2 | 04-21 PM | 라벨 정제 v2 (24건) + 전략 재비교 | 93.59% | 97.79% | v28이 "밝게→down" 잘못 학습 확인 |
| iter1 | 04-21 PM | Retrieval Hybrid (GT pool cosine) | 91.49% | 94.01% | 모든 threshold 실패 (pool sparse) |
| v68 | 04-21 | 라벨 오류 수정 + 재학습 | 90.7% | 97.5% | 학습데이터 46건 수정 효과 제한적 |
| v67 | 04-21 | 통합 파이프라인 검증 | 93.8% | 97.9% | STT 내성 100%, preprocess 120개 |
| v66 | 04-21 | Ensemble ONNX 배포 | 94.3% | 97.8% | **배포용 단일 파일 ★** |
| v65 | 04-21 | KLUE-RoBERTa emb | 89.8% | 97.3% | 실패 (ko-sbert가 우세) |
| v64 | 04-21 | Unfreeze embeddings | 90.5% | 97.3% | 실패 (frozen이 더 나음) |
| v63 | 04-21 | Conformer 2L | 79.6% | 94.8% | 실패 (Attn < CNN at 24.5K) |
| v62 | 04-21 | Multi-seed 3×avg | 92.0% | 97.6% | 실패 (v46 seed lucky) |
| v61 | 04-21 | Warm-start v28 | 92.4% | 97.3% | 실패 (초기화 무관) |
| v60 | 04-21 | Model Soup | 93.3% | 97.8% | 최적 α=0 (=v46 그대로) |
| v59 | 04-21 | Head masking | 83.0% | 97.2% | 실패 (backbone 학습 저해) |
| v58 | 04-21 | Targeted augment | 91.4% | 97.5% | 실패 (분포 왜곡) |
| v57 | 04-21 | Wider d=384 | 90.2% | 97.2% | 실패 (데이터 대비 과다) |
| v56 | 04-21 | Two-stage | 91.2% | 96.9% | 실패 (fn/judge freeze 효과 없음) |
| v55 | 04-21 | KD from ensemble | 92.1% | 97.7% | 실패 (soft label 재현 불가) |
| v54 | 04-21 | Self-training R2 | — | — | ROI 없음 (2% 차이) |
| v53 | 04-20 | Focal+Mixup | 90.5% | 97.7% | 실패 |
| v52 | 04-20 | CutMix | 90.4% | 97.5% | 실패 |
| v51 | 04-20 | SupCon+Mixup | 91.4% | 97.6% | 미미 |
| v50 | 04-20 | R-Drop | 46.9% | — | 붕괴 |
| v49 | 04-20 | CNN+Attn | 91.5% | 97.5% | 미미 |
| v47 | 04-20 | Mixup+LS | 91.6% | 97.3% | 중복 |
| **v46** | **04-20** | **Mixup** | **93.3%** | **97.8%** | **단일 모델 최적 ★** |
| v45 | 04-20 | Label Smoothing | 90.5% | 97.4% | - |
| v38 | 04-19 | 6L CNN | 91.1% | 97.1% | 미미 |
| v34 | 04-19 | Pseudo-labeling | 93.3% | 96.8% | 핵심 기여 +21.3%p |
| v33 | 04-19 | KE 직접 병합 | 90.5% | - | 실패 |
| v29-v32 | 04-19 | 패치 fix | - | - | regression |
| **v28** | **04-19** | **Test Suite 3K 확정** | **96.3%** | **75.5%** | **GT 최적 ★** |

### 최적 구성
- **단일 모델**: v46 (TS 93.3%, KE 97.8%)
- **앙상블**: v28+v46 Strategy B (TS 94.3%, KE 97.8%)
- **Production**: `sap_inference_v2.py`가 v46 사용

### 핵심 교훈
1. TS↔KE 트레이드오프는 데이터 분포의 구조적 한계
2. v28↔v46 weight space 비볼록 (Model Soup 실패)
3. CNN 4L > Conformer 2L at 24.5K 샘플
4. 소규모 패치는 항상 regression 유발

---

## iter5 (2026-04-21 PM) — ONNX FP16 최적화 + 배포 준비 문서
- **크기**: 105MB → 52.5MB (50% 감소) ✓
- **정확도**: 100% match (TS 93.59%, KE 97.79% 그대로)
- **Latency**: 0.55ms → 20.4ms (CPU에서 37x 느림) ❌
- **결론**: CPU 배포는 FP32 유지, FP16은 GPU/NPU 전용
- Vocab 분석: 32000 중 12.2%만 실제 사용 (pruning 여지 탐색)
- 문서: `DEPLOYMENT_CHECKLIST.md` (10섹션), `FEEDBACK_SYSTEM_DESIGN.md`

## iter4 (2026-04-21 PM) — v70 대규모 라벨 수정 (1,180건)
- Train v34 데이터에 suspects A category 전부 수정 후 v46 recipe retrain
- 결과: TS 90.04%, KE 97.33%, balanced 93.62
- Baseline Ensemble B (95.66) 대비 **-2.0p 하락**
- 원인: 부분 수정이 pseudo-labeled 나머지와 충돌 → 데이터 일관성 파괴

## iter3 (2026-04-21 PM) — v28b (patch retrain) + Ensemble rules
- **v28b**: v28 warm-start + 189 추가 샘플 (밝게/어둡게/모드로) + 5ep
  - TS combo 92.74% (-3.6%p regression) ❌
  - "밝게" 교정은 성공
- **Post-proc rules on Ensemble**:
  - 밝게→up, 어둡게→down, 엘리베이터→control, 모드로→set
  - TS 93.53% (-0.06%p) — 엘리베이터 rule 부작용으로 미세 손실

## iter2 (2026-04-21 PM) — 라벨 정제 v2 + Ensemble 전략 재비교
- test_suite 추가 24건 라벨 오류 발견 + 수정 (밝게→up 4건 포함)
- v28은 "밝게→dir=down" 잘못 학습 확인 (v46은 정확)
- **Ensemble 전략 5가지 비교** (수정된 라벨 기준):
  - v28 alone: TS 95.53%, KE 75.59%
  - Avg (logit avg): TS 94.45%, KE 87.76%
  - **B (current): TS 93.59%, KE 97.79%** ← balanced 최고
  - v46 alone: TS 93.36%, KE 97.79%

## iter1 (2026-04-21 PM) — Retrieval Hybrid (ROADMAP Option B) 기각
- GT pool + ko-sbert cosine similarity로 라벨 의존 감소 시도
- 모든 threshold에서 v46 baseline 미달
- 원인: GT 219 self-retrieval combo 49.8% (pool sparse)

## v68 (2026-04-21) — 학습 데이터 라벨 오류 수정 + 재학습
- **TS 90.7%, KE 97.5%** (수정된 test_suite 기준) — v46 대비 소폭 하락
- 학습 데이터 `train_final_v43.json`의 46건 라벨 수정:
  - "커턴 닫아" dir: open → close (2건)
  - "난방꺼줘" dir: on → off
  - 주차/카드 "등록해줘" dir: none → set (16건)
  - 승강기/엘리베이터 "호출/불러" exec: query → control (26건)
- 정규식 패턴 매칭으로 일관성 있는 수정 자동화
- 결론: 수정된 케이스는 올바르게 학습하나, 소량 수정의 분포 변화로 다른 케이스 성능 영향
- 체크포인트: `checkpoints/cnn_multihead_v68.pt`
- 데이터: `data/train_final_v68.json`
- 스크립트: `scripts/train_v68_fixed_labels.py`

## v67 (2026-04-21) — 통합 파이프라인 검증 + Test Suite 확장
- **Test Suite: TS 93.8%, KE 97.9%, STT 내성 100%**
- preprocess.py STT 교정 사전 54→120개 확장
  - 에어콘/뉴슈/씨원하게/몇시/도어렉 등 실사용 STT 패턴
  - 오늘날씨어때/뭐입을까 등 띄어쓰기 없는 질문형
- Test Suite 3043→3109 (+66개): 숫자/구어체/질의/엣지/방향 다양성
- Error Analysis 자동화 도구 (`scripts/error_analysis.py`)
- Test Suite 라벨 오류 11건 발견/수정
- MODEL_CARD.md, DEPLOYMENT_GUIDE.md 작성

## v66 (2026-04-21) — Ensemble ONNX 배포용 단일 파일 ★
- **배포 성공**: `nlu_v28_v46_ensemble.onnx` (104.9MB)
- Test Suite: fn 98.0%, exec 98.2%, dir 97.8%, **combo 94.3%**
- KoELECTRA fn: **97.8%** (PyTorch 앙상블과 완전 동일)
- **추론 지연: 0.48ms/sample (CPU)**
- 내부 구조: Strategy B (fn=v46, exec/dir/param=v28, judge=v46)
- 스크립트: `scripts/export_ensemble_onnx.py`, `scripts/verify_ensemble_onnx.py`

## v65 (2026-04-21) — KLUE-RoBERTa embeddings 실험
- **TS 89.8%, KE 97.3%** — ko-sbert 대비 -3.5%p
- KoELECTRA 다운로드 실패 (네트워크)로 KLUE-RoBERTa로 대체
- 한국어 NLU benchmark 특화 모델임에도 ko-sbert만 못함
- 결론: ko-sbert-sts가 이 task에 이미 최적
- 체크포인트: `cnn_multihead_v65` (archived)

## v64 (2026-04-21) — Unfreeze embeddings (마지막 15 epochs)
- **TS 90.5%, KE 97.3%** — v46(93.3%) 대비 -2.8%p
- Phase 1 (1-25): frozen, lr=1e-3
- Phase 2 (26-40): unfreeze, emb lr=1e-5, rest lr=5e-4
- 결론: ko-sbert 임베딩이 이미 잘 정렬되어 있어 unfreeze가 노이즈 유발
- 체크포인트: `cnn_multihead_v64` (archived)

## v63 (2026-04-21) — Conformer backbone 실험
- **Test Suite combo 79.6%, KoELECTRA fn 94.8%** — CNN 대비 대폭 하락
- Conformer 2L (d=256, 4-head attention + depthwise conv 15) + 같은 5-head
- 2.1M trainable params (vs CNN 1.5M)
- 결론: 24.5K 데이터 규모에서 **Attention < CNN**
- 체크포인트: `cnn_multihead_v63` (archived)

## v62 (2026-04-21) — Multi-seed ensemble
- **3-model avg: TS 92.0%, KE 97.6%** — 단일 v46(93.3%)보다 하락
- v46(seed=42) + v62(seed=123) + v62(seed=999) logit averaging
- seed 123: TS 90.6%, seed 999: TS 90.4% — v46보다 약함
- 결론: v46 seed 42는 lucky, 약한 모델과 평균은 성능 하락
- 체크포인트: `cnn_multihead_v62_s123`, `cnn_multihead_v62_s999` (archived)

## v61 (2026-04-21) — Warm-start from v28
- **TS 92.4%, KE 97.3%** — v46(93.3%, 97.8%) 미달
- v28 가중치로 초기화 후 v43 데이터로 학습
- 결론: 초기화는 최종 성능에 무관 (같은 수렴점)
- 체크포인트: `cnn_multihead_v61` (archived)

## v60 (2026-04-21) — Model Soup (weight interpolation)
- **최적 α=0 (순수 v46, TS 93.3%)** — 돌파 실패
- v28/v46 간 α=0.3~0.6에서 성능 급락 (loss landscape 비볼록)
- head별 다른 α 시도도 실패 (TS 19.4%)
- 결론: prediction-level 앙상블만 유효
- 스크립트: `scripts/model_soup.py`

## v59 (2026-04-21) — Head-specific masking
- **TS 83.0%, KE 97.2%** — exec/dir 대폭 하락
- KoELECTRA 샘플에서 exec/dir loss 마스킹
- 결론: backbone이 전체 gradient 필요, masking 역효과
- 체크포인트: `cnn_multihead_v59` (archived)

## v58 (2026-04-21) — Targeted augmentation
- **TS 91.4%, KE 97.5%** — 기대와 달리 하락
- 313개 타겟 증강 (schedule/query/direction 패턴) × 3 oversample
- v46 오류 204개 패턴 분석 → 맞춤 augment
- 결론: 소규모 패치가 분포 왜곡, v29-v33과 동일 패턴
- 체크포인트: `cnn_multihead_v58` (archived)

## v57 (2026-04-21) — Wider model (d=384)
- **TS 90.2%, KE 97.2%** — 작은 데이터로 큰 모델 비효율
- d_model 256→384, 3.3M trainable
- Sample-weighted: GT 2.0x, KoELECTRA 1.0x
- 결론: 24.5K 샘플로는 d=256 이상 불필요
- 체크포인트: `cnn_multihead_v57` (archived)

## v56 (2026-04-21) — Two-stage fine-tuning
- **TS 91.2%, KE 96.9%** — fn 보존 실패
- Stage 1: v46 recipe, Stage 2: fn+judge head freeze + exec/dir 미세조정
- 결론: backbone이 변하면 frozen head도 영향, 양쪽 다 하락
- 체크포인트: `cnn_multihead_v56` (archived)

## v55 (2026-04-21) — Knowledge Distillation from ensemble
- **TS 92.1%, KE 97.7%** — ensemble 재현 실패
- Teacher: v28+v46 (Strategy B, temperature=3.0)
- Student: 같은 CNN 5-head, alpha 0.7→0.3 anneal
- SWA 추가: v55_swa TS 92.1%, KE 97.7% (동일)
- 결론: soft label만으로 v28의 exec/dir 전문성 재현 불가
- 체크포인트: `cnn_multihead_v55`, `cnn_multihead_v55_swa` (archived)

## v54 (2026-04-21) — Self-training Round 2
- **ROI 없음** — v28↔v46 pseudo label 차이 2% 뿐
- v46으로 KoELECTRA 재-pseudo-label 시도
- exec 일치율 98%, dir 일치율 97.1%
- 결론: 라벨 품질은 이미 수렴, 재라벨링 무의미
- 체크포인트: `cnn_multihead_v54`, `cnn_multihead_v54_swa` (archived)

## v53 (2026-04-20) — Focal Loss + Mixup
- **TS 90.5%, KE 97.7%** — v46보다 약간 하락
- Focal Loss (γ=2) + Mixup
- 어려운 샘플에 집중하나 TS 손실
- 결론: Mixup이 이미 일반화 효과, Focal 중복

## v52 (2026-04-20) — CutMix
- **TS 90.4%, KE 97.5%** — mixup보다 저조
- 같은 fn 내 토큰 30% 교체
- 결론: 텍스트에서 토큰 교체는 의미 파괴, Mixup(발화 교체) 우세

## v51 (2026-04-20) — Supervised Contrastive Loss
- **TS 91.4%, KE 97.6%** — v46 수준 미세 개선
- SupCon (α=0.1) + Mixup
- feature 공간에서 같은 fn 클러스터링
- 결론: 효과 미미, 복잡도만 증가

## v50 (2026-04-20) — R-Drop (실패)
- **val combo 60.9%** — 학습 붕괴
- α=0.5 + Mixup 조합 시 KL divergence가 CE 압도
- 해결: α=0.1~0.2 필요하나 forward 2회로 2배 느림, ROI 낮음

## v49 (2026-04-20) — CNN + Self-Attention hybrid
- **TS 91.5%, KE 97.5%** — 순수 CNN 대비 미미
- CNN 4L + 1-head Self-Attention
- 1.8M trainable
- 결론: 이 규모에서 Attention 추가 효과 없음
- 체크포인트: `cnn_attn_v49` (archived)

## v47 (2026-04-20) — Mixup + Label Smoothing
- **TS 91.6%, KE 97.3%** — 중복 효과
- Mixup + LS(0.1) 조합
- 결론: 두 기법 모두 regularization, 중복

## v46 (2026-04-20) — **단일 모델 최적점 (현재 권장)**
- **Test Suite combo 93.3%, KoELECTRA fn 97.8%**
- CNN 4L + Mixup (30% 확률, 같은 fn 내 발화 교체)
- pseudo-labeling (v28이 KoELECTRA exec/dir 예측) + mixup
- 일반화 최고: KoELECTRA fn 97.8% (실제 정확도 ~98.8%, 16건 KE 라벨 오류)
- ONNX: `nlu_v46_generalization.onnx` (99.7MB)
- 체크포인트: `cnn_multihead_v46.pt`

## v45 (2026-04-20) — Label Smoothing
- **TS 90.5%, KE 97.4%** — 확신 감소

## v38 (2026-04-19) — 6L CNN (실패)
- **TS 91.1%, KE 97.1%** — 4L과 유사
- 층 확장 효과 미미
- 체크포인트: `cnn_6L_v38` (archived)

## v34 (2026-04-19) — Pseudo-labeling 도입 (핵심 기여 +21.3%p)
- **TS 93.3%, KE 96.8%** — KoELECTRA fn 75.5%→96.8% 도약
- v28으로 KoELECTRA 13,540개 exec/dir 재라벨링
- 원본 fn 유지 + v28이 예측한 exec/dir
- 28,763개 병합 학습
- ONNX: `nlu_v34_production.onnx`
- 체크포인트: `cnn_multihead_v34.pt`

## v33 (2026-04-19) — KoELECTRA 직접 병합 (실패)
- **TS 90.5%** — v28(96.3%)에서 regression
- KoELECTRA 원본 exec/dir 그대로 병합 시 라벨 충돌
- 해결: pseudo-labeling으로 전환 (v34)

## v29~v32 (2026-04-19) — 패치 기반 fix (실패)
- 작은 fix 데이터셋 추가 시 기존 패턴 regression
- 교훈: 패치는 분포 왜곡, v28에서 안정화 확정

## v28 (2026-04-19) — Test Suite 3,043개 달성, GT 최적
- **Test Suite 3,043개: fn 100%, exec 98.2%, dir 97.8%, combo 96.3%**
- **KoELECTRA fn 75.5%** (외부 데이터 일반화는 약함)
- STT 전처리 사전 54개 + 한글 숫자 변환 (scripts/preprocess.py)
- confidence fallback (conf<0.5 → unknown)
- param_type 규칙 보정 (open/close→none, judge→none, query/direct→none)
- 알려진 실패 32개 문서화 (docs/KNOWN_FAILURES.md)
- ONNX: `nlu_v28_final.onnx` (99.7MB, 0.32ms CPU)
- v10~v28 반복 개선 (v29~v33 실험 → regression으로 v28 확정)
- 누적 fix 패턴 778개 → accumulated_fixes.json 보존
- 데이터: ~20,815개 | val combo: 94.7%

## v25
- **Test Suite 114개: fn 100%, dir 100%, combo 99.1%**
- v22~v25 반복: "공기가 탁해"→control, "오늘 어때"→weather, 감탄형→control
- param_type 규칙 보정 추가 (모델 대신 후처리로 해결)
- Test Suite 자동화: `data/test_suite.json` + `scripts/run_test_suite.py`
- 데이터: ~20,800개 | val combo: 94.4%

## v24
- **direction head 100%** 달성 (내려→down, 줄여→down, 커줘→on 수정)
- exec 수정: 답답해/시끄러워→control, 에너지→query
- Test Suite 114개 combo 98.2%

## v23
- 멈춰/중지→direction:stop 수정
- combo: 94.6%

## v21
- 5-head combo 테스트 88.9% (8/9)
- judge head: 반팔→clothing, 미세먼지→air_quality 수정
- param_type: 난방→temperature, 세차/창문→none 수정
- 데이터: ~20,600개 | val combo: 94.6%

## v19
- 구어체 수정: 조용히→볼륨, 들어왔어→귀가, 자려고→취침, 히팅→난방
- val combo: **94.9%** (최고치)

## v18
- **STT 오류 내성 강화**: 띄어쓰기 없는 발화, 자음/모음 오류, 음절 탈락
- 오늘날씨어때, 재습, 면시야, 뉴쓰, 근쳐 등 20개 패턴 100% 정확
- val combo: 94.5%

## v17
- 불안꺼져→light query, 환기안되나→vent query, 냄새→vent on, 환하게→brightness up
- val combo: 94.5%

## v16
- 실사용 92개 테스트 오류 18개 전수 수정
- 은은하게/데워줘/블라인드/방범모드/집비울건데/외출하기좋아 등
- val combo: 94.5%

## v15
- Hard edge case 수정: "어둡게 해줘"→light, "공기가 안 좋아"→vent, "그만"→off
- "불 켤 수 있어?"→light, "난방 25도로 맞춰"→heat 수정
- 수정 패턴 6/6 정확
- 219 GT: 99.1% | combo: 94.6% | false rejection: 0건
- 데이터: 20,289개

## v14
- 78개 종합 테스트 fn **100%** (78/78), 219 GT **100%**, combo **94.7%**
- v11~v14 반복 개선으로 오분류 패턴 전수 수정:
  - 간접표현: "더워"→ac, "시끄러워"→vent off, "외출할게"→security
  - STT오류: "커턴"→curtain, "남방"→heat
  - 경계: "밖에 더워?"→weather vs "더워"→ac 구분
  - direction: "환기 세게"→up, "에어컨 온도 내려"→down, "에어컨 23도"→set
  - exec: "송풍 해줘"→control(not clarify), "뉴스 틀어줘"→query(not unknown)
- 데이터: 20,247개

## v13
- 확장 테스트 55개 fn 98.2% (54/55)
- "더워"→ac, "시끄러워"→vent, "외출할게"→security, "커턴"→curtain 수정
- "오늘 어때?"→market 오분류 1건 남음
- 219 GT fn: 100%

## v12
- v11 남은 오류 2개 수정 ("환기 세게"→dir:up, "뭐 할 수 있어?"→system_meta)
- 핵심 33개 combo **100%** (33/33)
- combo: 94.7%

## v11
- v10 종합 96개 테스트에서 발견된 9개 fn오류 + 7개 exec오류 + 7개 dir오류 전수 수정
- 난방낮춰→heat, 에어컨온도→ac, 제습→ac, 미세먼지→weather, 우산→weather, 뉴스틀어→news, 볼륨→home, 사용법→system
- 핵심 43개 combo 95.3% (41/43)
- combo: 94.6%

## v10
- HA 영어 583개 한국어 번역 (Home-Assistant-Requests → 규칙 기반 번역)
  - door_control +97, curtain_control +127, vent_control +283 보강
- "불 꺼줘" → light_control 패턴 수정 (light_off 44개 + light_on_room 24개 추가)
- 기본 테스트 14/14 정확
- 데이터: 19,966개 | combo: 94.7%

## v9
- 부족 fn 집중 증강 (elevator 218→615, gas 362→702, door 372→797)
- 전용 표현 추가 (엘리베이터/가스/도어락 자연어 변형)
- 데이터: 22,724개 | 219 GT fn: 99.1% | combo: 94.9%
- 문제: "거실 불 꺼줘" → ac_control 오분류 (light_off 학습 데이터 0개)

## v8
- v7 edge case 수정 시도 (볼륨, 난방온도, 우산, 미세먼지 패턴 추가)
- 데이터: 20,861개 | combo: 94.4%
- 문제: false rejection 발생 ("환기 켜줘"→unknown, "뉴스 틀어줘"→unknown)
- 결론: v7보다 나빠져서 폐기

## v7 (clean)
- **GT 원본 제거 후 재학습** — 학습 데이터에 테스트 원본 포함 문제 수정
- 219 GT 원본을 train에서 빼고 증강 변형만으로 학습
- 데이터: 20,515개 | 219 GT fn: 100% | combo: 94.5% | unknown: 90% | false rej: 0건

## v7
- **고다양성 증강** — 시나리오당 avg 25 → avg 94 유니크
- 변형 규칙 대폭 강화 (동의어, 조사변형, 단어탈락, 복합변형)
- GT 증강 19,237개 + CNN 보충 + unknown 4,818개 = 24,351개
- 데이터: 24,351개 | 219 GT fn: 100% | combo: 94.5% | unknown: 95%

## v6
- targeted hard negatives 추가 (택시, 병원예약, 종목추천 등 미지원 패턴)
- unknown 거부 개선 시도
- 데이터: 10,342개 | 219 GT fn: 97.3% | combo: 86.5% | unknown: 50%
- 문제: hard negative 수 부족 (targeted 39개), unknown 거부율 낮음

## v5
- unknown weight를 0.3으로 낮춤 → 거부율 오히려 하락
- 데이터: 10,342개 | 219 GT fn: 96.8% | combo: 86.3% | unknown: 50%
- 결론: v4보다 나빠져서 폐기

## v4
- **219개 원본 GT 기준 평가 시작** (이전까지 142개 파생 파일 사용)
- unknown class 추가 (fn 19→20)
- unknown weight 0.5로 보정
- 데이터: 10,342개 | 219 GT fn: 96.3% | combo: 86.3% | unknown: 70% | false rej: 0건

## v3
- unknown class 최초 도입 (MASSIVE OOD + HA vacuum/media)
- GT 증강 시나리오당 100개 목표 (실제 avg 25 유니크)
- 데이터: ~10K | combo: 89.5% | unknown: 75%
- 문제: 증강 다양성 부족, clarify 라벨 버그

## v2
- MASSIVE 한국어 5,069개 추가 + judge/clarify 특수 데이터 1,026개
- weather 다운샘플링 (10,005→2,000)
- 데이터: 13,904개 | 르엘142 fn: 95.1% | combo: 89.1%
- 문제: fn|exec 51개 조합 중 35개 목표 미달, 1개 0건

## v1
- **CNN 5-head 최초 학습**
- 기존 CNN 94-intent 데이터 16,879개 → 19 fn multi-head 라벨 재매핑
- 부족 fn 규칙 기반 증강 8,045개
- 데이터: 13,904개 | val fn: 96.5% | combo: 90.8%
- 문제: fn 레벨 밸런싱만 함 (fn|exec 조합 밸런싱 안 함), weather 10,005개 지배

---

## 아키텍처 변경 이력

| 시점 | 아키텍처 | 변경 이유 |
|------|---------|----------|
| 초기 | CNN 94-intent flat | 첫 NLU |
| SAP | TextConformer 3L + 8-head | multi-head 분해 시도 |
| v1~ | **CNN 4L + 5-head** | Conformer→CNN (94%>86%), 8→5 head (불필요한 head 규칙으로) |

## 데이터 소스 이력

| 소스 | 도입 시점 | 용도 |
|------|---------|------|
| kochat_intent.csv | 초기 | 한국어 의도 분류 |
| Amazon MASSIVE ko-KR | v2 | IoT/weather/news/alarm + OOD |
| 르엘 GT 219개 증강 | v3 | 핵심 학습 데이터 |
| Home-Assistant-Requests | v10 | 영어→한국어 번역 (door/curtain/vent/heat/ac) |
| CNN 94-intent 재매핑 | v1 | multi-head 라벨 변환 |

## 주요 버그/실수 기록

| 버전 | 내용 |
|------|------|
| v1 | fn 레벨로만 밸런싱 — fn|exec 조합 불균형 방치 |
| v1 | 142개 파생 파일을 GT로 사용 — 219개 원본이 진짜 GT |
| v3 | GT 원본을 train에 포함한 채 평가 — 100% 허수 |
| v3 | clarify 라벨 버그 — room 추가해도 exec_type 안 바꿈 |
| v1~v6 | "불 꺼줘" 학습 데이터 0개 — 기본 명령 오분류 |
| v8 | 패턴 수정하다 false rejection 발생 — v7보다 나빠짐 |

## 실험 기록

### 실패한 시도
| 버전 | 시도 | 결과 |
|------|------|------|
| v22 | param_type 데이터 수정 | 다른 head regression |
| v26 | 감탄형 exec 수정 | 환기/뉴스 regression |
| v27 | 왔어/에어컨필터 수정 | Test Suite 97.4% regression |
| v29 | 전체꺼 → light 수정 | 시끄러워/환율 regression |
| v30 | 22도/온풍기/이비인후과 수정 | 답답해/시끄러워 regression |
| v31 | 알람꺼/장마감/유가전망 수정 | 미세먼지/남방/냄새 regression |
| v32 | 통합 재증강 시도 | 전반적 하락 |
| v33 | 미세문지/손풍 STT 수정 | 주식시세/달러시세 regression |

### 교훈
- 패치 데이터 추가 시 기존 패턴이 밀림 (data poisoning)
- v28이 최적점 — 더 이상의 패치는 regression 유발
- param_type은 모델보다 규칙 보정이 효과적
- Test Suite로 regression 즉시 감지하는 것이 핵심
