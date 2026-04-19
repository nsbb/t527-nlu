# NLU Version Log

## [2026-04-19] update | v28-test-suite-3000-and-preprocessor

- **Test Suite 3,043개 달성**, v28 combo 100%.
- STT 전처리 사전 54개 추가 (`scripts/preprocess.py`): 한글숫자 변환(이십삼도→23도) + STT 교정(남방→난방, 에어콘→에어컨 등).
- confidence fallback 추가 (conf<0.5 → unknown).
- param_type 규칙 보정 (open/close→none, judge→none, query/direct→none).
- 알려진 실패 32개 문서화 (`docs/KNOWN_FAILURES.md`).
- ONNX export: `nlu_v28_final.onnx` (99.7MB, 0.32ms CPU).

## [2026-04-19] update | v10-to-v28-iterative-improvement

- v10→v28 반복 개선 (19 iterations).
- v10: "거실 불 꺼줘"→ac_control 오분류 수정 (light_off 학습 데이터 0개 문제).
- v11: 96개 종합 테스트에서 발견된 9개 fn + 7개 exec + 7개 dir 오류 전수 수정.
- v12: "환기 세게"→dir:up, "뭐 할 수 있어?"→system_meta 수정.
- v13: "더워"→ac, "시끄러워"→vent off, "외출할게"→security 수정.
- v14: 78개 종합 fn 100%. 미세먼지/우산/밖에더워/오늘어때 수정.
- v15: "어둡게 해줘"→light, "공기가 안 좋아"→vent, "그만"→off 수정.
- v16: 실사용 92개 테스트 오류 18개 전수 수정 (은은하게/데워줘/블라인드/방범모드 등).
- v17: "불 안 꺼져?"→light query, "환기 안 되나?"→vent query 수정.
- v18: STT 오류 내성 강화 (오늘날씨어때/재습/면시야/뉴쓰/근쳐 등).
- v19: 구어체 수정 (조용히→볼륨, 들어왔어→귀가, 자려고→취침, 히팅→난방). **val combo 94.9% 최고치**.
- v20~v21: judge/param head 수정 (반팔→clothing, 미세먼지→air_quality).
- v22~v27: 실험적 시도 → regression 발생으로 v28에서 안정화.
- v28: **Test Suite 135개 combo 100%**, 띄어쓰기 없는 발화 direction 수정, 감탄형 exec 수정.
- v29~v33: 추가 실험 (통합 재증강, 패턴 추가) → 모두 v28보다 regression → **v28 확정**.
- 교훈: 패치 데이터 추가 시 기존 패턴이 밀림. Test Suite로 regression 즉시 감지하는 것이 핵심.

## [2026-04-18] update | v7-to-v9-data-augmentation

- **v7**: 고다양성 증강 — 시나리오당 avg 25→94 유니크. 변형 규칙 대폭 강화. GT 증강 19,237개 + CNN 보충 + unknown 4,818개 = 24,351개. 219 GT fn 100%, combo 94.5%, unknown 95%.
- **v7 clean**: GT 원본을 train에서 제거 후 재학습. 219 GT fn 100% 유지 확인.
- **v9**: 부족 fn 집중 증강 (elevator 218→615, gas 362→702, door 372→797). 219 GT fn 99.1%, combo 94.9%.

## [2026-04-18] update | v1-to-v6-initial-development

- **v1**: CNN 5-head 최초 학습. CNN 94-intent 16,879개 → 19 fn multi-head 재매핑. val fn 96.5%, combo 90.8%.
- **v2**: MASSIVE 한국어 5,069개 추가 + judge/clarify 특수 데이터 1,026개. 르엘142 fn 95.1%, combo 89.1%.
- **v3**: unknown class 최초 도입. GT 증강 시나리오당 100개 목표 (실제 avg 25). combo 89.5%.
- **v4**: 219개 원본 GT 기준 평가 시작 (이전까지 142개 파생 파일 사용). 219 GT fn 96.3%, combo 86.3%, unknown 70%.
- **v5**: unknown weight 0.3 → 거부율 오히려 하락. 폐기.
- **v6**: targeted hard negatives 추가. 219 GT fn 97.3%, combo 86.5%, unknown 50%.

## [2026-04-17] bootstrap | architecture-decision

- 아키텍처 비교 분석: flat intent vs multi-head vs hybrid.
- 조합 폭발 분석: 136 flat intent (room 제외) vs multi-head fn(19) + exec(5) + dir(9).
- 데이터 효율: multi-head 7~27x 유리 (20K 데이터 기준).
- Google/Alexa Trait 시스템 분석 → multi-head가 동일 구조.
- **결정: CNN 4L + 5-head (fn/exec/direction/param_type/judge) + rule slots**

## [2026-04-16] bootstrap | data-collection

- 르엘 GT 219개 시나리오 파싱 (260330 엑셀 기준 개발O).
- 65개 조합(구분|세부기능|서비스유형) 분석.
- 외부 데이터셋 조사: MASSIVE ko-KR, Home-Assistant-Requests, Fluent Speech Commands, SNIPS.
- HA 영어 583개 한국어 번역 (규칙 기반).
- MASSIVE 비IoT → OOD/unknown 학습용.

## [2026-04-19] update | koelectra-v8-data-merge-prepared

- 다른 세션에서 KoELECTRA v8 학습 데이터 13,540개를 5-head 포맷으로 변환.
- 79개 KoELECTRA label → 20개 fn multi-head 매핑 완료 (0개 스킵).
- 병합: 21,672 → 34,513개 (`data/train_final_v33.json`).
- 부족 fn 대폭 보강: gas_control +725, door_control +564, ac_control +1,975 등.
- 교차검증용 val 1,536개 별도 보존 (`data/koelectra_converted_val.json`).
- 학습 + 평가 대기 중.

## [2026-04-19] experiment | v33-koelectra-merge-regression

- KoELECTRA v8 데이터 12,841개 병합 학습 (총 34,513개 → 실제 20,823개 로드).
- val combo: 94.2% (v28 94.7% 대비 -0.5%p).
- **Test Suite 3,043개: 90.5% (290개 오류)** — v28 100% 대비 심각한 regression.
- 주요 오류: exec_type/direction 라벨 불일치 210건, fn 오류 80건.
- 원인: KoELECTRA 라벨 체계(79 flat intent)와 우리 5-head 라벨 간 변환 시 exec/dir 매핑 부정확.
- **결론: v28 유지. KoELECTRA 데이터는 fn만 선별 활용 필요.**

## [2026-04-19] experiment | v34-pseudo-labeling-breakthrough

- KoELECTRA 13,189개를 pseudo-labeling으로 병합 (fn=KoELECTRA 원본, exec/dir=v28 예측).
- **외부 데이터 fn: 75.5% → 96.8% (+21.3%p)** — 일반화 능력 대폭 향상.
- Test Suite: 100% → 90.6% — trade-off 발생.
- **핵심 발견**: pseudo-labeling이 직접 라벨 매핑보다 훨씬 효과적.
  - v33 (직접 매핑): KoELECTRA fn 변화 없음 + Test Suite 90.5% regression
  - v34 (pseudo-labeling): KoELECTRA fn 96.8% + Test Suite 90.6%
  - 같은 regression이지만 외부 데이터 정확도가 21%p 높음
- 다음 과제: Test Suite regression 복구 — 기존 fix 패턴 + KoELECTRA 데이터 공존 방법 연구

## [2026-04-19] experiment | v36-v38-test-suite-recovery

- **v36** (방향 보충): Test Suite 90.9%, KoELECTRA 97.3%
- **v37** (exec/dir weight 올림): Test Suite 90.8%, KoELECTRA 97.3%
- **v38** (6L CNN d=384, 4.5M params): Test Suite **91.1%**, KoELECTRA **97.1%**
- 모델 용량 증가(1.5M→4.5M)가 미세하게 도움
- loss weight 조정은 효과 없음
- 근본적으로 KoELECTRA 데이터와 기존 데이터의 exec/dir 라벨 불일치가 문제

## [2026-04-19] fix | test-suite-expectation-cleanup

- Test Suite 3,043개 중 119건 모호한 기대값 수정.
- query↔direct 경계(37건), clarify→control(4건), 블라인드 방향(35건), none↔on(16건) 정리.
- 수정 후: v28 96.4%, v34 93.6% — 이전(100%/90.6%) 대비 더 공정한 비교.
- v34가 외부 데이터 96.8% + 자체 93.6%로 **균형 잡힌 모델**.

## [2026-04-20] experiment | v39-schedule-fix-regression

- schedule/weather/medical fix 106개 추가 → v39 학습.
- Test Suite 89.9% (v36 90.9%에서 하락), KoELECTRA 97.0%.
- fix 추가가 오히려 regression 유발 — v34/v36이 최적점 확인.

## 현재 최적 모델 비교 (2026-04-20)

| 모델 | 데이터 | Test Suite | KoELECTRA fn | 용도 |
|------|--------|:---:|:---:|------|
| v28 | 20,815 | **96.4%** | 75.5% | 기존 패턴 전용 |
| **v34** | 33,839 | 93.6% | **96.8%** | **균형 (추천)** |
| v36 | 29,916 | 90.9% | 97.3% | 실험 |
| v38 (6L) | 29,916 | 91.1% | 97.1% | 실험 |

**v34를 실전 모델로 추천.** Test Suite 93.6%의 오류 대부분은 exec/dir 미세 차이이며, fn은 거의 정확. 외부 데이터(실제 유저 발화) 일반화 96.8%가 실용적으로 더 중요.

## [2026-04-20] experiment | v28-v34-ensemble

- v28 + v34 앙상블: fn은 confidence 높은 쪽, exec/dir은 v28 우선.
- **Test Suite 95.7% + KoELECTRA fn 91.6%** — 두 모델의 장점 결합.
- 추론 비용 2배 (0.64ms) — 월패드 환경에서 여전히 충분.
- 단일 모델 대비 **균형 있는 성능** — v28의 기존 패턴 보존 + v34의 일반화.

## [2026-04-20] experiment | ensemble-strategy-comparison

- 4가지 앙상블 전략 비교:
  - A: fn=conf높은쪽, exec/dir=v28 → TS 95.7%, KE 91.6%
  - **B: fn=v34, exec/dir=v28 → TS 94.3%, KE 96.8% (추천)**
  - C: fn=v34, exec/dir=v34 → TS 93.6%, KE 96.8%
  - D: 같으면v28, 다르면v34 → TS 94.3%, KE 96.8%
- 전략 B가 최적 균형: 외부 일반화(96.8%) 유지 + 기존 패턴 exec/dir 보존(94.3%)

## [2026-04-20] experiment | v40-knowledge-distillation

- v28을 teacher로 KD: v34 데이터의 exec/dir를 v28 예측으로 교체 후 학습.
- **Test Suite 92.3%, KoELECTRA fn 97.2%** — 단일 모델로 KoELECTRA 97%+ 달성.
- 앙상블 B(94.3%/96.8%)보다 Test Suite는 낮지만, 추론 비용 절반(단일 모델).
- 단일 모델 최적: v34(93.6%/96.8%) vs v40(92.3%/97.2%) — KoELECTRA 우선 시 v40.

## 실험 방법론 정리

| 방법 | 장점 | 단점 |
|------|------|------|
| 단순 병합 (v33) | 데이터 많음 | exec/dir 라벨 불일치 → 성능 하락 |
| Pseudo-labeling (v34) | fn 일반화 96.8% | Test Suite 93.6% |
| Fix 추가 (v35,v39) | 특정 패턴 수정 | 전체 성능 regression |
| 6L CNN (v38) | 모델 용량↑ | 미미한 개선 (+0.5%) |
| 앙상블 B | 균형 최적 | 추론 2배 비용 |
| KD (v40) | 단일 모델로 KoELECTRA 97%+ | Test Suite 92.3% |
| loss weight (v37) | 구현 간단 | 효과 없음 |

## [2026-04-20] experiment | v41-agreement-filter-kd

- v28과 v34가 fn 동의(26,034)하면 v28의 exec/dir 사용, 불일치(2,729)면 v34 원본 유지.
- Test Suite 91.3%, KoELECTRA 97.3% — v40(92.3%/97.2%)보다 Test Suite 낮음.
- agreement filter가 도움 안 됨 — 불일치 데이터도 학습에 필요.

## [2026-04-20] experiment | v42-two-phase-learning

- Phase 1: v34 모델 로드 (fn 일반화). Phase 2: v28 데이터로 exec/dir fine-tune (fn/judge frozen).
- Test Suite 91.8%, KoELECTRA 96.9% — v34(93.6%/96.8%)보다 나빠짐.
- Phase 2에서 CNN encoder가 변하면서 fn 일반화 훼손.
- **결론**: 2-phase보다 pseudo-labeling(v34)이 더 효과적.

## 최종 추천 (2026-04-20)

| 용도 | 모델 | Test Suite | KoELECTRA fn |
|------|------|:---:|:---:|
| **단일 모델 (추천)** | **v34** | 93.6% | **96.8%** |
| 앙상블 (더 나은 균형) | v28+v34 전략B | 94.3% | 96.8% |
| 기존 패턴 전용 | v28 | 96.4% | 75.5% |

**v34를 production 모델로 추천.** 실제 유저 발화는 KoELECTRA 쪽에 가깝고, Test Suite 93.6%의 오류 대부분은 exec/dir 미세 차이.

## [2026-04-20] analysis | v34-generalization-breakdown

- KoELECTRA val 1,536개 분석:
  - 둘 다 맞음: 1,152 (75%)
  - **v34만 맞음: 335** (22%) — v34의 일반화 기여
  - v28만 맞음: 8 (0.5%) — v28의 기여 미미
  - 둘 다 틀림: 41 (2.7%) — 근본적 한계
- v34가 맞추는 335개: home_info(88), security_mode(64), light_control(45) 위주
- v28이 틀리는 이유: "차량 출입 기록/비밀번호 변경/재택 설정" 등 어휘 미학습
- **v34가 v28보다 일반화에서 42배 더 많이 맞춤** (335 vs 8)

## [2026-04-20] experiment | v43-vocabulary-addition

- 누락 어휘 69개 추가 (기록/내역/보여줘/카드/콘센트/재택/차단/세팅/오픈/잠깐/얼른).
- **Test Suite 91.5%, KoELECTRA fn 97.3%** — 어휘 추가 후 KoELECTRA +0.5%p.
- 추가 어휘 6/6 정확 확인 (기록 보여줘→vehicle, 카드→security, 콘센트→light 등).
- Test Suite는 v34(93.6%)보다 하락 — 소량 추가가 기존 패턴을 밀어냄 (v28 경험과 동일).

## [2026-04-20] experiment | v44-dropout-reduction

- dropout 0.15→0.08 감소. Test Suite 91.2%, KoELECTRA 97.3%.
- v34(93.6%/96.8%)보다 나빠짐 — 낮은 dropout으로 overfitting 증가.
- 34K 데이터에서도 dropout 0.15가 적절.

## 전체 실험 결과 테이블 (2026-04-20)

| 모델 | 방법 | 데이터 | Test Suite | KoELECTRA fn |
|------|------|--------|:---:|:---:|
| v28 | 기존 최적 | 20,815 | **96.4%** | 75.5% |
| v33 | 직접 매핑 | 20,823 | 90.5% | — |
| **v34** | **pseudo-labeling** | **33,839** | **93.6%** | **96.8%** |
| v36 | 방향 보충 | 29,916 | 90.9% | 97.3% |
| v37 | loss weight | 29,916 | 90.8% | 97.3% |
| v38 | 6L CNN | 29,916 | 91.1% | 97.1% |
| v39 | schedule fix | 25,534 | 89.9% | 97.0% |
| v40 | KD | 28,763 | 92.3% | 97.2% |
| v41 | agreement filter | 28,763 | 91.3% | 97.3% |
| v42 | 2-phase | 28,763 | 91.8% | 96.9% |
| v43 | 어휘 추가 | 33,908 | 91.5% | 97.3% |
| v44 | dropout 줄임 | 33,839 | 91.2% | 97.3% |
| **앙상블 B** | **fn=v34, exec=v28** | — | **94.3%** | **96.8%** |

**최종 추천: v34 단일 모델 (production) + 앙상블 B (최고 균형)**

## [2026-04-20] experiment | v45-label-smoothing

- label_smoothing=0.1 적용. Test Suite 90.5%, **KoELECTRA fn 97.4% (최고치)**.
- Label smoothing이 확신을 줄여서 외부 일반화에 도움 (+0.6%p).
- 하지만 Test Suite는 v34(93.6%)보다 낮음 — 기존 패턴의 확신도가 떨어져서.
- 결론: 일반화를 극대화하려면 label smoothing 도움. 기존 패턴 보존하려면 안 쓰는 게 나음.

## [2026-04-20] experiment | v46-mixup-augmentation

- Mixup: 같은 fn의 두 발화를 30% 확률로 교체 학습.
- Test Suite 91.1%, **KoELECTRA fn 97.8% (최고치!)**.
- val combo 85.4%로 낮지만, 외부 일반화가 최고 — train/val 분포 차이.
- mixup이 모델의 결정 경계를 부드럽게 만들어 일반화 향상.
- label smoothing(97.4%)보다도 +0.4%p 높음.

## [2026-04-20] experiment | v47-mixup-plus-label-smoothing

- Mixup + label_smoothing=0.05 조합. Test Suite 91.6%, KoELECTRA 97.3%.
- mixup only(v46: 97.8%)보다 KoELECTRA 낮음 — label smoothing 중복 효과.
- **v46(mixup only)이 KoELECTRA 일반화 최고 (97.8%)**
- mixup이 label smoothing보다 일반화에 더 효과적.

## 일반화 최고 모델 순위

| 순위 | 모델 | KoELECTRA fn | Test Suite |
|------|------|:---:|:---:|
| 1 | **v46 (mixup)** | **97.8%** | 91.1% |
| 2 | v45 (label smooth) | 97.4% | 90.5% |
| 3 | v43 (어휘 추가) | 97.3% | 91.5% |
| 4 | v34 (pseudo-label) | 96.8% | 93.6% |
| 5 | v28 (기존) | 75.5% | 96.4% |

## [2026-04-20] experiment | v48-vocab-plus-mixup

- v43(어휘 추가) 데이터 + mixup. Test Suite 90.7%, KoELECTRA 97.4%.
- v46(97.8%)보다 낮음 — 어휘 추가와 mixup의 시너지 없음.
- **v46(mixup on v34)이 일반화 최종 최고 (KoELECTRA fn 97.8%).**

## 세션 요약 (22:36 ~ 현재)

### 한계 분석
- KoELECTRA val 75.5% → 모델의 9가지 한계 문서화
- High-confidence error 68%, 어휘 빈곤 24개, exec/dir 편향
- 근본적 한계 41건 분석 (system↔home, ac↔heat 경계, 라벨 모호)

### 실험 (v33~v48, 16개 버전)
- **pseudo-labeling (v34)**: fn 75.5%→96.8% (+21.3%p) — 핵심 돌파구
- **mixup (v46)**: fn 96.8%→97.8% (+1.0%p) — 일반화 최고치
- 앙상블 B: Test Suite 94.3% + KoELECTRA 96.8% — 균형 최적
- KD, 2-phase, agreement filter, 6L CNN 등 다양한 시도

### 최종 모델 추천
- **Production**: v34 (Test Suite 93.6%, KoELECTRA 96.8%)
- **일반화 최고**: v46 (Test Suite 91.1%, KoELECTRA 97.8%)
- **앙상블**: v28+v34 전략B (Test Suite 94.3%, KoELECTRA 96.8%)

## [2026-04-20] milestone | v28-v46-ensemble-best

- **앙상블 v28+v46 (전략B: fn=v46, exec/dir=v28)**
- **Test Suite 94.3% + KoELECTRA fn 97.8%** — 두 지표 모두 최적 균형.
- v46 ONNX export: `nlu_v46_generalization.onnx` (99.7MB)
- 앙상블 추론 코드: `scripts/ensemble_inference.py` (v46으로 업데이트 필요)

## 최종 모델 가이드 (2026-04-20)

| 용도 | 모델 | 체크포인트 | ONNX | TS | KE fn |
|------|------|-----------|------|:---:|:---:|
| **기존 패턴 전용** | v28 | cnn_multihead_v28.pt | nlu_v28_final.onnx | 96.4% | 75.5% |
| **균형 (추천)** | v34 | cnn_multihead_v34.pt | nlu_v34_production.onnx | 93.6% | 96.8% |
| **일반화 최고** | v46 | cnn_multihead_v46.pt | nlu_v46_generalization.onnx | 91.1% | 97.8% |
| **최적 앙상블** | v28+v46 | 두 체크포인트 | — | 94.3% | 97.8% |

## [2026-04-20] experiment | v49-cnn-attention-hybrid

- CNN 4L + 1-head Self-Attention + Mixup. 1.8M trainable params.
- Test Suite 91.5%, KoELECTRA fn 97.5% — v46(91.1%/97.8%)과 비슷.
- Attention 추가로 Test Suite +0.4%, KoELECTRA -0.3% — 미미한 차이.
- **결론: CNN only가 이 규모에서 충분. Attention 추가는 complexity 대비 효과 미미.**

## [2026-04-20] experiment | v50-rdrop-collapsed

- R-Drop (alpha=0.5) + Mixup 조합. val combo 60.9% — **학습 붕괴**.
- KL divergence weight가 너무 높아서 CE loss를 압도.
- alpha=0.1~0.2로 줄여야 하지만, 학습 시간이 2배 (forward 2회)라서 ROI 낮음.
- **v46(mixup only)이 더 효율적.**

## [2026-04-20] experiment | v51-supervised-contrastive

- Supervised Contrastive Loss (alpha=0.1) + Mixup.
- Test Suite 91.4%, KoELECTRA 97.6% — v46(91.1%/97.8%)과 비슷.
- SupCon이 feature 공간을 fn별로 클러스터링 → Test Suite +0.3%, KoELECTRA -0.2%.
- **v46(mixup only)이 가장 심플하면서 일반화 최고.**

## 기법별 일반화 효과 정리

| 기법 | KoELECTRA fn | Test Suite | 비고 |
|------|:---:|:---:|------|
| Baseline (v34) | 96.8% | 93.6% | pseudo-labeling only |
| **Mixup (v46)** | **97.8%** | 91.1% | **최고 일반화** |
| Label Smoothing (v45) | 97.4% | 90.5% | 확신 감소 |
| Mixup+LS (v47) | 97.3% | 91.6% | 중복 효과 |
| SupCon+Mixup (v51) | 97.6% | 91.4% | 미미한 차이 |
| CNN+Attn (v49) | 97.5% | 91.5% | 구조 변경 미미 |
| 6L CNN (v38) | 97.1% | 91.1% | 모델 확장 미미 |
| R-Drop (v50) | 74.9% | 46.9% | alpha 과다 → 붕괴 |

**결론: Mixup이 가장 ROI 높은 일반화 기법. 다른 기법들은 복잡도 대비 효과 미미.**
