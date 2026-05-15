# NLU 버전 매트릭스 (모델 / 앙상블 / 룰 / NB)

## Current state

4 종류 버전이 섞여있어 헷갈리기 쉬움. 분리해서 한 곳에 정리.

### A. 학습 모델 (`checkpoints/cnn_multihead_v*.pt`)

| 버전 | 날짜 | TS combo | KE fn | 르엘 219 | 특징 | 평가 |
|---|---|---|---|---|---|---|
| v1~v9 | 4/18 | ~90% | — | — | 초기 5-head 학습 | history |
| v10~v27 | 4/19 | 95~98% | — | — | 반복 fix | 폐기 |
| **v28** | 4/19 | **96.4%** | 75.5% | — | TS 100%, GT 전용 | 기존 패턴 강함 ⭐ |
| v33 | 4/19 | 90.5% | — | — | KoELECTRA 직접 병합 | ❌ regression |
| **v34** | 4/19 | 93.6% | **96.8%** | — | pseudo-label 돌파 | 단일 균형 |
| v40 | 4/20 | 92.3% | 97.2% | — | KD from v28 | KE 우선 |
| **v46** | 4/20 | 93.3% | **97.8%** | — | mixup augmentation | 단일 일반화 최고 ⭐ |
| v47~v62 | 4/21 | 모두 v46 미달 | — | — | 9+ 가지 강화 (KD/2-phase/wider/masking/Soup/Conformer 등) | ❌ 다 실패 |
| v63 | 4/21 | 79.6% | 94.8% | — | Conformer 2L | ❌ |
| v64~v68 | 4/21~22 | — | — | — | 데이터 품질 정비 | 부분 개선 |
| v71 | 4/28 | 93.5% | 97.2% | 93.6% | 어두침침 fix + paraphrase | 중간 |
| **v72** | 4/28 | **94.4%** | 97.2% | **93.2%** (실측) | 비유 3,961개 + 간접 20/21 | **production** ⭐ |
| v73~v78 | 4/28 | — | — | — | **모델 아님** — 룰/코드 revision (C 섹션 참조) | — |

### B. Ensemble (prediction-level 결합)

| 조합 | 전략 | TS | KE fn | 르엘 219 | 비고 |
|---|---|---|---|---|---|
| v28+v34 (B) | fn=v34, exec/dir=v28 | 94.3% | 96.8% | — | 옛 |
| **v28+v46** | fn=v46, exec/dir=v28 | 94.3% | 97.8% | ~81% | **디바이스 현재 사용** ⚠️ |
| v28+v71 | 동일 패턴 | — | — | 93.6% | 중간 |
| **v28+v72** | fn=v72, exec=v28, dir=v72, judge=v72 | **94.4%** | 97.2% | **94.06%** | **서버 production** ⭐ |

`scripts/export_ensemble_onnx.py` 의 `EnsembleTupleModel.forward()`에서 head별 선택 변경 가능.

### C. 룰/코드 revision (모델 없음)

| 버전 | 날짜 | 내용 | 효과 |
|---|---|---|---|
| iter9 | 4/21 | rule + DST slot filling | TS 95.76% (+2.2%p) |
| v100 | 4/26 | 불만/수사적/존댓말 | 부분 |
| v104~v118 | 4/26 | STT 교정 추가 (남방→난방, 에어껀→에어컨, 등) | preprocess 보강 |
| v126~v133 | 4/26 | 비유/은유/exec_type 보정 | 비유 +23%p |
| v73 | 4/28 | 간접 21/21 완성 + 냉난방 충돌 자동 해소 | 룰 |
| v74 | 4/28 | 종합 자동 테스트 + 비유 dir 교정 | 룰 |
| v77 | 4/28 | 한국어 NLU 한계 연구 기반 규칙 | 룰 |
| v78 | 4/28 | 집 상태 전이 멀티턴 심화 + DST 강도 상속 | 룰/DST 코드 |

→ Kotlin (`t527_smart_v2`)에는 `PostRulesV4`까지 (v133 룰 일부 + 자동 36 + 수동 25) 적용됨.

### D. NPU NB 변환

| NB | 모델 | 양자화 | fn 정확도 | 상태 |
|---|---|---|---|---|
| cnn_body_v28.nb | v28 단독 | uint8 | 71.7% | 비교용 폐기 |
| cnn_body_v46.nb | v46 단독 | uint8 | **54.5%** | ❌ 폐기 (양자화 -42%p 손실) |
| **cnn_body_v46_int16.nb** | v46 단독 | int16 fl=15 | **97%** (ONNX 동등) | **현재 NPU 모드** ⭐ |
| cnn_body_v72.nb | v72 단독 | — | — | ❌ **미변환** (production gap) |

NPU 실측 (2026-05-07):
- vip_run_network 호출 확인 — VIP9000NanoSI Plus HW 가속 작동
- Latency: warmup 후 ~1.5ms/추론 (CPU 21~30ms 대비 14~20x)

### E. 위치별 현재 사용 (3 곳 gap)

| 위치 | 모델 | 룰 | 르엘 219 |
|---|---|---|---|
| **서버 production** (`deployment_pipeline_v2.py`) | v28+v72 ensemble | v78 | **94.06%** |
| 디바이스 CPU (`t527_smart_v2` 앱) | v28+v46 ensemble | PostRulesV4 (v133 일부) | ~81% |
| 디바이스 NPU | v46 단독 cnn_body int16 | PostRulesV4 | **60.7%** |

## Known-good settings

- **현재 production 단일 진실**: `deployment_pipeline_v2.py` 가 single source — `nlu_v28_v72_ensemble.onnx`
- Ensemble head 선택: fn=v72, exec=v28, dir=v72, param=v28, judge=v72
- 모델 학습 종료점: **v72 (4/28)** — 그 이후는 룰/DST 보강만
- NPU 양자화는 **int16 필수** (v46 mixup 모델 wide dynamic range → uint8 부족)
- 새 모델 학습 전에 `docs/VERSION_LOG.md` + `docs/CHANGELOG.md` 먼저 읽기 (이미 시도/실패한 기법 9가지 + 데이터 패치 함정 등 기록됨)

## Open issues

- **v72 cnn_body NPU 변환 미완** — production v72 ensemble을 NPU에서 못 살림
- 디바이스에 옛 v46 ensemble ONNX 그대로 → CPU 모드 정확도 한 세대 뒤처짐
- v72 ensemble은 head별 다른 모델 (fn=v72, exec=v28) → NPU 변환 시 두 cnn_body NB 둘 다 만들고 head별 선택 필요
- 학습 데이터 `train_v72.py` 파일 없음 — v72 학습은 인라인 스크립트로 실행됨. v73(=모델) 학습하려면 `train_v71.py` 복사해서 데이터 경로만 수정
- 르엘 GT-219 라벨 중 "뉴스 dir=on/exec=control" 13건 의심 — 팀 분류 받을 때 검토

## Related sources

- `raw/source-notes/src-model-version-gap-20260513.md` — 서버↔디바이스 ONNX md5 비교
- `wiki/models/cnn-multihead-v72.md` — production 모델 단독 페이지
- `wiki/projects/t527-npu-integration.md` — NPU NB 변환 + JNI 통합
- `wiki/issues/server-device-version-gap.md` — 3 위치 gap
- `wiki/decisions/regression-harness.md` — pre-commit 회귀 체크
- `docs/VERSION_LOG.md` (416줄) — 버전별 실험 결과
- `docs/CHANGELOG.md` (2294줄) — 룰 revision r1~r1990 변경 기록
- `docs/HANDOFF_2026_04_28.md` — v72 핸드오프 (가장 최신 종합 문서)

## Last updated

- `2026-05-15`
