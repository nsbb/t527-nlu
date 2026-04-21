# T527 온디바이스 AI 프로젝트 — 통합 개요 (2026-04-21 기준)

르엘 어퍼하우스 AI 월패드 (Allwinner T527 NPU)에서 동작하는 **완전 오프라인 음성 AI 파이프라인**.

## 전체 파이프라인

```
사용자 음성 입력
    │
    ├── FM1388 디지털 마이크 (월패드 내장)
    │
    ▼
[STT] Conformer QAT (102MB NB, CER 8.86%)
    │  250ms/chunk (3초 오디오)
    │  출처: SungBeom Korean Conformer + AIHub 1M 샘플 QAT
    ▼
한국어 텍스트 (STT 결과)
    │
    ├── preprocess.py (120개 STT 교정 + 한글숫자 변환)
    │
    ▼
[NLU] Ensemble v28+v46 (105MB ONNX, TS 93.59% + KE 97.79%)
    │  0.55ms CPU 추론
    │  fn(20) + exec(5) + dir(9) + param(5) + judge(5) 5-head 분류
    ▼
구조화된 Action
    │
    ├── Rule-based slots (room, value regex)
    ├── DST (멀티턴 상태 추적, 10초 timeout)
    ├── Post-processing rules (param_type, 밝게/어둡게, conf fallback)
    │
    ▼
[Response] 템플릿 매칭 + 디바이스 제어
    │
    ▼
TTS (외부) → 사용자 응답
```

## 두 개의 핵심 모델

### 1. STT: Conformer QAT (CER 8.86%)

| 항목 | 값 |
|------|------|
| 베이스 | SungBeom/stt_kr_conformer_ctc_medium |
| 아키텍처 | Conformer-CTC (CNN + Attention hybrid) |
| Params | 122.5M (114M trainable) |
| Layers | 18 |
| Vocab | 2049 BPE |
| 양자화 | uint8 QAT (FakeQuantize + MarginLoss) |
| 데이터 | AIHub 1M × 1 epoch (59,375 steps) |
| NB 크기 | **102MB** |
| T527 NPU 추론 | **250ms/chunk** |
| **CER** | **8.86%** (11개 데이터셋 18,368 샘플 평균) |
| 배포 | `conformer/models/qat_1m_ep01_best/network_binary.nb` (Git LFS) |

**평가 결과 (데이터셋별)**:
| 데이터셋 | CER |
|---------|:---:|
| modelhouse_2m (실측) | 6.22% |
| modelhouse_3m (실측) | 12.40% |
| 7F_HJY | 7.48% |
| 7F_KSK | 8.21% |
| eval_clean | 4.12% |
| eval_other (noisy) | 7.89% |
| 저음질 | 12.15% |
| 강의 | 10.34% |

**레포**: [Bitbucket hdclabs/t527-stt](https://bitbucket.org/hdclabs/t527-stt)

### 2. NLU: Ensemble v28+v46 (TS 93.59% + KE 97.79%)

| 항목 | 값 |
|------|------|
| 아키텍처 | CNN 4L × 2 (ko-sbert 768d frozen + 1.5M trainable per model) |
| Heads | 5 (fn, exec_type, param_direction, param_type, judge) |
| 앙상블 전략 | Strategy B (fn=v46, exec/dir/param=v28, judge=v46) |
| 훈련 데이터 | 24.5K (GT 219 × 25배 증강 + KoELECTRA 13K pseudo-labeled) |
| ONNX 크기 | **105MB** (FP32, embedding 공유) |
| CPU 추론 | **0.55ms** |
| **TS combo** | **93.59%** (수정 라벨 3,043 케이스) |
| **KoELECTRA fn** | **97.79%** (외부 1,536 케이스) |
| STT 오류 내성 | **100%** (10/10) |
| 배포 | `checkpoints/nlu_v28_v46_ensemble.onnx` |

**레포**: [Bitbucket hdclabs/t527-nlu](https://bitbucket.org/hdclabs/t527-nlu), [GitHub nsbb/t527-nlu](https://github.com/nsbb/t527-nlu)

## 실험 요약

### STT 실험 (2026-02~04)

총 18개 QAT 모델 비교:
```
83k × 1ep    → CER 9.80%
166k × 1ep   → CER 9.59%
330k × 1ep   → CER 9.06%
500k × 2ep   → CER 9.12% (반복 효과 없음)
1M × 1ep     → CER 8.86% ★ (최고, 데이터 다양성 > step 수 증명)
1M × 2ep     → 감소 없음
```

**핵심 교훈**: **데이터 다양성 > 반복** (같은 데이터 2번보다 2배 다른 데이터 1번이 우월)

**참고**:
- `docs/QAT_BEST_MODEL.md` — 상세 레시피
- `WORK_SUMMARY_2026.md` — 2개월 작업 전체 요약
- 논문 초안 SLT 2026 준비 중

### NLU 실험 (2026-04-13~21)

총 **68개 버전 + iter1-6** = 74개 실험:

**성공한 milestones**:
- v28 (04-19): Test Suite 3K, GT 100% 커버리지
- v34 (04-19): Pseudo-labeling +21%p KE 향상
- **v46 (04-20)**: Mixup → 단일 모델 최적 (TS 93.3%, KE 97.8%)
- **v66 (04-21)**: Ensemble ONNX 배포 (TS 94.3%, KE 97.8%)

**시도했으나 실패한 것**:
- v47~v53: Label Smoothing, SupCon, CutMix, Focal 등 (모두 미미 or 실패)
- v55~v62: KD, Two-stage, Wider, Head masking, Model Soup, Warm-start, Multi-seed (모두 실패)
- v63: Conformer 2L (CNN이 더 나음, 24.5K 규모에서)
- v64~v65: Unfreeze emb, KLUE-RoBERTa emb (ko-sbert 유지가 최적)
- v68~v70: 라벨 수정 + 재학습 (부분 수정이 일관성 파괴)
- iter1 (Retrieval Hybrid): Pool sparse
- iter5 (FP16): CPU 속도 37x 느림

**핵심 교훈**:
1. **"v46은 수학적 ceiling"** — 현 데이터/레시피 조합에서 더 올릴 방법 없음
2. **Patch 기반 수정은 regression** (작든 크든 다 실패)
3. **Ensemble B가 balanced 최고** — TS/KE trade-off의 최적 결합

## 핵심 철학적 발견 (토론 결과)

### 1. 라벨은 진짜 GT가 아니다

엑셀 원본에는 **사용자 발화 + AI기대응답**만 있음.  
Multi-head 라벨 (fn, exec, dir...)은 **`parse_gt_scenarios.py`의 규칙 기반 guess**.

- fn: 95% 신뢰 (세부기능 컬럼 매핑)
- exec: 70% 신뢰
- **dir: 60% 신뢰** (slash 포함 행에서 혼란: "가스 밸브 닫아줘 / 열어줘" → open 잘못 라벨)

→ 전체 벤치마크 수치가 **자기복제 측정** (잘못된 라벨에 모델이 맞춤).

### 2. Multi-head vs Flat intent

업계는 대부분 structured output:
- Google Trait, Alexa Slot Filling, Siri Domain/Intent/Parameter
- Facebook MTOP (우리 설계 근거)
- 순수 Flat intent는 드묾

**실험 증명** (v18 Flat 94 vs v46 MH):
- OOD 거부: Flat 1/5 vs MH 5/5
- KoELECTRA: Flat 72.5% vs MH 97.8% (+25.3%p)

단 이 차이는 **아키텍처 독립 기여는 분리 안 됨** (v46은 KE 데이터도 사용).

### 3. Unknown과 Multi-head 조합 일반화는 **보완 관계**

테스트 증명:
- In-domain 학습 안 본 조합 (6/6): 모두 conf=1.0으로 정확 조립
- Out-of-domain (4/5): 모두 fn=unknown 정확 거부

서로 다른 문제를 담당 → 충돌 없이 작동.

## 배포 상태

### STT
- ✅ T527 디바이스에서 `vpm_run` 검증 완료
- ✅ Android 앱 (t527_ondevicevoice_service)에 배포
- ✅ 8.86% NB 파일 Bitbucket hdclabs/t527-stt/main에 있음

### NLU
- ✅ Ensemble ONNX 준비 (CPU에서 0.55ms 검증)
- ⚠️ T527 NPU 변환 미완 (ONNX → NB)
- ⚠️ Android JNI 통합 미완
- ⚠️ 현재 CPU ONNX로만 실행 가능

## 다음 단계 (우선순위)

### 즉시 (이번 주)
- [ ] NLU ONNX → T527 NPU NB 변환 시도
- [ ] Android 앱에 NLU 통합 (t527_ondevicevoice_service)
- [ ] 첫 번째 end-to-end 파이프라인 테스트 (STT → NLU → Response)

### 단기 (1~2개월)
- [ ] **실사용 로그 수집 파이프라인** (NLU 성능 향상의 유일한 경로)
  - 상세: `docs/FEEDBACK_SYSTEM_DESIGN.md`
- [ ] Weekly review queue 운영
- [ ] 첫 번째 피드백 기반 재학습

### 중기 (3~6개월)
- [ ] GT 수동 재구축 (전문가 검수)
- [ ] A/B 테스트 infra
- [ ] 월간 재학습 cycle 자동화

### 장기 (6개월+)
- [ ] 논문 제출 (SLT 2026) — STT QAT 기여
- [ ] Drift 모니터링 + 대응
- [ ] 확장 도메인 검토

## 핵심 산출물 위치

### STT (`t527-stt` repo)
```
conformer/models/qat_1m_ep01_best/
├── network_binary.nb          ★ 배포 NB (102MB LFS)
├── nbg_meta.json
├── inputmeta.yml
└── README.md                    상세 사용 가이드
WORK_SUMMARY_2026.md            2개월 전체 정리
conformer/docs/QAT_BEST_MODEL.md  QAT 상세
```

### NLU (`t527-nlu` repo)
```
checkpoints/nlu_v28_v46_ensemble.onnx  ★ 배포 ONNX (105MB)
checkpoints/cnn_multihead_v28.pt         개별 모델
checkpoints/cnn_multihead_v46.pt         개별 모델
scripts/sap_inference_v2.py              end-to-end 추론
scripts/ensemble_inference_with_rules.py  후처리 rule 적용 버전
scripts/dialogue_state_tracker.py         DST
scripts/preprocess.py                     STT 교정 (120개)

docs/
├── MODEL_CARD.md                        공식 모델 카드
├── DEPLOYMENT_CHECKLIST.md              ★ 10-section 배포 체크
├── DEPLOYMENT_GUIDE.md                  상세 통합 가이드
├── FEEDBACK_SYSTEM_DESIGN.md            ★ 실사용 로그 설계
├── KNOWN_FAILURES.md                    실패 패턴
├── CHANGELOG.md                         전체 버전 히스토리
├── ROADMAP.md                           향후 계획
├── DISCUSSION_2026_04_21.md             설계 토론 (엄격판)
├── SESSION_2026_04_21_PM.md             오후 세션 종합
└── PROJECT_OVERVIEW_2026_04_21.md       이 문서
```

## 한 줄 결론

> **"STT CER 8.86% + NLU TS 94%/KE 98% 달성. 모델 실험 소진 — 남은 개선 경로는 실사용자와 함께 배우는 것뿐."**
