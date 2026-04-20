# NLU Roadmap — 향후 개선 방향

## 현재 상태 (v68, 2026-04-21)

✅ **달성**: Ensemble ONNX 배포, TS 94.3%, KE 97.8%, STT 내성 100%, 0.48ms

## 우선순위별 개선 항목

### 🟢 P1 — 즉시 가능 (코드/설정 변경만)

#### 1. ~~Majority Vote 3-model 앙상블 배포 검토~~ ❌ 기각됨 (2026-04-21 검증)
- **Hard majority vote (PyTorch)**: TS 94.91% (+0.66%p)
- **Soft avg ONNX (실제 배포 가능 형태)**: TS 94.05% (-0.25%p), KE 97.07%
- Hard voting은 ONNX export 어려움 (Counter 연산 기본 제공 없음)
- Soft average는 2-model보다 모든 면에서 열세
- **결론: 2-model Strategy B 유지**

#### 2. "오늘날씨어때" 등 띄어쓰기 없는 패턴 preprocess 확장
- **현황**: 120개 STT 교정 사전
- **개선**: "지금몇시야", "저기가스는", "부산은어때" 등 20~30개 추가
- **영향**: STT 내성 100% → 유지/개선

#### 3. Response 템플릿 상황별 다양화
- **현황**: fn당 1~2개 템플릿 (반복적)
- **개선**: 시간대/상황별 variation (아침/밤, 여름/겨울)
- **예**: "네, 거실 에어컨을 켰습니다" → "네, 거실 에어컨 가동할게요"
- **구현**: RESPONSE_TEMPLATES를 리스트로 바꾸고 random.choice

#### 4. Confidence 기반 세밀한 fallback
- **현황**: conf<0.5 → unknown
- **개선**: conf 0.3-0.5 → clarify (재질의), conf<0.3 → unknown
- **효과**: 모호한 발화에 대한 UX 개선

### 🟡 P2 — 중간 작업 (1~3일)

#### 5. 학습 데이터 라벨 오류 체계적 수정
- **현황**: 46건 수정됨 (v68), 더 많을 것으로 추정
- **작업**: train_final_v43.json 전체 검토, 자동 탐지 + 수동 확인
- **예상 개선**: Test Suite +1~2%p
- **리스크**: regression (v58처럼) — 신중하게 진행

#### 6. Test Suite 확장 (3,043 → 5,000+)
- **현황**: 3,109개 (v67 확장), 66개 케이스가 취약 패턴 노출
- **추가 필요**:
  - schedule_manage (알람/타이머) 100+
  - 구어체 방향 (따뜻하게=up) 50+
  - 멀티턴 시나리오 100+
  - 극단 STT 오류 50+
- **구축**: 실제 사용 로그 기반 (없으면 수동 작성)

#### 7. DST 고도화
- **현황**: room/device/confirm follow-up, 10초 timeout
- **개선**:
  - 5턴 이상 context window
  - Slot filling (온도 25도 맞춰줘 → 다음 턴 "더 올려줘")
  - Context confidence (오래된 context 점진적 감소)
  - "아, 역시" 같은 correction 패턴 추가

#### 8. Android JNI 통합 (awnn VNN)
- **목표**: T527 NPU에서 Ensemble ONNX 실행
- **경로**: ONNX → Acuity pegasus → NB (network_binary)
- **이슈**:
  - 토크나이저 Java/Kotlin 포팅 (BertTokenizer)
  - STT 전처리 사전 포팅 (120개 regex)
  - DST 상태 관리 (timer 등)

### 🔴 P3 — 대규모 작업 (1주+)

#### 9. 실제 사용 로그 수집 파이프라인
- **목표**: production 발화 로그 → feedback loop
- **구조**:
  ```
  User 발화 → NLU → [Log + Confidence]
    ↓ conf<0.7 or manual correction
  Review Queue → 라벨링
    ↓
  학습 데이터 증강 → 월간 재학습
  ```
- **인프라 필요**: 로깅 서버, 익명화, review UI

#### 10. Larger model 탐색 (Distil-KoELECTRA 등)
- **현재**: 1.5M trainable, 26M total (frozen emb)
- **탐색**: KLUE-DistilBERT/KoELECTRA-small fine-tune
- **조건**: T527 NPU 지원 확인 (현재 CNN 기반만 검증됨)

#### 11. 다중 언어 지원 (영어 혼용)
- **상황**: "turn on AC", "Hi wonder" 등 영어 혼용 발화
- **현황**: 제한적 (영어만 들어오면 unknown 많음)
- **작업**: multilingual embeddings + mixed-lang 학습 데이터

### ⚪ P4 — 장기 연구

#### 12. 완전한 Value 추출 (현재는 규칙 기반)
- **현황**: 정규식 기반 (23도, 30분, 50%)
- **개선**: Token-level span prediction (NER 스타일)
- **가능성**: SAP 모델 (plan에 있음)에 value pointer head

#### 13. Compositional generalization
- **현황**: 학습 데이터에 없는 조합 일부 취약
  - "안방 에어컨 제습 27도" 같은 복합 발화
- **접근**: Compositional data augmentation

#### 14. Streaming inference (발화 중간에 예측)
- **현황**: 발화 끝난 후 처리
- **목표**: 실시간 부분 발화에서 점진적 예측
- **근거**: UX 개선 (반응성)

## 측정 목표 (Nice-to-have)

| 지표 | 현재 | 목표 | 기간 |
|------|:---:|:---:|:---:|
| Test Suite combo | 94.3% | 97%+ | 6개월 |
| KoELECTRA fn | 97.8% | 99%+ | 6개월 |
| 실제 사용자 만족도 | - | 90%+ | 실배포 후 |
| STT 내성 | 100% (10/10) | 확장 100+ 케이스 | 3개월 |
| CPU latency | 0.48ms | < 1ms (NPU: < 0.3ms) | 3개월 |

## 의존성 그래프

```
P1.1 (3-model 배포)
  ↓ (성공 시)
P2.5 (라벨 정제) ────→ P2.6 (TS 확장) ──→ P3.9 (로그 파이프라인)
                          ↓
                     P2.7 (DST 고도화)
                          ↓
P2.8 (Android 통합) ──────┴──→ P3.10 (Larger model)
```

## 실패 기록 (v54-v68 교훈)

| 시도 | 결과 | 교훈 |
|------|------|------|
| KD from ensemble | -2%p | Soft label로 ensemble 재현 불가 |
| Wider model | -3%p | 데이터 대비 capacity 과다 |
| Head masking | -11%p | Backbone은 전체 gradient 필요 |
| Model soup | 실패 | Loss landscape 비볼록 |
| Conformer | -14%p | 24.5K 데이터에서는 CNN 우세 |
| 패치 증강 | -2%p | 분포 왜곡으로 regression |
| 임베딩 unfreeze | -3%p | ko-sbert 이미 최적 |
| KLUE-RoBERTa emb | -4%p | 분류용도 ko-sbert 우수 |

**핵심 교훈**: Architecture와 학습 기법 탐색은 v46에서 끝났음.  
다음 개선은 **데이터 품질**과 **운영 인프라**에 있음.

## 참고 문서

- [`EXPERIMENT_SUMMARY.md`](EXPERIMENT_SUMMARY.md) — v1-v68 전체
- [`MODEL_CARD.md`](MODEL_CARD.md) — 공식 모델 카드
- [`DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) — 배포 가이드
- [`KNOWN_FAILURES.md`](KNOWN_FAILURES.md) — 실패 패턴
