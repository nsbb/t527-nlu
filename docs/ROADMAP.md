# NLU Roadmap — 향후 개선 방향

## 현재 상태 (2026-04-22 업데이트, iter9 완료)

✅ **달성**:
- Ensemble ONNX 배포: `nlu_v28_v46_ensemble.onnx` (변경 없음)
- 14개 모델/레시피 실험 전부 실패 확인 → **현 데이터로 수학적 ceiling**
- ONNX FP16 변환 가능 (GPU/NPU용, CPU는 비효율)
- 배포 체크리스트 + 피드백 시스템 설계 완료
- **iter9 후처리 rule 14개 + DST 고도화 + 배포 인프라 완성**

🔴 **핵심 결론**:
모델 레벨 실험 완전 소진. **진짜 개선 경로는 실사용 데이터뿐** (P3 #9).

**iter 9 성과 (2026-04-21 22:18 ~ 2026-04-22 06:45, ~8.5시간)**:
- 후처리 rule 14개 적용 → **Ensemble + rules TS: 93.53% → 95.76% (+2.23%p)**
- DST 고도화: slot filling, 5-turn history, correction 확장
- DeploymentPipeline 클래스 완성 (end-to-end 통합)
- Preprocess 210+ entries (2-pass, 사투리/존댓말/영어 포함)
- 규제 test 26개, GT 95.0%, Latency 0.67ms
- 문서 9개 업데이트 + 3개 신규 (ARCHITECTURE, API_USAGE, SCRIPTS_INDEX)
- **38+ commits GitHub main 푸시 완료**

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

#### 3. ~~Response 템플릿 상황별 다양화~~ ✅ **완료 (iter8)**
- **현황**: fn당 1~2개 템플릿 (반복적)
- **개선**: fn당 2~3개 variation, `random.choice`
- **적용**: `scripts/sap_inference_v2.py` (2026-04-21 iter8)
- **TS 영향 없음** (라벨 유지), UX 자연스러움 증가
- **확장 아이디어 (미구현)**: 시간대/상황별 (아침/밤, 여름/겨울) — 실사용 로그 기반 추후 구현

#### 4. ~~Confidence 기반 세밀한 fallback~~ ⚠️ iter8 검증 — 유보
- **현황**: conf<0.5 → unknown
- **시도**: conf 0.3-0.5 → clarify (재질의), conf<0.3 → unknown
- **결과**: TS -0.04%p (측정 손실), UX 이득은 실사용 없이 불확실
- **결정**: 실사용 로그 수집 후 실제 low-conf 분포 확인 뒤 재검토

### 🟡 P2 — 중간 작업 (1~3일)

#### 5. ~~학습 데이터 라벨 오류 체계적 수정~~ ❌ **검증됨 (2026-04-21)**
- 1,180건 수정 후 v46 recipe full retrain (v70) → TS -3.16%p regression
- 원인: 부분 수정이 KoELECTRA pseudo-labeled 나머지와 충돌 → 일관성 파괴
- **완전한 재라벨링(전체 일관된 규칙)**이 아니면 불가능
- iter 8: GT v2 재파싱 → 219개 중 **3개만 변경** (1.4%). 영향 미미.
- → 진짜 GT 수동 재구축 (P3)으로 통합 이동

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

#### 9. 실제 사용 로그 수집 파이프라인 ⭐ **최고 우선순위 (모델 실험 소진 후)**
- **목표**: production 발화 로그 → feedback loop
- **상세 설계**: [`docs/FEEDBACK_SYSTEM_DESIGN.md`](FEEDBACK_SYSTEM_DESIGN.md) 참조
- **구조**:
  ```
  User 발화 → NLU → [Log + Confidence]
    ↓ conf<0.7 or manual correction
  Review Queue → 라벨링
    ↓
  학습 데이터 증강 → 월간 재학습
  ```
- **인프라 필요**: 로깅 서버, 익명화, review UI
- **예상 비용**: ~$200 + 0.25 FTE / 월
- **예상 개선**: 6개월에 93.6% → 95%+

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
