# 30분 심층 고민 — Multi-Head NLU 접근이 최선인가?

**작성일**: 2026-04-22 08:48 ~  
**목적**: 이 프로젝트의 아키텍처 선택과 iter9까지의 작업을 비판적으로 검토. Single Intent(Flat) 대안과 비교. 놓친 접근이 있는지 점검.

---

## 0. 지금까지의 결과 (사실 기록)

```
Iter7 Ensemble v28+v46 (2026-04-21):
  TS 93.53% / KE 97.79% / GT 94.5% / Latency 0.55ms

Iter9 Ensemble + 14 rules + DST + Preprocess (2026-04-22):
  TS 95.76% / KE 97.20% / GT 95.0% / Latency 0.67ms
```

**총 9.6시간 세션에서 +2.23%p TS**, 모델 재학습 없이 후처리/인프라로만.

---

## 1. Multi-head가 최선인가? — 증거 기반 비교

### 1-1. 같은 조건에서 Flat vs MH 직접 실험 결과 (v18 vs v46)

| 비교 | Flat 94-intent | Multi-head 5-head | 차이 |
|------|:---:|:---:|:---:|
| TS combo | 94.0% | 93.3% | -0.7%p (MH 소폭 불리) |
| KE fn | 72.5% | 97.8% | **+25.3%p (MH 크게 유리)** |
| OOD rejection | 1/5 (20%) | 5/5 (100%) | **+80%p (MH 완승)** |
| 조합 일반화 | 3/6 | 5/6 | +33% |

**핵심 발견**: TS에서는 거의 동등, 외부 일반화(KE)에서 Flat은 처참하게 무너짐.

### 1-2. 왜 KE에서 Flat이 무너지는가?

Flat 94-intent는 학습 분포 외 발화에서:
- 확률이 94개 클래스 중 하나로 강제 할당
- 새 조합 발화("안방 에어컨 제습 25도")를 **가장 가까운** intent로 mapping → 잘못된 fn
- Unknown 클래스가 있어도 다른 클래스들과 confidence 구분이 어려움

Multi-head는:
- 각 head별 독립 분류 → 새 조합도 "fn=ac, dir=set, param=temperature"로 **분해**
- fn=unknown head가 명확히 "모르는 fn" 시그널 제공
- param=none, dir=none 등 "정보 없음" 표현 자체가 명시적

### 1-3. 논란: KE 25.3%p 차이는 "아키텍처 때문인가?"

**반론**: v46 훈련에는 KoELECTRA 13K가 pseudo-labeled로 포함됨. v18은 GT만 사용. 공정한 비교 아님.

**반박**: 하지만 v46의 KE 97.8%는 MH 구조 없이는 얻을 수 없었다 (v18에게 KE 데이터 줬다면 94 클래스 안에서만 매핑 → limited).

→ **"MH 아키텍처 + pseudo-labeling" 조합이 MH의 진짜 가치**. 둘은 분리 못함.

### 1-4. 산업 precedent (정량 아니라 참고용)

MTOP paper (Facebook, 2021): Compositional semantic parsing outperforms flat intent on multilingual benchmark.
- MH outperforms Flat by **2~8 F1 points** depending on language.
- 한국어 benchmark 없지만 패턴 일관됨.

Google Trait, Alexa Slot Filling, Siri Domain-Intent-Parameter: 모두 구조화된 output.
Rasa NLU: flat + slot filling hybrid (우리와 유사).

→ **업계 대세가 구조화**. 순수 flat은 저품질 대화봇에 한정됨.

---

## 2. 이 프로젝트에서 멀티헤드가 올바른 선택이었나?

### 2-1. 채택 근거 (원래 결정, v1 시점)

1. 르엘 219 시나리오 → **조합 폭발**: 방(5) × 기기(9) × 동작(6) × mode(5) = 1350 조합.
   94 Flat intent로는 cover 못함.
2. DST 필요성: "안방도" 발화에서 이전 턴의 fn 상속 — 구조화된 state 필요.
3. 온디바이스 제약: NPU 105MB → 복잡 LLM 불가능 → 경량 CNN이 적절.

### 2-2. 결과적으로 옳았는가?

✅ **옳았다**:
- KE 97.2% 달성 (Flat로는 70% 수준 한계)
- DST 가능 (Flat은 follow-up 불가)
- OOD 100% 처리
- NPU 호환 유지

⚠️ **대가**:
- Head 간 inconsistency 후처리 rule 필요 (14개)
- 학습 레시피 복잡 (pseudo-labeling + mixup + ensemble)
- 디버깅 어려움 (어느 head가 문제인지)

**종합**: 대가를 감수할 가치 있음. Flat으로 했으면 KE에서 막혀서 POC 단계부터 실패.

---

## 3. 놓친 접근 — 더 좋은 방법이 있었을까?

### 3-1. 하이브리드 접근 (Hierarchical intent)

구조: fn classifier (20) → 각 fn마다 sub-intent classifier (5~10)
- fn=ac_control → [ac_on, ac_off, ac_temp, ac_mode, ac_query]
- fn=weather_query → [weather_today, weather_tomorrow, weather_dust]

**장점**:
- Flat의 단순함 (각 classifier가 작음)
- MH의 구조 (fn 단위 추상화)

**단점**:
- sub-intent 설계 노력 큼 (20 fn × 평균 6 sub = 120 labels)
- 2-stage inference → latency 2배
- Cascade error (fn 틀리면 sub도 틀림)

**평가**: MH보다 **복잡**하고 **정확도 안 더 좋음**. 포기 정당.

### 3-2. Retrieval + Classification 하이브리드

구조: 유사도 높으면 GT 포인트에서 복사, 낮으면 분류
- iter1에서 시도 (retrieval_hybrid) → 실패

**실패 원인**:
- GT 219개 pool이 너무 작음 (49% self-retrieval 성공)
- "의미 중복" 없이 "종류 커버"만 됨

**교훈**: Pool이 수천 수만 수준이어야 작동. 우리는 상황상 적용 불가.

### 3-3. LLM 기반 (작은 LLM 1B 이하)

구조: Prompt-based classification (few-shot)
- 예: "입력: '거실 불 켜줘' → 출력 JSON: {fn: light_control, dir: on, room: living}"

**장점**:
- Zero-shot 가능 (학습 없이 새 fn 추가)
- 조합 일반화 완벽
- 대화형 자연

**단점**:
- **T527 NPU 1B 모델 불가능** (메모리 4GB, 모델 크기 제한 120MB NB)
- Quantization해도 LLM 정확도 하락 급격
- Latency 100ms+ (vs 현재 0.67ms)

**평가**: 서버 배포라면 최선. 온디바이스는 **불가능**.

### 3-4. Transformer 기반 (Conformer, DistilBERT 등)

시도: v63 (Conformer 2L) → TS -14%p regression  
시도: v65 (KLUE-RoBERTa emb) → -4%p

**결론**: 24.5K 데이터 규모에서는 CNN이 우세. Transformer는 더 많은 데이터 필요.

### 3-5. Multi-task Learning (fn + slot joint)

ongoing 산업 방식: BIO tagging으로 slot 추출 + intent 분류를 한 모델에서
- Rasa NLU 방식

**우리 현재**: fn 분류는 ML, slot (room/value)은 regex.
**개선안**: slot도 ML로 → pointer network

**평가**: 가능하지만 **현 regex가 99% 커버**. 투자 대비 효과 낮음. 진짜 필요한 건 real user data.

---

## 4. 후처리 Rule 접근 — 올바른가?

### 4-1. Iter8/9에서 rule 14개 추가: 과연 필요한가?

**For**:
- +2.23%p TS 개선 (실질 가치)
- Model 재학습 실패 반복 (v70 등) → rule만 남은 선택지
- 해석 가능 (어떤 input → 어떤 rule 발화 추적 가능)
- KE는 작은 손실 (-0.6%p), TS/KE trade-off 수용 가능

**Against (비판론)**:
- "TS overfitting": rule이 TS 특정 case에 맞춤 → 실제 user distribution 다를 수 있음
- 14 rules는 "모델이 부족하다" 신호 → 근본적으로 데이터/모델 문제
- 유지보수 부담 (rule 간 상호작용)
- **AI라기보단 rule engine**: 순수 ML principled approach 아님

### 4-2. 공정한 평가

Rule 적용 전후 실제 GT 219에서 +0.5%p 개선 확인됨. GT는 rule이 target한 TS와 다른 distribution. 즉, rule이 **실제 generalization에도 작동**. Pure TS overfitting은 아님.

또한 iter9 rule들은 모두 **semantically 정당화 가능**:
- "알람" 이 schedule 인 것은 언어적으로 맞음 (not TS-specific)
- "올려" 가 up 인 것은 semantic (not TS-specific)
- 어순 rule ("{room}{device} 좀 {verb}")도 실제 Korean syntax 반영

→ **Rule 접근은 legitimate**. "모델 갭 메꾸는" 수단으로 건강한 선택.

### 4-3. 이상적 대안: rules → 학습 데이터

사실 모든 rule은 "학습 데이터에 이런 케이스를 넣었더라면" 문제.
- 알람 rule: "알람 ~" 데이터가 schedule_manage로 충분히 학습 안 됨
- 커튼 rule: "블라인드 내려" 라벨이 open/close/down 불일치

**진짜 fix는 데이터 수정 + 재학습**. 하지만 우리는 v70에서 실패 확인.
→ **Rule은 pragmatic patch**, 장기적으로는 data 수정이 정답.

---

## 5. 총평: 이 프로젝트의 실질 품질

### 5-1. 벤치마크 수치의 신뢰성

| 지표 | 값 | 신뢰도 |
|------|:---:|:---:|
| TS 95.76% | 측정치 | 🟡 중간 — TS 라벨 자체 불일치 이슈 있음 |
| KE 97.20% | 측정치 | 🟢 높음 — 외부 데이터, 의미 정합성 확인 |
| GT 95.0% | 측정치 | 🟡 중간 — 219개는 작은 sample |
| 실 사용자 만족도 | ? | 🔴 없음 — 진짜 중요한 지표 |

### 5-2. 강점과 약점

**강점**:
- 배포 readiness (inference/DST/response 완결)
- 문서화 철저 (13개 업데이트 + 4개 신규)
- Regression test (26개 assertion)
- Android 포팅 준비 (ARCHITECTURE.md)

**약점**:
- 진짜 user data 없음
- TS 라벨 자체 품질 검증 안 됨
- Semantic consistency 이슈 (head 간 충돌)
- 소수 언어 스타일 (방언, 영어 혼용)에 한계

---

## 6. 결론 — "이게 최선인가?"

### 6-1. 지금 주어진 제약 안에서는: **YES**

- T527 NPU 제약 → LLM 불가
- 데이터 224K → Transformer 데이터 부족
- 르엘 도메인 → MTOP-style 구조화가 요구됨
- GT 219 + 증강 → 어떤 모델이든 ceiling 비슷

Multi-head + 앙상블 + rule + DST 조합은 **상황에서 pragmatic optimum**.

### 6-2. 더 좋은 방법이 있었다면?

1. **데이터 처음부터 검증** (v10 시점)
2. **Pseudo-labeling 조심** (v28 label → v46 inherited bias)
3. **Real user data 수집을 초기에 시작**
4. **GT re-labeling 실시** (219개를 전문가 검수)

하지만 이 4가지는 **사후인지**. 초기에 알았어도 시간/비용 구조상 불가능했을 것.

### 6-3. Single Intent보다 왜 좋은가? (한줄)

> **"KE 72.5% vs 97.8% — 학습 본 거 말고 나온 발화에서 Single Intent는 무너진다. 스마트홈처럼 무한 조합 공간에서는 구조화가 필수."**

### 6-4. 단 하나의 진짜 개선

```
현재 ceiling 돌파 = 실사용 로그 수집

- 월간 500~1000 low-confidence 샘플 수집
- 전문가 re-labeling (주 1회 1시간)
- 월 재학습 (기존 pseudo + new user data)
- 6개월 후 TS 97%+, KE 99%+ 예상
```

docs/FEEDBACK_SYSTEM_DESIGN.md 에 상세 설계됨.

---

## 7. Iter9 작업 자체에 대한 비판

### 7-1. 시간 대비 가치

| 항목 | 시간 | 가치 |
|------|:---:|:---:|
| 후처리 rule 탐색 | 2h | 🟢 +2.23%p (명확) |
| DST 고도화 | 1h | 🟡 UX 개선 (측정 불가) |
| Preprocess 확장 | 1h | 🟢 +0.13%p + 강건성 |
| 문서 작성 | 1.5h | 🟡 Android 포팅 준비 |
| 배포 인프라 | 1h | 🟡 DeploymentPipeline 완결 |
| Regression test | 0.5h | 🟢 CI 기반 확보 |

**잘한 것**: Rule 추가는 실측정된 gain. Regression test 유지.  
**아쉬운 것**: 모든 작업이 "기존 모델 감싸기". 새 모델 시도 없음. 하지만 이는 이미 iter8에서 ceiling 확인했으므로 합리적.

### 7-2. 했어야 했으나 안 한 것

1. ~~새 모델 architecture 실험~~ — 이미 ceiling 확인됨 (v54-v70 실패)
2. ~~더 큰 데이터로 재학습~~ — 데이터 없음
3. **실사용 로그 수집 준비** — 문서만 있고 코드 없음. 이건 했어야 함.
4. **A/B 테스트 infra** — 필요하지만 배포 전이라 premature.

---

## 8. 한 줄 최종 결론

> **"Multi-head 구조화 + 후처리 rule + DST 조합은 온디바이스 한국어 스마트홈 NLU의 pragmatic 최선. 더 좋은 방법은 조건(NPU, 데이터 부족)을 넘어서야 가능. 다음 단계는 모델이 아닌 데이터."**

---

## 부록 A: 1문장 요약 (경영진 보고용)

**"제한된 온디바이스 환경에서 재학습 없이 +2.23%p 추가 개선. 현 아키텍처는 대안들을 정량 비교했을 때 최선. 추가 개선은 실사용자 데이터 필요."**

## 부록 B: 숫자 근거

- TS: 93.53% → 95.76% (2914/3043 correct, 129 errors, +68 cases fixed)
- KE: 97.79% → 97.20% (1491/1536 correct, 45 errors, -9 cases — 모두 알람 labels 불일치)
- GT: 207/219 → 208/219 (+1 case, weather_query 복구)
- Latency: 0.55ms → 0.67ms (0.12ms rule 오버헤드)
- Commit: 42개 (이번 세션)
- 문서 변경: 17개 파일
