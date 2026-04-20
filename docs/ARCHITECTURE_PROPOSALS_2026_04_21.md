# 현재 아키텍처 문제 분석 + 개선 제안

작성: 2026-04-21 (토론 후)
상태: 실험 단계 제안

---

## 1. 현재 아키텍처 근본 문제 목록

### 🔴 Critical (해결 필수)

#### P1. 라벨이 규칙 기반 guess
- 엑셀 GT는 발화+응답뿐, 멀티헤드 라벨 없음
- `parse_gt_scenarios.py` 규칙이 60~70% 신뢰도
- 학습 데이터 1,176건 확실한 오류, 2,407건 의심
- **모든 후속 벤치마크가 이 위에서 자기복제**

#### P2. 라벨 오류의 재귀 전파
- v28이 잘못된 라벨로 학습
- v28이 KoELECTRA 13K를 pseudo-label (같은 오류 기준)
- v46이 그걸 학습 → 같은 오류 고착
- **"KE 97.8%"도 잘못된 기준에 맞춘 것**

#### P3. 학습 데이터 = 규칙 기반 증강
- 219 시나리오 × 평균 25배 증강 = 24.5K
- "우리가 생각한 변형"만 포함
- 실사용자 실제 발화 패턴과 괴리

### 🟡 Major (큰 영향)

#### P4. Value 추출이 Regex 의존
- "23도" regex로 잡음 → "이십삼도"는 preprocess 없으면 실패
- "거실 23도, 안방 25도" 같은 복합 명령 → room 하나만
- 자연어 시간 표현 ("내일 오전 7시") 한계

#### P5. DST 기본 수준
- 10초 timeout, room/device/confirm 패턴만
- 5턴 이상 복잡 문맥 처리 불가
- Slot carry-over 미비 (이전 값 유지 등)

#### P6. Multi-head negative transfer 가능성
- fn head와 exec head가 backbone 공유
- Loss weighting(fn×2, exec×2, dir×1.5 ...)으로 대응
- 하지만 이론적 최적 아님

#### P7. 복합 명령 표현 불가
- "거실 에어컨 23도, 안방 에어컨 25도" — 현재 구조로는 하나만
- 한 발화에 여러 action 처리 불가
- BIO slot 있으면 가능하지만 구현 안 함

### 🟢 Minor (개선 여지)

#### P8. Ensemble 배포 부담
- 단일 ONNX 105MB (v28 + v46 내장)
- T527 NPU에는 아직 변환 안 됨 (CPU ONNX만)
- Latency 0.48ms로 좋지만 메모리 부담

#### P9. 실사용 피드백 루프 없음
- Production 배포 후 잘못된 케이스 수집 경로 없음
- 개선이 수동 발견에만 의존

---

## 2. 개선 방향별 Option 분석

### Option A: 데이터 품질 프로젝트 (근본 해결)

```
Phase 1: 엑셀 219 시나리오 수동 라벨링 (2일)
  - 각 시나리오 × 5 head = 1,095 라벨 결정
  - 발화-응답 쌍 보고 의미적으로 올바른 라벨
  - 결과: 진짜 GT 확보

Phase 2: 학습 데이터 스팟 검증 (3일)
  - 24.5K 중 500 샘플링
  - 수동으로 검증
  - 규칙 기반 자동 수정 + 수동 검토 결합

Phase 3: KoELECTRA val 정제 (1일)
  - 1,536개 중 의심 케이스 수동 재라벨
  
Phase 4: 재학습 (2시간)
  - v46 recipe로 재학습
  - 진짜 벤치마크 성능 측정
```

**예상 효과**:
- 진짜 모델 성능 측정 가능
- KE 97.8%가 과대평가인지 검증
- 실제는 95~97%? (추정)

**리스크**:
- 수동 라벨링 일관성 문제
- 시간 비용 (총 1주+)

### Option B: Retrieval-Augmented NLU (창의적)

**아이디어**: 학습 기반 분류 대신 **검색 기반**으로 전환.

```
입력 발화
  ↓ Sentence embedding (ko-sbert)
  ↓
GT 219개 + 사용자 검증 샘플 pool
  ├─ Top-K cosine similarity
  │
  ├─→ If max_sim > 0.85: 해당 GT의 라벨 그대로 사용
  │                      (OOD 거부도 자연스러움: 유사한 GT 없으면 unknown)
  │
  └─→ If 0.5 < max_sim < 0.85: Multi-head classification (fallback)
      If max_sim < 0.5: unknown → 서버
```

**장점**:
- **라벨 품질에 덜 의존** — GT pool만 좋으면 됨
- **새 시나리오 추가 쉬움** — pool에 넣으면 됨, 재학습 불필요
- **OOD 거부 자연스러움** — 유사 없으면 거부
- **설명 가능** — 왜 이 라벨인지 ("이 GT와 유사해서")

**단점**:
- Retrieval 비용 (하지만 219개는 작음)
- Threshold 튜닝 필요
- Long-tail 처리 부족

**구현 복잡도**: 낮음 (1~2일)

### Option C: Value Pointer Head 추가 (기존 handoff 계획)

원래 설계(`handoff_semantic_action_parser_complete.md`)에 있던 것:

```
현재: fn + exec + dir + param_type + judge (모두 classification)

추가: has_value + start_ptr + end_ptr (token 위치)
```

**효과**:
- "이십삼도", "두시간 후" 같은 자연어 값 추출
- 복합 값 ("23도로 1시간 예약") 동시 처리
- Regex 보완

**장점**:
- Regex 한계 극복
- 복합 발화 대응력 ↑

**단점**:
- 토큰 위치 라벨링 필요 (annotation cost)
- 모델 확장

**구현 복잡도**: 중간 (3~5일)

### Option D: Intent + BIO Joint Model (업계 표준)

Multi-head를 버리고 표준 NLU로:

```
Input → Encoder
   ├─→ Intent head (classification, flat 60~100개)
   └─→ Slot head (BIO tagging per token)
```

**장점**:
- 업계 표준 (ATIS, SNIPS 벤치마크 스타일)
- JointBERT 등 검증된 접근
- 복합 명령 자연스러움

**단점**:
- **우리 시스템 전면 재작성**
- 모든 데이터 BIO 재라벨링 (비쌈)
- NPU 호환성 재검증

**구현 복잡도**: 매우 높음 (2주+)

### Option E: LLM 기반 접근

Small LLM (1B 이하)로 zero/few-shot NLU.

```
Input + Few-shot examples → LLM → structured output (JSON)
```

**장점**:
- 가장 유연
- Compositional generalization 우수
- 실시간 GT 추가 가능

**단점**:
- **T527 NPU 배포 불가** (100MB+ 제한 초과)
- Latency 수백 ms
- 현재 하드웨어 범위 밖

**구현 복잡도**: 매우 높음

### Option F: Real User Log Feedback Loop

```
Production 배포
  ↓
모든 발화 + 예측 + confidence 로그
  ↓
Low-confidence 샘플 리뷰 큐
  ↓
수동 라벨링 (주 1회 100개)
  ↓
월간 재학습
```

**장점**:
- **진짜 일반화 데이터 확보**
- 지속적 개선
- 실사용자 피드백

**단점**:
- 배포 필요
- 인프라 구축
- 초기엔 효과 없음

**구현 복잡도**: 중간 (+ 시간)

---

## 3. 추천 전략 (우선순위)

### 🥇 단기 (1~2주) — Retrieval Hybrid (Option B)

**이유**:
- 라벨 품질 문제 즉시 완화 (GT pool만 깨끗하면 됨)
- 구현 간단
- 기존 모델 재학습 불필요
- OOD 거부 강화

**구현 단계**:
1. Option A Phase 1 (수동 GT 219 라벨링) 먼저 완성
2. 검증된 GT를 retrieval pool로
3. ko-sbert embedding precompute
4. Inference 시 cosine similarity + fallback

**예상 효과**:
- GT 분포 근처 발화 → 라벨 품질 ↑↑
- OOD 거부 정확도 ↑
- "안 본 조합" 처리는 multi-head fallback 유지

### 🥈 중기 (1개월) — Value Pointer + Real-world Data (C + F)

**이유**:
- 복합 명령 대응 (value pointer)
- 실사용 데이터로 진짜 일반화 확보
- 기존 multi-head와 호환

**구현**:
1. Value pointer head 추가
2. 219 GT에 value span 주석 (수동)
3. 재학습 + 평가
4. 동시에 production 배포 + 로그 수집

### 🥉 장기 (분기) — 진짜 GT 프로젝트 (Option A 전체)

**이유**:
- 모든 측정 지표의 신뢰도 회복
- Ablation 실험 가능 (진짜 GT로 multi-head vs flat 공정 비교)
- 학술적 근거 강화

### ❌ 비추천

- **Option D (전면 재작성)**: ROI 낮음, 리스크 큼
- **Option E (LLM)**: 하드웨어 제약

---

## 4. 실험 우선순위 매트릭스

| 제안 | 효과 | 비용 | 리스크 | 추천도 |
|------|:---:|:---:|:---:|:---:|
| **A1 (GT 219 수동 라벨)** | 🟢 High | 🟡 Med | 🟢 Low | ⭐⭐⭐ |
| **B (Retrieval hybrid)** | 🟢 High | 🟢 Low | 🟢 Low | ⭐⭐⭐ |
| **C (Value pointer)** | 🟡 Med | 🟡 Med | 🟡 Med | ⭐⭐ |
| **F (Log feedback)** | 🟢 High | 🟡 Med | 🟢 Low | ⭐⭐ |
| **D (BIO joint)** | 🟡 Med | 🔴 High | 🔴 High | ⭐ |
| **E (LLM)** | 🟢 High | 🔴 High | 🔴 High | ❌ |

---

## 5. 구체적 실행 계획 (다음 단계)

### Week 1: Ground Truth 확보 + Retrieval Prototype

**Day 1-2**: 엑셀 219 시나리오 수동 재라벨링
- 각 발화-응답 쌍 보고 5 head 라벨 결정
- 산출: `data/gt_219_manual_v2.json`

**Day 3**: Retrieval 인덱스 구축
- ko-sbert embedding 219 × 768d precompute
- FAISS 또는 numpy cosine
- 산출: `data/gt_embeddings.npy`

**Day 4**: Retrieval + Multi-head 하이브리드 추론 구현
```python
def predict(text):
    emb = sbert.encode(text)
    top_k = cosine_top_k(emb, gt_embeddings, k=5)
    
    if top_k[0].score > 0.85:
        return top_k[0].labels  # 검증된 GT 라벨 그대로
    elif top_k[0].score > 0.5:
        mh_pred = multi_head_predict(text)
        return combine(top_k[0].labels, mh_pred)  # 혼합
    else:
        return {'fn': 'unknown'}
```
- 산출: `scripts/retrieval_hybrid_inference.py`

**Day 5**: 평가 + Ablation
- 219 GT에서 self-retrieval 정확도
- KoELECTRA 1,536에서 retrieval+fallback 성능
- v46 단독 vs retrieval-hybrid 비교

### Week 2: Value Pointer + Ablation Study

**Day 6-7**: Value pointer head 추가
- `model_cnn_multihead.py`에 head 추가
- Training loss에 포함

**Day 8-9**: 219 GT value span 주석
- "23도" → start=7, end=9 같은 위치

**Day 10**: Ablation 실험
- Flat v18 + KE pseudo-label 재학습
- v46 vs 공정한 flat 비교
- **"multi-head 구조 자체의 기여도" 드디어 측정**

### Week 3+: Real-world Pipeline

- 로그 수집 인프라
- 주간 리뷰 프로세스
- 월간 재학습 자동화

---

## 6. 핵심 질문들 (답 필요)

1. **GT 수동 재라벨링에 누가 시간 쓸 건가?** (전문가 필요)
2. **Retrieval threshold** 최적값? (실험으로 결정)
3. **Value pointer annotation** 얼마나 필요? (219 × 평균 1.5 value = ~330)
4. **Production 배포 일정**과 로그 수집 인프라 연계?
5. **Ablation 결과로** multi-head 결정 재검토?

---

## 7. 가장 중요한 메시지

**현재 벤치마크 수치는 신뢰 제한적**. 숫자 개선보다:

1. **진짜 GT 확보** (데이터 품질)
2. **Retrieval로 라벨 의존도 감소** (아키텍처 유연성)
3. **실사용 피드백** (진짜 일반화)

이 세 개가 **숫자보다 훨씬 중요**. 새 기법/아키텍처 실험은 이게 안 되면 의미 적음.

**"모델은 이미 충분히 좋다. 데이터가 부족하다."**
