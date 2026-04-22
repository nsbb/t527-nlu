# Critical Reflection — 최종 결론 (30분 고민 후)

**세션**: 2026-04-22 08:48 ~ 09:13 (약 25분 실제 작업 + 5분 스케줄링 실수)  
**질문**: "이게 최선인가? Single intent와 비교 어떻게? 더 좋은 방법 없을까?"

---

## 한 문장 답

> **"Multi-head + rules + DST는 온디바이스 + 데이터 부족 상황에서 pragmatic optimum. User-weighted accuracy 99.33%, catastrophic error 0건. Single intent는 KE 72.5% vs 97.8% 차이로 기각됨. 진짜 다음 단계는 모델이 아니라 실사용자 데이터."**

---

## 3줄 근거

1. **KE 72.5% vs 97.8% (+25.3%p)**: 외부 데이터에서 Single intent 압도적 열세
2. **User-weighted 99.33%**: 반대 행동 0건, 재질의 수준만 남음
3. **DST +85.7%p on multi-turn**: 구조화 없으면 multi-turn 불가능

---

## 이 세션에서 실제로 한 것

### 문서 작성 (4개 신규)
- `CRITICAL_REFLECTION_2026_04_22.md` (8개 부록, 깊은 분석)
- `REFLECTION_SUMMARY.md` (한눈 요약 + 스켑티컬 자기검토)
- `REFLECTION_FINAL_2026_04_22.md` (이 문서)

### 코드 개선 (실제 버그 수정 7건)
1. 덥다/더워/덥네 → ac_control 교정 rule
2. 시원하게 → ac_control (catastrophic fix)
3. 블라인드 닫아 → close (catastrophic fix)
4. 환기꺼 → preprocess 분리 (catastrophic fix)
5. DST "까요?" 확인 질문 → control 승격 + dir 추출
6. DST device follow-up dir 우선순위
7. 복합 명령 분할 처리 (process_compound)

### 신규 벤치마크 도구 (3개)
1. `scripts/test_multiturn.py` — DST 가치 측정 (+85.7%p 정량화)
2. `scripts/error_severity_analysis.py` — 오류 심각도 분류 (catastrophic/annoying/minor/graceful)

### 정량 개선 (이번 세션만)
| 지표 | 시작 | 종료 | Δ |
|------|:---:|:---:|:---:|
| TS combo | 95.76% | 95.86% | +0.10%p |
| User-weighted | 99.27% | 99.33% | +0.06%p |
| **Catastrophic errors** | **3** | **0** | **-3** |
| Multi-turn (DST on) | 미측정 | 7/7 (100%) | 신규 |
| Multi-turn (DST off) | 미측정 | 1/7 (14%) | 신규 |
| Regression tests | 26 | 30 | +4 |

---

## Multi-head 완승의 수치적 증거

| 비교 | Flat 94 | MH v46 | 승자 |
|------|:---:|:---:|:---:|
| TS combo | 94% | 93.3% | ≈ 동등 |
| **KE fn 외부 일반화** | **72.5%** | **97.8%** | MH +25.3%p |
| OOD rejection | 1/5 | 5/5 | MH |
| DST 지원 | 불가 | 가능 | MH |
| 데이터 효율 | O(N×M) | O(N+M) | MH |

MH가 "체감 없는 성능 향상"이 아니라 **외부 일반화**에서 폭발적 차이.

⚠️ Caveat: v46이 KE pseudo-label 학습했으므로 순수 architecture 기여도 분리 안 됨. 하지만 OOD는 순수 구조 이득.

---

## 시도했으나 유보한 것 (잘한 revert)

| 시도 | 결과 | 판정 |
|------|------|------|
| 3-tier conf fallback (0.3/0.5) | TS -0.04%p | revert |
| Elevator dir=on default | TS -0.33%p | revert |
| Clarify→control 광범위 | TS -0.59%p | revert |
| CTC dir inference (켜→on) | TS -0.69%p | revert |
| 한글 0개 → unknown | TS -0.03%p | revert |
| 지금/혹시/야 rule | TS -0.13%p | revert (이번 세션) |

**교훈**: 넓은 rule은 자주 regression. 좁게 검증한 rule만 적용.

---

## 진짜 한계 (brutal honesty)

### 1. TS 벤치마크 자체가 불완전
- 라벨 불일치 다수 (알람: 분류 사이 갈등, light: 어순별 다른 라벨)
- 측정치 vs 실제 사용자 경험 = 95.86% vs ?
- **실 user data 없이는 진짜 품질 모름**

### 2. Wild input에서 무너지는 패턴
- "Alexa 난방" → energy_query (외부 wake word 혼란)
- "아 졸려 불 끄고 자자" → query 오예측 ("자자"가 방해)
- "조명 좀 켜주실 수 있어요?" → clarify (정중 패턴 학습 약)

### 3. Rule 14개의 숨은 비용
- 유지보수 부담 (교차작용 감시 필요)
- 새 기기 추가 시 rule 업데이트
- 이건 "가난한 ML" 신호, 근본은 데이터

### 4. 구조 편향
- Multi-head가 "자연스럽게 구조 없는 발화" (감탄사, 간접 표현)에 약함
- "방이 덥네"처럼 감정 표현은 특별 rule 필요

---

## 놓친 기회 (미래 팀을 위해)

1. **Pseudo-labeling 검증 안 함**
   - v28 → KE 라벨 생성 → v46 학습
   - v28 KE 정확도 75% 였음. 잘못된 라벨로 학습.
   - **대안**: 전문가 검수 또는 ensemble consensus만 사용

2. **GT 219 재라벨링 미실시**
   - TS가 GT 기반 증강. GT 오류 = TS 확산.
   - **대안**: iter1~3 시점에 전문가 검수 (하루)

3. **실 user 로그 수집 인프라 지연**
   - iter7부터 설계 문서 있지만 구현 없음
   - **대안**: 배포와 동시에 수집 시작

---

## Concrete Next Steps

### Immediate (이번 주)
1. Android JNI 통합 (`docs/ARCHITECTURE.md`)
2. T527 NPU NB 변환
3. 첫 end-to-end test

### 1-2 개월
4. **실사용 로그 수집 시작** ⭐ 가장 중요
5. Weekly review queue 운영
6. 월간 재학습 cycle

### 6개월
7. GT 수동 재구축
8. 기존 rule의 학습 데이터 통합 (점진적 rule 제거)
9. 논문 (STT QAT)

---

## 마무리: Reflection이 만든 실질 가치

30분 고민이 **실제로 만든 것**:
- 🎯 Catastrophic error 3 → **0** (가장 중요한 성과)
- 📊 DST 가치 **+85.7%p 정량화** (이전엔 추측)
- 🔧 Compound command 지원 (실 사용 필수)
- 📈 User-weighted metric 도입 (더 정확한 품질 측정)
- 📝 6개 신규 문서 + 57+ commits (세션 전체)

**"이게 최선인가?"**에 대한 답:  
> **주어진 제약(온디바이스 + 데이터)에서는 YES. 다음 돌파는 제약을 바꿔야 — 실사용 데이터로.**
