# Iteration 9 — 심층 에러 분석 + 후처리 rule 집중 개선 (2026-04-21 22:18-)

## 배경

Iter 8에서 후처리 rule만으로 TS +0.85%p 달성 확인. 남은 162 오류(5.32%) 심층 분석 → 패턴별 타겟 룰 추가.

## 진행 (22:18~22:35 시점)

### 적용된 rule (9개 commit)

| # | 규칙 | TS 영향 | KE 영향 |
|---|------|:---:|:---:|
| 1 | `{room} {device} 좀 {verb}` 어순 CTC 복구 | +0.20%p | 0 |
| 2 | curtain 올려 → up / 블라인드 내려 → close | +0.06%p | 0 |
| 3 | 블라인드 → 커튼 preprocess 제거 (sap 일관성) | 0 | 0 |
| 4 | heat_control CTC + none → on | +0.10%p | 0 |
| 5 | 화면/월패드/알림/음량 → home_info (capability query 제외) | +0.20%p | 0 |
| 6 | 커튼 pred=open 확장 (up/close/stop) | **+0.62%p** | 0 |
| 7 | 블라인드 no-action → stop | (포함) | 0 |
| 8 | 전화 entity-less → unknown | +0.10%p | 0 |
| **총** | **9개 규칙** | **+1.28%p** | **0** |

## Metric

| 평가 | Iter8 후 | Iter9 후 | Δ |
|------|:---:|:---:|:---:|
| TS combo | 94.38% | **95.66%** | **+1.28%p** |
| KE fn | 97.20% | 97.20% | 0 |
| GT 219 | 95.0% | 95.0% | 0 |

## 핵심 발견

### 1. 어순이 라벨을 결정

"{room} 좀 {device} {verb}" (adverb before device) → **clarify**
"{room} {device} 좀 {verb}" (adverb after device) → **control_then_confirm**

같은 단어들, 다른 라벨. 학습된 패턴을 후처리로 보완.

### 2. 모델 예측의 의외: curtain 'open'

"블라인드 내려", "블라인드 올려" 같은 명확한 동작 발화에서 모델이 'open' 예측하는 경우 다수.
기존 rule은 stop/none만 커버 → 'open' 포함 확장으로 **+0.62%p 한 번에 획득**.

### 3. Entity-aware 어휘 분류

"전화" 키워드 자체는 ambiguous:
- "전화해줘" → 일반 전화 (OOD, unknown)
- "관리사무소 전화" → 관리실 번호 (in-domain, home_info)

Entity 유무로 구분하는 rule 필요.

### 4. 라벨 불일치 한계

TS 내부에서도 라벨 모순 다수:
- "거실 좀 불 켜줘" → CTC+on
- "안방 좀 불 켜줘" → clarify+none
- 같은 패턴, 다른 라벨 → 후처리로 해결 불가

TS 라벨 자체 재검토 필요 (P3 #9 실사용 로그 기반).

## 규칙 설계 원칙 (iter9에서 정립)

1. **Narrow first**: 넓은 rule 먼저 쓰고 regression 생기면 좁힘
2. **KE 교차 검증 필수**: TS gain vs KE regression trade-off 측정
3. **Capability query 제외**: "어떻게 할 수 있어" 형 질문은 system_meta 유지
4. **Entity marker**: 애매 keyword는 문맥(entity) 기반 분기

## 남은 오류 분석 (154→135 감소)

Top 5 remaining (iter9 이후):
1. [14] light_control CTC+none→on (다운라이트 켜 — TS 라벨 오류 수준)
2. [10] ac_control CTC none→set (에어컨 모드 — TS 불일치)
3. [ 9] light_control CTC→clarify ({room} {adverb} 불 켜줘 — TS 불일치)
4. [ 8] unknown direct→direct (전화해줘 — 내 rule 처리 중)
5. [ 5] news_query direct→query (뉴스 브리핑 — TS 라벨 오류)

대부분 TS 라벨 자체 문제로 더 이상 rule-based 개선 어려움.
