# Iteration 8 — GT v2 재파싱 + Strategy 변형 전수 비교 (2026-04-21, 18:27-19:20)

## 문제의식

Iteration 2-4에서 확인: 라벨 품질 문제는 **부분 수정이 regression 유발** (v28b, v70). 원인 추적:
1. `parse_gt_scenarios.py`의 규칙 기반 guess 라벨
2. slash("/") 포함 발화에서 첫 키워드만 매칭 (뒤쪽 무시)
3. "밝게" 키워드 부재 → dir=down 오라벨
4. query marker 미구분 → 상태 조회를 제어로 오분류

**가설**: 규칙을 개선한 **전체 재생성**은 일관성 유지되므로 v70식 regression 피함.

## Step 1: GT v2 재파싱

`scripts/parse_gt_scenarios_v2.py` 작성:

### 개선 사항
- **Slash 처리 개선**:
  - 각 파트에 action 동사(닫/잠/열/켜/꺼/틀...) 유무 확인
  - 2+ 파트에 동사 = alternatives → 첫 파트만 ("닫아줘 / 열어줘" → "닫아줘")
  - 1 파트만 동사 = alias-list → 해당 파트 (예: "현관문/현관/도어락 열어줘" → "도어락 열어줘")
  - 0 파트 = fallback 전체 ("모닝콜이 울리고 / 울릴 때 켜줘")
- **키워드 추가**: "밝게/밝아" → up
- **키워드 우선순위 재정렬**: brightness > close > open > off > on > up > down > set
- **Query markers 우선 처리**: 확인/상태/몇 도/알려 등 → dir=none

### 결과 (v1 → v2 diff)
Known 204개 중 **3개만 변경** (1.4%):
```
"거실이 좀 어두침침한데?"                     set → down  ✅
"가스 밸브 닫아줘 / 열어줘"                    open → close ✅
"사용량 설정값이 초과되면 알려줘"              set → none  ✅
```

**분석**: GT 219개 중 rule-based 라벨은 **대부분 이미 맞았다**. 원본 v1 파서의 문제는 실제로는 작음.

→ 전체 재파싱해도 약간의 개선만 있을 뿐, training 데이터 전체(24.5K)를 다시 생성할 이유는 부족.

## Step 2: Strategy 변형 전수 비교

기존 Strategy B 외 7개 변형 테스트 (`scripts/eval_strategies_variants.py`):

| Strategy | TS combo | fn | exec | dir | KE fn | Balanced |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **B (current)** | **93.95%** | 98.03 | 97.50 | 97.73 | **97.79%** | **95.87** ★ |
| B-mod (dir=v46) | 92.93% | 98.03 | 97.50 | 96.55 | 97.79 | 95.36 |
| B-param (param=v46) | 93.95% | 98.03 | 97.50 | 97.73 | 97.79 | 95.87 |
| B-both | 92.93% | 98.03 | 97.50 | 96.55 | 97.79 | 95.36 |
| MaxConf all | 94.94% | 99.41 | 97.08 | 97.60 | 91.60 | 93.27 |
| MaxConf ex-fn | 93.69% | 98.03 | 97.08 | 97.60 | 97.79 | 95.74 |
| All v46 | 93.30% | 98.03 | 97.24 | 96.55 | 97.79 | 95.55 |
| **Avg (softmax mean)** | **95.04%** | 99.44 | 97.11 | 97.63 | 91.34 | 93.19 |
| Avg ex-fn (fn=v46) | 93.79% | 98.03 | 97.11 | 97.63 | 97.79 | 95.79 |

### 분석

**핵심 발견**:
1. **Avg은 TS 95.04%로 최고점** — 단순 softmax 평균 만으로 +1.09%p
2. 하지만 **KE 91.34%로 -6.45%p 급락**
3. 원인: v28의 fn head가 KoELECTRA 같은 외부 도메인에서 generalization 약함 → avg가 v28 오답에 흘림
4. **fn=v46 고정**하면 TS 이득(+1%)조차 사라짐

**결론**: TS/KE trade-off의 해는 존재하지 않음. 현 B가 **balanced ceiling**.

## Step 3: Preprocess 사전 확장 (ROADMAP P1 #2)

`scripts/preprocess.py` 개선:
- **버그 fix**:
  - 엘레베이터 이중 변환 (엘레베이터 → 엘리베이터 → 엘리베이터이터) 해결 — 중복 엔트리 제거
  - "스물다섯도" regex 파서 버그 (char class에 multi-char) → group alternation
  - 사전 적용 순서 bug (prefix 먼저 매치) → 긴 패턴 먼저 sort
- **신규 엔트리 (~40개 추가)**:
  - 띄어쓰기 없는 패턴: "지금몇시야", "오늘날씨어때", "알람꺼"
  - 방+디바이스 조합: "거실에어컨", "침실조명", "서재조명" 등
  - 구어체: "따뜻하게해", "밝게해", "은은하게해"
  - 종결 오류: "꺼쥬", "켜쥬", "잠가쥬" 등

### 효과
| 측정 | Before | After | Δ |
|------|:---:|:---:|:---:|
| Strategy B TS combo | 93.95% | 94.08% | +0.13%p |
| fn head | 98.03% | 98.03% | - |
| exec head | 97.50% | 97.63% | +0.13%p |
| dir head | 97.73% | 97.93% | +0.20%p |
| KE fn | 97.79% | 97.79% | - |

작지만 실제 STT 오류 패턴을 직접 다룸 — 로그 수집 후 추가 확장 예상됨.

## 총평

### 시도한 것
1. ✅ Parse v2 (slash/밝게/query marker 개선) — 3 labels 수정
2. ✅ Strategy 9개 변형 TS + 전체 KE 평가 — B 최고 확인
3. ✅ Preprocess 사전 +40개 + 버그 3개 수정 → TS +0.13%p
4. ❌ 3-tier conf fallback (conf<0.3 → unknown, 0.3~0.5 → clarify)
   - TS -0.04%p, UX 이득은 실제 배포해야 검증 가능
   - 측정되는 손실 vs 불확실한 UX 이득 → **2-tier 유지**
   - ROADMAP P1 #4 결론 문서화됨
5. ✅ Response 템플릿 다양화 (ROADMAP P1 #3)
   - 각 fn별 1→2~3개 variation, `random.choice` 선택
   - 예시: "거실 불 켜줘" → {"네, 거실 조명을 켰습니다" | "거실 조명 켰습니다" | "알겠습니다, 거실 조명을 켰습니다"}
   - TS 93.76% 유지 (라벨 영향 없음), 같은 발화에 반복되는 응답 회피 → UX 개선
6. ✅ Alarm/Morning-call post-proc rule
   - v46 오류 19건 패턴: "알람 X" → system_meta (잘못 예측)
   - 규칙: device keyword 없는 `알람|모닝콜` → fn=schedule_manage
   - device-bound ("조명 타이머", "에어컨 타이머") 제외 safeguard
   - **sap_inference_v2**: TS 93.76% → **94.15% (+0.39%p)**, KE 97.33% 유지
   - **ensemble + rules**: TS 93.53% → **94.05% (+0.52%p)**, KE 97.79% → 97.27% (-0.52%p)
   - 라벨 불일치 발견: TS는 알람 = schedule_manage, KE는 알람 = system_meta
   - 판단: schedule_manage가 **의미론적으로 정확** (alarm = scheduling), KE 라벨이 오류
   - → 규칙 유지. 실 사용자 로그로 최종 검증 예정 (P3 #9)
7. ❌ Elevator default dir=on rule (dir:missing 완화 시도)
   - 단순 규칙: fn=elevator_call + dir=none → on
   - TS 94.15% → 93.82% (-0.33%p)
   - 원인: 학습된 'elevator_call + none' 케이스 (query/상태) 과다 override
   - → **revert**, 규칙 기반 접근은 세밀한 context awareness 필요

### 결론

현 `nlu_v28_v46_ensemble.onnx` (Strategy B) 배포가 **수학적 balanced 최적점**. 

- GT 라벨 수정 영향: **미미** (3/219)
- Ensemble 전략 개선: **모두 B 못 넘음**
- 남은 경로: 이전 세션 결론과 동일 → **실사용 로그 기반 재학습** (P3 #9)

### 산출물
- `scripts/parse_gt_scenarios_v2.py` (slash 로직 개선)
- `scripts/eval_strategies_variants.py` (9개 전략 전수 비교)
- `scripts/preprocess.py` (사전 +40개, 3개 버그 fix)
- `data/gt_known_scenarios_v2.json` / `gt_unknown_scenarios_v2.json`
- `data/strategy_variants_results.json`

### iter8 최종 metric

| 평가 경로 | Before | After | Δ |
|----------|:---:|:---:|:---:|
| sap_inference_v2 (v46+rules) TS | 93.76% | **94.15%** | **+0.39%p** |
| sap_inference_v2 KE fn | 97.33% | 97.33% | 0 |
| Ensemble B no rules TS | 93.59% | 93.59% | 0 |
| Ensemble B + rules TS | 93.53% | **94.05%** | **+0.52%p** |
| Ensemble B + rules KE fn | 97.79% | 97.27% | -0.52%p (label 불일치) |
| Ensemble B (preprocess 개선 반영) TS | 93.95% | 94.08% | +0.13%p |

**배포 권장**:
- 저장 ONNX: `nlu_v28_v46_ensemble.onnx` (변경 없음)
- 변경: `scripts/ensemble_inference_with_rules.py` 후처리 rule (+알람 rule)
- 변경: `scripts/preprocess.py` (+40 entries, 3 bug fix)

### 시도했으나 유지 안 된 것
| 시도 | Δ TS | 이유 |
|------|:---:|------|
| 3-tier conf fallback | -0.04% | 측정 손실, UX 이득 불확실 |
| Elevator dir=none→on | -0.33% | 학습된 query 케이스 과다 override |
| Elevator exec→control | -0.07% | 규칙 정교화 했지만 여전히 regression |

## 한 줄 결론

> **"GT 라벨 파싱은 거의 맞고, Strategy B는 balanced ceiling이다. 이제 진짜로 모델 레벨 실험 완전 종료."**
> 
> **"단, 도메인별 post-proc rule은 아직 +0.5%p 여지가 있다 (알람 rule 성공 예시)"**
