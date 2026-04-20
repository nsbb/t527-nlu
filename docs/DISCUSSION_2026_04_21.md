# NLU 설계 토론 정리 (2026-04-21, 엄격판)

이 세션 후반부에 나눈 핵심 토론을 정리. 측정 지표의 진짜 의미, multi-head 아키텍처의 당위성, 라벨 품질 문제 등 **구조적 질문**에 대한 답.

**이 문서는 두 번 작성됨**:
- v1: 초판 (일부 과장/추론 포함)
- v2: **이 버전** — 재검토 후 엄격하게 수정, 과장 제거, 한계 명시

---

## 0. 이 문서의 신뢰도 가이드

| 표기 | 의미 |
|------|------|
| ✅ **확실** | 실측 데이터로 증명 |
| ⚠️ **추정** | 타당한 추론이지만 증명 안 됨 |
| 🔬 **가설** | 검증 필요한 주장 |
| 🎯 **실무** | 이론은 약하지만 실용적 결론 |

---

## 1. v46이 ceiling이라는 게 무슨 의미인가

### 배경
v47-v68 (15개 실험) 모두 v46 초과 못함. "v46이 최적"이라고 결론.

### 검증된 사실 ✅
- v54-v68 모두 Test Suite combo에서 v46(93.3%) 미달
- Model Soup에서 v28↔v46 interpolation 성능 급락 (loss landscape 비볼록)
- Patch data 추가 시 regression 재현성 있음 (v29-v33, v58, v68)

### 추정/가설 ⚠️
- "데이터 24.5K가 상한" — 더 큰 데이터 안 써봄, 모름
- "Conformer가 CNN 못 이김" — Conformer는 더 긴 학습 필요할 수 있음
- "v46 Mixup이 '정확히' 필요한 정규화" — 왜 잘 되는지 이론적 설명 부족

### 🎯 실무적 결론
현재 설정 범위 내에서 v46이 가장 좋았음. **"Ceiling"이라 부르는 건 과장**이고 **"현재 탐색 범위의 최고"**가 정확한 표현.

---

## 2. v68 라벨 수정 — 이전 버전들 잘못된 데이터로 학습?

### 질문 (User)
> "68에 라벨수정한거는 데이터가 잘못된거 아닌가 그럼 68 전에 버전들은 잘못된데이터로 학습한거아님?"

### 검증된 사실 ✅
- v68이 학습 데이터 46건 라벨 수정 (예: "커턴 닫아" dir=open→close)
- 이 46건은 v43 데이터셋에 있던 오류
- 그 전 모든 버전(v34, v46 등)은 이 오류 포함된 데이터로 학습
- 전수 검토 결과: 학습 데이터 1,176건 확실한 오류 (그 외 404건 모호)
- Test Suite 11건 라벨 오류 수정 (95% 확실)

### 재귀적 문제 ⚠️
이전 답변에서 놓친 점:
- **KoELECTRA pseudo-label은 v28이 해줬음**
- v28이 잘못된 라벨로 학습 → KoELECTRA 13K를 그 관점에서 라벨링 → v46이 그걸 학습
- **같은 오류가 재귀적으로 전파**
- v46의 KE 97.8%도 "잘못된 스크립트 기준"에 맞춘 것

### 🎯 결론
벤치마크 수치 모두 **잘못된 라벨에 잘 맞춘 결과**. "v46 93.3%"는 실제 성능 아님.

**실제 모델 품질 측정 불가** — 라벨 자체가 스크립트 guess라서.

---

## 3. GT가 있나? 엑셀에는 발화-응답만 있는데?

### 질문 (User)
> "지금 엑셀에는 사용자발화 -> ai기대응답(문장출력) 인데 그걸 어케 멀티헤드로 gt를 만들었다는겅미"

### 검증된 사실 ✅
- 원본 엑셀 컬럼: `구분 / 세부기능 / 서비스유형 / 사용자 발화문 / AI기대응답 / ...`
- **fn/exec_type/param_direction/param_type/judge 라벨 없음**
- 멀티헤드 라벨은 `scripts/parse_gt_scenarios.py`가 **규칙 기반 생성**

### 변환 과정 신뢰도
| Head | 생성 방식 | 신뢰도 |
|------|---------|:---:|
| fn | `세부기능` 컬럼 매핑 (27→20 fn) | ~95% (거의 결정적) |
| exec_type | `서비스유형` + 응답 해석 + 특수 키워드 | ~70% (규칙 복잡) |
| param_direction | 발화에서 키워드 guess (켜/꺼/열어/닫아) | ~60% (슬래시 묶음 버그) |
| param_type | 키워드 guess | ~70% |
| judge | 키워드 guess | ~80% (소수 카테고리) |

### 구체적 버그 발견
```python
for kw, d in [('열어', 'open'), ..., ('닫아', 'close'), ...]:
    if kw in utt: return d
```
원본: `"가스 밸브 닫아줘 / 열어줘"` (슬래시로 두 명령)
→ `열어`가 먼저 매칭 → `open` 반환 → **"닫아"가 open 라벨로 잘못 생성**

### 🎯 결론
"GT"라 불러온 건 엄밀히 **"스크립트 추정 라벨"**. 엑셀의 **진짜 GT는 발화+응답 텍스트**뿐.

모든 후속 모델/벤치마크가 이 위에서 구축됨 → **측정 수치의 근본적 한계**.

---

## 4. 일반화(KE)가 더 중요함 — v28의 3%p는 무의미

### 질문 (User)
> "일반화가 더 중요함 우리 데이터 증강으로만 한거라서 당연히 잘되는거니까 그건 원래 잘되는게 맞음 그거 3프로 높다고 좋은게 아니고 애초에 학습할떄 아예 못봣던 문장들을 70퍼밖에안되는게 별로인거임 28버전은"

### ✅ 동의 — 완전히 맞음

v28 GT 97.7%는 "우리 증강 데이터에 과적합"된 지표. 의미 있는 지표:

1. **외부 데이터 성능** (KE 97.8% @ v46) ← **진짜 중요**
2. **Unknown fallback 정확도** (실제 거부해야 할 것 거부)
3. **사용자 만족도** (실사용 로그 기반)

내부 Test Suite는 자기복제 측정.

### 주의 ⚠️
**단 KoELECTRA도 일반 한국어 분류 데이터**지 월패드 특화 평가는 아님. 진짜 production 일반화는 **실사용 로그로만 측정 가능**.

---

## 5. Multi-head vs Flat+Unknown — 정말 다른가?

### 질문 (User)
> "단일 인텐트 nlu 모델도 unknwon 인텐트 만들어서 gt 테이블에 없는 모든 문장들은 무조건 unknown으로 보내버리면 되는거 아니가?"

### 원래 결정 근거
`handoff_semantic_action_parser_complete.md` (2026-04-09):
> "233개 시나리오를 전부 인텐트로 정의하면 거대한 룰베이스가 된다... AI는 안 본 걸 잘해야 AI인데."

### 실제 비교 실험 (2026-04-21)
`scripts/compare_flat_vs_multihead.py`:

| 항목 | Flat v18 (94 intent) | Multi-head v46 | 차이 |
|------|:---:|:---:|:---:|
| 안 본 조합 (6) | 5/6 | 6/6 | +1 |
| OOD 거부 (5) | **1/5** | 5/5 | +4 |
| GT 219 fn | 91.8% | 95.4% | +3.6%p |
| **KoELECTRA 1,536 fn** | **72.5%** | **97.8%** | **+25.3%p** |

### ⚠️ 이 비교의 **불공정성**

**중요 자기 수정**: 앞서 "multi-head 우위 증명됨"이라 했던 것 **부분적으로 잘못**.

| 요소 | v18 | v46 |
|------|:---:|:---:|
| 아키텍처 | Flat 94-intent | Multi-head 5 |
| 학습 데이터 | GT 전용 (~21K) | GT + KE pseudo (34K) |
| Pseudo-labeling | ❌ | ✅ |
| Mixup | ❌ | ✅ |

**v18 vs v46 차이 중 multi-head 구조 기여도는 분리 안 됨**.

### 진짜 공정한 비교 (미실시)
```
실험 A: v18 + KE pseudo-label + Mixup 학습 (flat+same data/tech) → ? KE
실험 B: v46 동일 조건 → 97.8% KE

A vs B 비교해야 "multi-head 구조" 기여도 측정 가능.
```

이 ablation을 **안 했음**. 따라서:
- ✅ **증명됨**: "v46 recipe (multi-head + pseudo + mixup + KE data) > v18 recipe"
- ❌ **증명 안 됨**: "Multi-head 구조가 Flat보다 본질적으로 우수"

### 🎯 실무적 결론
**현재 조건에서 multi-head 경로가 우월** — 하지만 이게 **아키텍처 덕분인지 데이터/기법 덕분인지 분리 안 됨**.

### 업계 관찰 (사실만)
- Google Smart Home: Trait 시스템 (capability별 분리)
- Amazon Alexa: Intent + Slot filling (BIO)
- Apple Siri: Domain + Intent + Parameter (계층)
- Facebook MTOP: Tree-structured semantic parsing
- Rasa/Dialogflow: Intent + Entity

**패턴**: 모두 **structured output**. 순수 flat intent만으로 production NLU는 드묾.  
하지만 구현 방식은 제각각. **"Multi-head만 정답"은 아님** — "구조화가 표준" 정도.

---

## 6. 고차원 분리의 기하학적 직관

### 질문 (User)
> "차원이 여러개면 다른 벡터보다 멀리 떨어뜨릴 수 있으니까. 멀리떨어뜨리면 classification 은 무조건 이득 아님?"

### 정확한 답 (앞서 약간 뭉뚱그렸음)

**내부 feature space는 양쪽 다 256d 동일**. Flat이든 Multi-head든 backbone은 동일 구조.

차이는 **output parameterization**:
```
Flat:        1 × softmax(94)      — 94개 1-of-N 분류
Multi-head:  5 × softmax(20/5/9/5/5) — 5개 독립 분류
```

### User 직관의 진짜 본질
User가 느낀 건 **output factorization**:

예) "거실 에어컨 켜줘" 학습 → "거실 에어컨 **꺼줘**" (학습 안 봄) 예측:
- **Flat**: `turn_on_living_ac` 클래스만 배움 → `turn_off_...`는 모름 → 오분류
- **Multi-head**: fn=ac, exec=control, dir=on 각각 배움. 다른 샘플에서 dir=off 배움 → **조립 가능**

즉 **고차원 분리의 이점이 아니라 "factored output의 조합 일반화"**.

### ⚠️ 하지만 이것도 조건부
Flat도 충분한 데이터(각 조합별 샘플)가 있으면 같은 결과. Multi-head의 이점은 **적은 데이터로 조합 커버**.

---

## 7. Unknown 폴백이 Multi-head 취지 깨는가?

### 질문 (User, 예리)
> "처음 시작이 멀티헤드로 하면 학습할때 못봣던 문장이 들어오더라도 강건하게 대응 가능인데. 지금 unknown 태그를 넣어버리면 그 이점이 없어진거아닌ㄱ다"

### 테스트 결과 ✅

**In-domain 학습 안 본 조합** (6/6, 모두 conf=1.0):
```
"안방 에어컨 제습 모드 25도"      → ac_control + control + set ✓
"큰방 난방 2시간 예약 22도"        → heat_control + control + set ✓
"모든 방 에어컨 시원하게 해줘"      → ac_control + control + on ✓
... (6개 전부 조립 성공)
```

**Out-of-domain** (4/5, conf=1.0으로 unknown):
```
"트럼프가 누구야"     → unknown
"주식 사고싶어"       → unknown
"배달 시켜줘"         → unknown
"월패드 비번 변경"    → system_meta (미지원으로 런타임 차단)
```

### 🎯 결론
Multi-head compositional generalization과 Unknown 폴백은 **서로 다른 대상**:
- Multi-head → **표현 가능 범위 내** 새 조합
- Unknown → **표현 불가 범위** 거부

실제로 충돌 없이 작동함 확인됨. User 우려는 기우.

---

## 8. "학습 더 많이 하면 Flat도 되지 않나?"

### 🎯 실무적 답
**원리적으로 가능, 우리 규모에서는 어려움.**

### 정량 근거의 한계 ⚠️
앞서 "Multi-head가 4-5배 데이터 효율"이라 했는데:
- ✅ 이론: Multi-task learning이 데이터 효율 좋음 (논문 다수)
- ⚠️ 우리 상황: 실제 측정 안 했음, 추정치

### 조합 폭발 논지의 정정
"Flat 94로 900 조합 못 담음"이라 했는데:
- 실제 르엘 조합은 65개 (엑셀 기준) 또는 219 시나리오
- 94 intent로도 실용적 커버 가능했을 것
- **본질 문제는 "조합 폭발"이 아니라 "새 조합 일반화 효율"**

### 솔직한 결론
- Flat은 **이론적으로 가능**하지만
- **Unknown 학습 데이터 수집이 비쌈** (OOD 무한 공간)
- **업계는 일반적으로 structured output 선택** (완전 flat 드묾)
- 우리 규모에서는 Multi-head가 **더 실용적**

---

## 9. Intent + BIO Slot은 어떤가?

### 질문 (User)
> "intent+slot이 intent+bio slot 그건가?"

### ✅ 맞음

BIO tagging:
```
"안방 에어컨 23도로 맞춰줘"
안방   → B-room
에어컨 → O (or B-device)
23     → B-temp
도     → I-temp
로/맞춰/줘 → O
```

### 우리 5-head vs BIO 비교

| 항목 | Intent + BIO | 우리 5-head |
|------|------------|------------|
| Slot 표현 | 토큰 span 명시 | sentence-level 분류 |
| "23도" 추출 | B-temp, I-temp | regex `(\d+)도` |
| Annotation | 토큰당 라벨 (비쌈) | 문장당 5 라벨 (싸) |
| 복합 명령 | 자연스러움 | 규칙 확장 필요 |
| NPU | 복잡 (sequence output) | 단순 (classification) |

### ⚠️ 우리 선택의 한계 (앞서 소홀히 다룸)
- **복합 명령 불가**: "거실 23도, 안방 25도" — 우리 방식은 room 하나만 추출
- **자연어 숫자**: "이십삼도" preprocess 의존 — 없으면 실패
- **복합 시간 표현**: "내일 오전 7시부터 저녁까지" 같은 건 regex로 처리 어려움

**우리는 복잡한 발화를 피하도록 설계 타협**함. BIO 썼으면 더 유연했을 것.

---

## 10. "확실해?" 자기 검토

### 앞서 과장한 것들

1. **"Multi-head 우위 증명됨"** → 실제로는 v46 recipe 우위. 아키텍처 독립 기여 분리 안 됨.
2. **"900 조합 못 담음"** → 실제 조합은 65개. 이론적 과장.
3. **"데이터 효율 4-5배"** → 직관 기반 추정. 실측 안 함.
4. **"Unknown 구축 불가능"** → 어려움이지 불가능 아님. Alexa는 함.
5. **"고차원이 유리"** → 내부 feature dim은 동일. Output factorization이 진짜 이점.

### 앞서 놓친 것들

1. **v46도 자주 틀림** — demo에서 "방이 덥다→heat" 오류
2. **라벨 품질의 재귀적 문제** — KoELECTRA pseudo-label도 잘못됨
3. **Head 간 negative transfer** 언급 부족
4. **BIO slot의 복합 명령 대응 우위** 인정 부족
5. **업계 Trait vs 우리 head의 추상화 레벨 차이**

### 부족한 증명 (해야 할 ablation)

1. v18 + KE data 재학습 → flat도 KE 오를까?
2. Joint intent+slot (BIO) vs 우리 5-head
3. Same-params flat vs multi-head (공정 비교)
4. 실사용 로그 평가

---

## 11. 총정리 — 엄격판

### ✅ 확실한 사실
1. v54-v68 모든 실험이 v46 미달 (TS + KE 종합)
2. v68 라벨 수정 46건 + Test Suite 11건 (확실한 오류)
3. 학습 데이터 1,176건 규칙 기반 오류 존재
4. 멀티헤드 라벨은 스크립트 guess (엑셀 GT 아님)
5. KE +25.3%p (v18 72.5% vs v46 97.8%)
6. OOD 거부 1/5 vs 5/5

### ⚠️ 추정/가설
1. Multi-head 구조 자체의 기여도 = 불분명
2. Flat이 같은 조건(KE data, mixup)이면 어땠을지 = 모름
3. v46이 "수학적 ceiling"인지 = 모름 (더 탐색 안 함)

### 🎯 실무적 결론
- 현재 v46 (또는 v28+v46 앙상블)이 가장 좋은 제품 성능
- Multi-head는 업계 표준 패턴과 일치 (구조화 output)
- **벤치마크 수치는 과대평가 가능성 높음** (라벨 품질 의존)

### 📉 해야 할 것 (안 한 것)
1. 학습 데이터 라벨 수동 검수
2. 진짜 GT 엑셀에서 재구축
3. 실사용 로그 수집
4. Ablation: flat + 같은 조건 비교

---

## 12. 이 토론의 의미

User의 질문 순서:
1. v46 ceiling 의미 → 데이터 상한 언급
2. v68 라벨 수정 → 이전 모든 버전 의심
3. GT 존재 → 스크립트 guess 폭로
4. 일반화 중요성 → KE가 진짜 지표
5. Multi-head vs Flat → 아키텍처 정당성
6. 고차원 직관 → Output factorization
7. Unknown vs 조합 일반화 → 공존 증명
8. 학습 더 많이 → 이론 vs 실무 구분
9. BIO slot → 업계 표준 구조
10. 재검토 요청 → **이 문서의 v2 작성 계기**

**각 질문이 우리 설계의 가정을 하나씩 시험**함. 결과:
- 아키텍처 선택은 합리적이지만 증명 부족
- 벤치마크 수치는 신뢰 제한적
- 진짜 개선 경로는 **데이터 품질**에 있음

---

## 13. 산출물

| 파일 | 용도 |
|------|------|
| `scripts/interactive_test.py` | 대화형 3-모델 비교 |
| `scripts/test_gt_scenarios.py` | GT 219 자동 테스트 + CSV |
| `scripts/compare_flat_vs_multihead.py` | Flat vs Multi-head 비교 |
| `scripts/comprehensive_label_audit.py` | 라벨 전수 검토 |
| `scripts/categorize_suspects.py` | Suspect 카테고리 분류 |
| `scripts/find_label_errors.py` | 라벨 오류 탐지 |
| `scripts/fix_test_suite_labels.py` | 라벨 자동 수정 |
| `data/gt_test_results.csv` | GT 219 전체 결과 |
| `data/label_audit_with_model.json` | 전수 검토 보고서 |
| `data/suspects_categorized.json` | 카테고리별 의심 |
| `docs/DISCUSSION_2026_04_21.md` | **이 문서 (v2 엄격판)** |
| `docs/ARCHITECTURE_PROPOSALS_2026_04_21.md` | 다음 단계 제안 |
