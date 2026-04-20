# NLU 설계 토론 정리 (2026-04-21)

이 세션 후반부에 나눈 핵심 토론을 정리. 측정 지표의 진짜 의미, multi-head 아키텍처의 당위성, 라벨 품질 문제 등 **구조적 질문**에 대한 답.

---

## 1. v46이 ceiling이라는 게 무슨 의미인가

### 배경
v47-v68 (15개 실험) 모두 v46 초과 못함. "v46이 최적"이라고 결론.

### 진짜 이유
```
v46 = CNN 4L + pseudo-labeling + mixup (24.5K 샘플로)
```

1. **데이터 상한 도달** — 24,507 샘플로 뽑을 수 있는 최대. 더 큰 모델 과적합
2. **TS vs KE 트레이드오프는 구조적** — GT 데이터와 KoELECTRA 데이터의 exec/dir 라벨 체계가 다름
3. **v46 Mixup이 필요한 정규화 정확히 제공** — 다른 기법은 중복이거나 깨뜨림
4. **Weight space 비볼록** — Model Soup 실패가 증거. 결합은 prediction-level(앙상블)만 가능
5. **패치 데이터는 항상 regression 유발** — v29-v33, v58, v68 동일 패턴

---

## 2. v68 라벨 수정 — 이전 버전들은 잘못된 데이터로 학습한 것 아닌가?

### 질문 (결정적)
> "68에 라벨수정한거는 데이터가 잘못된거 아닌가 그럼 68 전에 버전들은 잘못된데이터로 학습한거아님?"

### 답
**네, 맞음.** 이게 이 세션의 가장 중요한 발견.

- v68에서 학습 데이터 46건 라벨 오류 수정
- 이 46건은 v43 데이터셋에 있던 오류 → v34, v46 등 **전부 잘못된 패턴 학습**
- Test Suite도 동일한 오류 11건 있었음
- **"v46이 정답 맞힘"은 잘못된 라벨 × 잘못된 예측이 일치**한 것

### 전수 검토 결과 (scripts/comprehensive_label_audit.py)
| 데이터셋 | Suspect 건수 | Unique 발화 |
|---------|:---:|:---:|
| Test Suite (3,043) | **287건** | 255 unique |
| Train Data (24,507) | **2,767건** | 2,407 unique |
| Cross-conflicts | **112건** | - |

v68이 수정한 46건은 **빙산의 일각** — 학습 데이터에 최소 1,176건의 확실한 오류 존재.

### 벤치마크의 진짜 의미
- 학습 데이터 틀림 → 모델이 틀리게 학습
- 테스트 데이터도 같은 방식으로 틀림 → 모델의 틀린 예측이 "맞는 것"으로 측정
- **둘이 같은 방향으로 틀려서 숫자가 좋아 보임**
- **"v46 93.3%"는 망가진 벤치마크의 최적 맞춤**, 실제 모델 성능 아님

---

## 3. GT가 있나? 엑셀에는 발화-응답만 있는데?

### 질문
> "지금 엑셀에는 사용자발화 -> ai기대응답(문장출력) 인데 그걸 어케 멀티헤드로 gt를 만들었다는겅미"

### 답
엑셀 원본은 `사용자 발화문` + `AI기대응답` 텍스트만. **fn/exec_type/param_direction 같은 멀티헤드 라벨 없음**.

### 변환 과정 (scripts/parse_gt_scenarios.py)
```python
def guess_direction(utt, stype):  # ← 이름이 "guess"
    for kw, d in [('켜', 'on'), ('꺼', 'off'), ...
                   ('열어', 'open'),            # ← 먼저 체크
                   ('닫아', 'close'), ...]:     # ← 나중에 체크
        if kw in utt: return d
```

원본 엑셀 행: `"가스 밸브 닫아줘 / 열어줘"` (슬래시로 두 명령 묶임)
→ `열어`가 리스트에서 먼저 나옴 → `open` 반환 → **"닫아" 라벨이 `open`으로 잘못 생성**

### 결론
- 엑셀에 진짜 GT는 **발화 + 응답 텍스트**뿐
- 멀티헤드 라벨은 **규칙 기반 스크립트의 추측**
- fn은 비교적 신뢰 (세부기능 매핑, 거의 1:1)
- **exec/dir/param/judge는 60~70% 신뢰** (키워드 guess)
- 우리가 "GT"라고 부른 건 엄밀히 **"스크립트 추정 라벨"**

이게 1,176+건의 오류의 진짜 원인.

---

## 4. 일반화(KE)가 더 중요함 — v28의 3%p는 무의미

### 질문
> "일반화가 더 중요함 우리 데이터 증강으로만 한거라서 당연히 잘되는거니까 그건 원래 잘되는게 맞음 그거 3프로 높다고 좋은게 아니고 애초에 학습할떄 아예 못봣던 문장들을 70퍼밖에안되는게 별로인거임 28버전은"

### 답
**완전히 맞음.**

v28 GT 97.7%는 "우리 증강 데이터에 과적합" 지표. v46 KoELECTRA 97.8%가 **진짜 가치 있는 숫자** — 학습 분포 밖 발화에 대한 일반화.

### 올바른 지표 우선순위
1. **KE fn (외부 데이터 일반화)** ← 진짜 중요
2. **Unknown fallback 정확도** (거부해야 할 것 거부하는지)
3. **실사용 사용자 만족도** (데모에서 확인)
4. 내부 Test Suite 지표 — 자기복제 측정일 뿐

---

## 5. Multi-head vs Flat intent + unknown — 진짜 차이?

### 질문
> "애초부터 지금 그냥 단일 intent 뽑는거 대비해서 우리꺼 좋은 이유가 뭐임? 단일 인텐트 nlu 모델도 unknwon 인텐트 만들어서 gt 테이블에 없는 모든 문장들은 무조건 unknown으로 보내버리면 되는거 아니가?"

### 원래 결정 근거 (handoff_semantic_action_parser_complete.md, 2026-04-09)

> **"233개 시나리오를 전부 인텐트로 정의하면 사실상 거대한 룰베이스가 된다. 모델이 하는 건 '어떤 룰을 적용할지 고르는 것'뿐이고, 학습하지 않은 새로운 조합에 대응할 수 없다."**
>
> **"핵심 질문: '이게 AI인가, 그냥 룰베이스 아닌가? AI는 안 본 걸 잘해야 AI인데.'"**

해결책: **독립 축(axis) 병렬 예측 → 조합 일반화**. Facebook MTOP (2020) 참조.

### 실제 이점 (head별)

| Head | 진짜 이득 | Flat으로 대체 가능? |
|------|---------|---------|
| fn (20) | 있음 (데이터 효율) | 94 intent면 클래스당 샘플 ¼ |
| exec_type (5) | **있음** | fn×exec 복제 필요 (20→60 intent) |
| param_direction (9) | 약함 | `light_on`/`light_off` 만들면 됨 |
| param_type (5) | **없음** | rule-based로 뽑는 게 더 정확 |
| judge (5) | 약함 | 카테고리별 intent로 가능 |

### 핵심 이득 정리
1. **exec_type 분리**는 확실한 이득 (query/control/clarify 구분)
2. **데이터 효율** — fn 20 classes × 24K = class당 1,225 샘플 (flat 94면 255)
3. **조합 표현** — `fn × exec × dir` = 900 조합 표현 가능

---

## 6. 고차원이 classification에 유리? (기하학적 직관)

### 질문
> "flat intent라고 하면 수직선에다가 쭉 일열로 나열한거인가? 조금만 빗나가면 클래스 잘못 고르면 오판단인데. 우리꺼로 하면 차원이 여러개니까 벡터로 해서 표현하면 1차원보다 2차원이, 2차원보다 3차원이 다른 벡터보다 멀리 떨어뜨릴 수 있으니까. 멀리떨어뜨리면 classification 은 무조건 이득 아님?"

### 답 — 직관 거의 맞음

내부 feature space는 양쪽 다 256차원(동일). 차이는 **출력 구조**:

```
Flat (94 intents):
  output = [logit_1, ..., logit_94]   # 한 번에 하나 선택

Multi-head (20×5×9):
  fn: [...20]    exec: [...5]    dir: [...9]
  → factored output = (fn_i, exec_j, dir_k) 튜플
```

### 진짜 이점 ("차원 분리" 직관의 정체)

예: "거실 에어컨 켜줘" 학습:

**Flat**: `turn_on_living_ac` 클래스 1개 배움 (94개 중)
- "거실 에어컨 **꺼줘**" (안 배움) → 다른 intent로 오분류 가능

**Multi-head**: fn=ac / exec=control / dir=on **각각 배움**
- 다른 샘플에서 dir=off 배움 → 조합 "ac+control+off"는 **안 봐도 조립 가능**

이게 "output factorization"의 본질 — **조합 일반화**.

### 추가 이점
- **Multi-task learning** — fn 학습 gradient가 exec에도 도움
- **Parameter 효율** — 20+5+9=34개 출력 → 94개 조합 표현

### 단점
- Head 간 독립 가정 (실제는 의존: fn=weather면 exec 거의 항상 query)
- 부조리 조합 가능 (fn=weather + dir=on)

---

## 7. Unknown 폴백이 multi-head 취지 깨는 것 아닌가?

### 질문 (예리함)
> "처음 시작이 멀티헤드로 하면 학습할때 못봣던 문장이 들어오더라도 강건하게 대응 가능인데. 지금 unknown 태그를 넣어버리면 그 이점이 없어진거아닌ㄱ다"

### 일견 모순처럼 보임
- Multi-head 원래 취지: "안 본 조합도 각 축 조립"
- Unknown 추가: "학습 안 된 건 서버로"
- ↑ 상충?

### 실제 테스트 (v46, 2026-04-21)

**In-domain 학습 안 본 조합** (6/6 성공, 모두 conf=1.0):
```
"안방 에어컨 제습 모드 25도"      → ac_control + control + set ✓
"큰방 난방 2시간 예약 22도"        → heat_control + control + set ✓
"아이방 커튼 반만 열어"            → curtain_control + control + open ✓
"주방 환기 10분 후 꺼"             → vent_control + control + off ✓
"거실 조명 은은하게 30퍼센트로"    → light_control + control + set ✓
"모든 방 에어컨 시원하게 해줘"      → ac_control + control + on ✓
```

**Out-of-domain** (4/5):
```
"트럼프가 누구야"     → fn=unknown         (conf=1.0)
"주식 사고싶어"       → fn=unknown         (conf=1.0)
"배달 시켜줘"         → fn=unknown         (conf=0.99)
"월패드 비번 변경"    → fn=system_meta     (런타임 UNSUPPORTED로 차단)
```

### 결론 — 두 메커니즘은 보완 관계

```
┌─────────────────────────────────────────────┐
│ In-domain 새 조합 ─────→ Multi-head 조립     │
│                          (fn+exec+dir 각 head)│
│                                             │
│ Out-of-domain ────────→ 모델이 fn=unknown    │
│                          직접 학습            │
│                                             │
│ 애매 (conf<0.5) ──────→ 폴백 안전장치        │
│                          (거의 안 걸림)        │
└─────────────────────────────────────────────┘
```

- Multi-head → 표현 가능 범위 **내** 조합 일반화
- Unknown → 표현 가능 범위 **밖** 거부
- **역할 분담**, 충돌 아님

User의 "모순 우려"는 검증 결과 **기우**였음. 원래 설계 의도가 실제로 작동 중.

---

## 총정리 — 이 세션의 교훈

1. **v46 ceiling**의 진짜 의미: 데이터 상한 도달, 모델/기법 탐색 소진
2. **모든 이전 버전 라벨이 rule-based guess** — 진짜 GT가 아니었음
3. **벤치마크 수치는 과대평가** — 학습/테스트 데이터가 같은 방향으로 틀림
4. **측정 우선순위**: KE fn > Unknown fallback > 사용자 만족도 > 내부 Test Suite
5. **Multi-head 설계는 타당** — 조합 일반화 실제 작동 확인
6. **Unknown + Multi-head는 보완** — 서로 다른 대상 처리
7. **다음 단계는 데이터 품질** — 모델/기법은 한계, 라벨 정제가 유일한 개선 경로

## 이 토론에서 만든 산출물

| 파일 | 내용 |
|------|------|
| `scripts/interactive_test.py` | 대화형 3-모델 비교 테스트 |
| `scripts/test_gt_scenarios.py` | GT 219개 자동 테스트 + CSV |
| `scripts/comprehensive_label_audit.py` | 라벨 전수 검토 (규칙+모델) |
| `scripts/categorize_suspects.py` | Suspect 카테고리 분류 |
| `scripts/find_label_errors.py` | 기본 라벨 오류 탐지 |
| `scripts/fix_test_suite_labels.py` | 테스트 라벨 자동 수정 |
| `data/gt_test_results.csv` | GT 219개 전체 결과 (Excel 열람 가능) |
| `data/label_audit_with_model.json` | 전수 검토 상세 보고서 |
| `data/suspects_categorized.json` | 카테고리별 의심 케이스 |
| `docs/DISCUSSION_2026_04_21.md` | **이 문서** |

## 이후 작업 제안 (실제 개선 경로)

### Option A: 라벨 품질 개선 프로젝트
- 학습 데이터 24,507건 라벨 수동 검수 (예상 1~2일)
- A 카테고리 자동 수정 (1,176건) → 수동 검토 → 재학습
- 진짜 모델 성능 측정 가능

### Option B: 수동 GT 재구축
- 엑셀 219개 행을 수동으로 멀티헤드 라벨 (2~3시간)
- 진짜 벤치마크 확보
- 기존 모델 재평가

### Option C: 실사용 로그 수집
- Production 배포 후 실제 발화 로그 → 피드백 루프
- 가장 시간 걸리지만 가장 가치 있는 방향
- ROADMAP P3
