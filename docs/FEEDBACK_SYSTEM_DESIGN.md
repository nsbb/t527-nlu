# Production Feedback System Design

작성: 2026-04-21 (iter5)
상태: 설계 문서 (구현 전)

## 목표

**모든 모델/학습 기법 실험 소진 (v46 ceiling 확인)** 이후 유일한 진짜 개선 경로:  
**실사용 로그 + 피드백 루프 기반 데이터 품질 개선**

## 현재 상태

```
학습 데이터 24.5K = 규칙 기반 증강 (rule-based)
라벨 = 스크립트 추측 (parse_gt_scenarios.py guess)
벤치마크 93.59% TS + 97.79% KE = 자기복제 측정
실사용자 실제 발화 패턴 = 미지 (데이터 없음)
```

## 설계 원칙

1. **Privacy first** — 사용자 음성은 디바이스 내 처리, 텍스트만 로그
2. **Opt-in** — 사용자 동의 기반 로그 수집
3. **Low friction** — 수동 피드백은 선택적, 자동 로그 우선
4. **Iterative** — 월간 재학습 사이클

## 시스템 아키텍처

```
[월패드 T527 (온디바이스)]
    ↓ 사용자 발화 + NLU 예측
    ↓ ONLY low-confidence OR manual flag
    ↓ (암호화, 익명화)
[로그 서버]
    ↓
[Review Queue (internal dashboard)]
    ↓ 주간 검토 (데이터 팀)
    ↓ 수정된 라벨
[학습 데이터 저장소]
    ↓ 월간 증분
[재학습 파이프라인]
    ↓
[새 NLU 모델]
    ↓ A/B 테스트
[배포]
```

## 로그 수집 세부

### 무엇을 로그할 것인가

**자동 수집 (conf<threshold 또는 unknown)**
```json
{
  "ts": "2026-04-21T15:30:00Z",
  "device_id": "HASH(mac)",  // 익명화
  "session_id": "uuid",
  "turn": 3,
  "stt_text": "거실 에어컨 켜줘",
  "nlu": {
    "fn": "ac_control",
    "exec": "control_then_confirm",
    "dir": "on",
    "fn_conf": 0.65,  // 낮음 → 수집 대상
    "top3_fn": [["ac_control", 0.65], ["heat_control", 0.22], ["unknown", 0.13]]
  },
  "slots": {"room": "living"},
  "user_action": "executed"  // 또는 "cancelled"
}
```

**조건**:
- conf<0.7 (전체의 ~5-10% 예상)
- fn=unknown
- 사용자 수동 피드백 ("잘못 이해함" 버튼)

**무엇을 로그하지 않는가**:
- 음성 원본 (privacy)
- 사용자 식별 정보 (device_id는 단방향 해시)
- STT confidence <0.5 (노이즈 과다)
- conf>0.95 (충분히 확실한 경우)

### 용량 추정

```
1 로그 엔트리 ≈ 1KB
월 발화 추정: 사용자당 1000건
  × 10% 수집 대상 = 100건/사용자/월
  × 1000명 = 100K 엔트리/월 = 100MB/월
```

서버 부담 낮음.

## Review Queue 운영

### 주간 검토 프로세스 (데이터 팀)

1. **자동 분류**
   - Confidence bucket (low/mid/high)
   - 예측 fn 분포
   - 모델 간 불일치 (ensemble에서 v28 vs v46)

2. **우선순위 샘플링**
   - High-conf errors (모델 자신만만하게 틀림): 최우선
   - Unknown → known 후보: 둘째
   - 특정 fn 오분류 패턴: 셋째

3. **수동 라벨링**
   - 주 100-200개 목표 (2-4시간 작업)
   - 전문가 1명 + 검증자 1명 (더블 체크)

4. **학습 데이터 추가**
   - 수정된 라벨 → `data/feedback_labeled.json`
   - 월별 증분 배치

### Review UI 설계 (간단 web form)

```
┌─────────────────────────────────────────┐
│ 리뷰 큐: 우선순위순 (조 conf)             │
├─────────────────────────────────────────┤
│ [1/100] "거실 에어컨 켜고 시원하게"        │
│  NLU: fn=ac_control (0.52)              │
│       exec=control_then_confirm (0.90)  │
│       dir=on (0.88)                     │
│                                         │
│  올바른 라벨:                            │
│    fn: [ac_control ▼]                   │
│    exec: [control_then_confirm ▼]       │
│    dir: [on ▼]                          │
│    param_type: [mode ▼]  ← 수정 가능     │
│                                         │
│  태그: [ ] 복합 명령                     │
│        [ ] STT 오류                     │
│        [ ] 미지원 기능                   │
│                                         │
│  [ 건너뛰기 ] [ 저장 & 다음 ]             │
└─────────────────────────────────────────┘
```

## 재학습 Cycle

### 월간 업데이트

**Week 1-3**: 로그 수집 + 주간 리뷰
**Week 4**: 재학습 + A/B 테스트

```bash
# 재학습 파이프라인
1. Load train_final_v43.json (baseline)
2. Merge data/feedback_labeled_month_N.json (이번 달 검토된 것)
3. Apply all A-category label fixes (suspects_categorized.json)
4. Run v46 recipe training
5. Evaluate on TS + KE + Feedback holdout
6. A/B test 2주
7. 이전보다 개선 시 배포
```

### 예상 개선 궤적

```
월 1: baseline 93.59% (Ensemble B)
월 2: +300 samples → 94.0%? (conservative)
월 3: +300 → 94.5%?
월 6: +2000 → 96%?
월 12: 지속적 개선 + drift 보정
```

## 수동 피드백 UX (월패드 UI)

**최소 간섭 원칙**:
- 사용자가 명령 거부 시 자동 기록 (conf 낮았을 가능성)
- "잘 이해 못했어요" 응답 후 3초 내 재발화 → 이전 낮은 conf 기록

**Explicit feedback (선택)**:
```
[사용자]: 거실 에어컨 켜줘
[앱]: 네, 거실 에어컨을 켰습니다.
[사용자]: (음량 내려짐)
[사용자]: 아니 내가 말한 건 그거 아니야
     ↓
[앱]: "무엇을 원하셨나요? (아래 선택)"
   ○ 난방 켜기
   ○ 에어컨 *끄기*
   ○ 다른 것 (음성 재녹음)
```

사용자가 선택하면 강한 신호 → 레이블 교정 후보.

## 성공 지표 (6개월 후)

- [ ] 누적 1000+ 수동 교정 샘플
- [ ] Test Suite combo 95%+ (현재 93.59%)
- [ ] Low-conf 비율 <3% (현재 7%)
- [ ] 사용자 리포트 기반 "잘못된 해석" 월간 감소

## 구현 우선순위

### Phase 1 (이번 달)
1. [ ] 로그 스키마 확정
2. [ ] 디바이스 로깅 모듈 개발 (conditional, encrypted)
3. [ ] 서버 수집 API (HTTPS, auth)
4. [ ] 데이터 팀 브리핑

### Phase 2 (다음 달)
5. [ ] Review UI 프로토타입
6. [ ] 첫 100 샘플 pilot review
7. [ ] 재학습 파이프라인 자동화

### Phase 3 (3개월+)
8. [ ] 월간 리뷰 루틴 확립
9. [ ] 재학습 자동화
10. [ ] A/B 테스트 인프라

## 비용 추정

| 항목 | 월간 |
|------|:---:|
| 서버 (로그 수집) | ~$50 |
| 데이터 팀 (검토) | 0.25 FTE |
| GPU 재학습 | 1-2시간 × $5 |
| 총 | ~$200 + 0.25 FTE |

## 리스크

1. **Privacy 우려**: opt-in 필수, 암호화 엄격
2. **라벨 불일치**: 더블 체크 + guideline 문서
3. **Drift**: 시간 지나면 발화 패턴 변화 → 오래된 데이터 가중치 감소
4. **Cold start**: 첫 3개월은 수집량 적음 → 예상 만큼 개선 안 될 수도

## 결론

**이 시스템 없이는 NLU 성능은 현재 93.6%에서 정체됨**.

모델/레시피 레벨 실험 전부 실패 (이번 세션 iter 1-4 확인) → 진짜 개선은 **데이터 품질 증가만이 답**.
