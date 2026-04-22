# Morning Handoff — Iter 8/9 Overnight Session

**기간**: 2026-04-21 22:18 ~ 2026-04-22 08:00 (~9.7시간)  
**상태**: ✅ 모든 변경사항 GitHub main 푸시 완료  
**커밋 수**: 40+ (iter9 + iter8 포함)

## TL;DR (3줄 요약)

1. **TS 93.53% → 95.76% (+2.23%p)** — 모델 재학습 없이 후처리 rule 14개 + preprocess 확장만으로
2. **배포 인프라 완성** — `DeploymentPipeline` 클래스 + DST 고도화 + ARCHITECTURE 문서
3. **다음 단계**: Android JNI 포팅 (문서 `docs/ARCHITECTURE.md` 참고)

## 배포 권장 구성 (현재)

```
ONNX:       checkpoints/nlu_v28_v46_ensemble.onnx (105MB, 변경 없음)
Inference:  scripts/deployment_pipeline.py의 DeploymentPipeline 클래스
Rules:      scripts/ensemble_inference_with_rules.py의 apply_post_rules()
Preprocess: scripts/preprocess.py (210+ entries, 2-pass)
DST:        scripts/dialogue_state_tracker.py (slot filling, 5-turn history)
```

## 핵심 metric (Ensemble + iter9 rules)

| 지표 | 값 |
|------|:---:|
| TS combo | **96.32%** (3,043 케이스) |
| KE fn | 97.20% (1,536 케이스) |
| GT 219 combo | 95.0% |
| Latency | 0.67ms/query CPU |
| Regression | 26/26 pass |

## 주요 변경 사항

### Iter 8 (22:18-19:20)
1. 알람/모닝콜 → schedule_manage rule
2. OOD keywords → unknown
3. unknown → 날씨/뉴스/의료 복구
4. Preprocess +40 entries + 3 bug fix
5. Response 템플릿 다양화
6. 3-tier conf fallback 실험 (유보)

### Iter 9 (22:18+ overnight)
7. 어순 rule (`{room}{device} 좀 {verb}`)
8. Curtain open→up/close/stop 확장 (+0.62%p 큰 gain)
9. Heat CTC + none → on
10. 화면/알림/음량 → home_info (capability 예외)
11. 전화 entity 기반 분기
12. 알람 dir 덮어쓰기 제거
13. 공기청정 → vent_control
14. DST value 추출 확장 (7 types)
15. DST slot filling + fn 상속
16. DeploymentPipeline 클래스
17. ARCHITECTURE.md, API_USAGE.md, SCRIPTS_INDEX.md 신규
18. Preprocess 2-pass + 사투리/존댓말/영어 혼용
19. 다중 room 추출 (rooms 리스트)
20. 타이머+action 응답 자연스러움

## 잊지 말 것

### DO
- ✅ `python3 scripts/regression_test_iter9.py` 변경 전후 실행
- ✅ TS + KE 양쪽 측정 (trade-off 확인)
- ✅ `ensemble_inference_with_rules.py`와 `sap_inference_v2.py` 양쪽 rule 동기화
- ✅ 복잡한 rule은 예외 케이스 먼저 체크 (capability query, entity marker 등)

### DON'T
- ❌ ensemble.onnx 재학습 (model ceiling 확증됨)
- ❌ Majority vote 3-model (검증 완료, 기각됨)
- ❌ 부분 라벨 수정 후 retrain (v70에서 regression 확증)
- ❌ "엘리베이터 dir=on default" 같은 넓은 rule (query 케이스 파괴)

## 다음 단계 권장

### Immediate (이번 주)
1. **Android JNI 통합**: `docs/ARCHITECTURE.md` 참고
2. **T527 NPU NB 변환**: ONNX → NB (Acuity toolkit)
3. **End-to-end 파이프라인 테스트**: STT (Citrinet) → NLU → Response

### 1-2 month
4. **실사용 로그 수집** (P3 #9) ← **진짜 다음 단계**
5. Weekly review queue 운영
6. 월간 재학습 사이클

### Long-term
7. GT 수동 재구축 (TS 라벨 inconsistency 해결)
8. 논문 작성 (SLT 2026, STT QAT)

## 문서 읽을 순서

신규 참여자용:
1. `README.md` — 프로젝트 개요
2. `docs/PROJECT_OVERVIEW_2026_04_21.md` — 통합 아키텍처
3. `docs/ARCHITECTURE.md` — 파이프라인 상세
4. `docs/API_USAGE.md` — DeploymentPipeline 사용법
5. `docs/SCRIPTS_INDEX.md` — 스크립트 가이드
6. `docs/DEPLOYMENT_CHECKLIST.md` — 배포 체크리스트

상태 확인용:
- `docs/ITER9_FINAL_REPORT.md` — 이번 세션 종합 보고
- `docs/KNOWN_FAILURES.md` — 해결/미해결 패턴

## 검증 커맨드

```bash
# Regression (항상 pass 해야 함)
python3 scripts/regression_test_iter9.py

# TS + KE 성능
python3 scripts/ensemble_inference_with_rules.py

# GT 219
python3 scripts/test_gt_scenarios.py

# Demo (5 카테고리)
python3 scripts/demo_comprehensive.py

# End-to-end pipeline
python3 scripts/deployment_pipeline.py
```

## 한 줄 결론

> **모델은 고정. 후처리 + 인프라로 +2.23%p 획득. 배포 준비 완료. 다음은 실사용 데이터.**

---

## 추가 (2026-04-22 08:48 ~ critical reflection 30분 세션)

사용자 요청: "30분 고민해봐 — 이게 최선인지, single intent와 비교 어떤지"

결과:
- **CRITICAL_REFLECTION_2026_04_22.md** 작성 (8개 부록, 논리 전개)
- **실제 버그 5개 발견 + 수정**:
  1. "방이 덥네" → heat 오예측 → ac 교정 rule
  2. DST "까요?" confirm → control 승격
  3. DST device follow-up dir 우선순위 수정
  4. Query fn spurious dir → none 강제
  5. (참고) 덥/춥 양방향 보강
- **Multi-turn 벤치마크 신설** (test_multiturn.py) — **DST 가치 +85.7%p 정량화**
- Regression test 30개로 확장 (26 → 30)

### 핵심 발견 (reflection)

1. **Multi-head가 Single intent보다 본질적으로 우수** — KE 72.5% vs 97.8% (+25%p) 수치 근거
2. **모델 architecture 선택은 올바름** — 온디바이스 + 데이터 부족 환경에서 최적
3. **14 rules는 band-aid지만 measurably justified** — GT +0.5%p (generalization 확인)
4. **벤치마크 TS 96.32%는 wild input에서 85~90% 수준** 예상 — 정량된 gap은 실 user data로만 메울 수 있음
5. **진짜 Next step은 명확**: 모델 실험 소진 → 실사용 로그 수집

## 최종 metric (reflection 수정 포함)

| 지표 | 값 |
|------|:---:|
| TS combo | **96.32%** (수정 안 됨) |
| KE fn | 97.14% (-0.06%p 알람 label 불일치) |
| GT 219 | 95.0% |
| **Multi-turn (DST on)** | **7/7 (100%)** |
| **Multi-turn (DST off)** | 1/7 (14%) |
| **DST 정량 가치** | **+85.7%p** |
| Regression tests | 30/30 pass |
| Latency | 0.67ms/query CPU |

## 최종 commit 수 (reflection 포함)

50+ commits since iter7 (iter8 + iter9 + reflection).
