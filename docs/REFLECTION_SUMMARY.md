# Critical Reflection Summary — "이게 최선인가?"

사용자 질문에 대한 답. 30분 고민 + 실제 테스트 결과.

## Q1: Multi-head vs Single Intent — 무엇이 더 좋은가?

| 차원 | Flat 94-intent | Multi-head 5-head | 승자 |
|------|:---:|:---:|:---:|
| TS combo | 94% | 93.3% | ≈ |
| **KE fn (외부 일반화)** | 72.5% | **97.8%** | 🏆 MH **+25.3%p** |
| OOD rejection | 1/5 | 5/5 | 🏆 MH |
| 조합 일반화 | 3/6 | 5/6 | 🏆 MH |
| DST 지원 | 불가능 | 가능 | 🏆 MH |
| 데이터 효율 | O(N×M) | O(N+M) | 🏆 MH |

**결론**: **Multi-head 완승** (TS는 동등하지만 현실 케이스에서 압도적)

실험 근거: `scripts/compare_flat_vs_multihead.py` + 히스토리 실제 측정치

## Q2: iter9까지 한 작업이 최선인가?

**YES, 주어진 제약에서는.**

| 제약 | 우리 상황 | 대안 |
|------|----------|------|
| T527 NPU 120MB 제한 | ✓ 105MB 앙상블 | LLM 불가능 |
| 데이터 24.5K | ✓ MH + augment | Transformer는 데이터 부족 |
| 온디바이스 < 1ms | ✓ 0.67ms | 서버 LLM은 100ms+ |
| 한국어 구조화 | ✓ 5-head | 영어 dataset 없음 |

### 이번 세션 성과

```
iter7 (시작점)       →  iter9 + reflection (종점)
─────────────────       ─────────────────────────
TS combo:  93.53%   →   95.76% (+2.23%p)
KE fn:     97.79%   →   97.14% (-0.65%p, 알람 label 불일치)
GT 219:    94.5%    →   95.0% (+0.5%p)
Multi-turn: 미측정   →   7/7 (DST +85.7%p 가치)
Latency:   0.55ms   →   0.67ms (rule 오버헤드)
문서:      부분     →   완전 (ARCHITECTURE, API_USAGE, SCRIPTS_INDEX)
```

**모델 변경 없이 +2.23%p**, 14개 rule + DST 고도화 + 배포 인프라 + 실 테스트.

## Q3: 더 좋은 방법은 없는가?

검토한 대안들:

### ❌ Tree-structured semantic parser (MTOP style)
- 복잡한 decoder 필요, 24K 데이터 부족
- 예상 TS +0.5%p, 구현 비용 2주 → **ROI 낮음**

### ❌ 작은 LLM (1-3B)
- NPU 120MB 제한 위반 → **기술적 불가능**

### ❌ Hierarchical intent (2-stage)
- Latency 2배 ↑, cascade error
- **복잡도 증가 대비 이득 미미**

### ❌ 더 큰 Transformer
- v63 Conformer → -14%p regression 확인
- 데이터 부족 → **이미 실패 증명됨**

### ✅ 실사용 로그 수집 (진짜 다음 단계)
- 모델 실험 소진, 데이터 품질 개선만 남음
- `docs/FEEDBACK_SYSTEM_DESIGN.md` 에 설계 있음
- **이것만이 95%→97% 돌파 경로**

## Q4: 놓친 건 없나?

Reflection 과정에서 **실제로** 발견한 5개 버그/개선:

1. **"방이 덥네" → heat 오예측** → ac 강제 rule 추가
2. **DST "까요?" 확인 질문** → control 승격 로직 추가
3. **DST device follow-up dir** → 이전 action 우선순위
4. **Query fn spurious dir** → none 강제
5. **Multi-turn 벤치마크 부재** → `scripts/test_multiturn.py` 신설

이들은 TS 벤치마크로는 안 잡히지만 wild input에서 실제 사용자 경험을 해치는 것들.

## Q5: 현재 deployable한가?

**YES.**

- ONNX: `checkpoints/nlu_v28_v46_ensemble.onnx` (105MB)
- API: `scripts/deployment_pipeline.py`의 `DeploymentPipeline` 클래스
- 테스트: 30개 assertion + 7개 multi-turn 시나리오 모두 통과
- 문서: `docs/ARCHITECTURE.md` Android JNI 포팅 가이드
- Latency: 0.67ms CPU (NPU 포팅 시 <0.3ms 예상)

## 한 문장 결론

> **"제약 조건(온디바이스 + 데이터 부족) 안에서 최선이다. KE 72%→97% 차이가 Single intent 기각 근거. 다음은 모델이 아닌 실사용 데이터."**

## 부록: 실측 숫자 (모두 재현 가능)

```bash
# TS + KE
python3 scripts/ensemble_inference_with_rules.py
# → TS 95.76% / KE 97.14%

# GT 219
python3 scripts/test_gt_scenarios.py
# → Ensemble + rules: combo 95.0% (208/219)

# Multi-turn (reflection에서 신설)
python3 scripts/test_multiturn.py
# → DST on 7/7, DST off 1/7

# Regression
python3 scripts/regression_test_iter9.py
# → 30/30 pass
```
