# NLU 배포 체크리스트 (Production Release)

작성: 2026-04-21 (iter9 update)
대상 모델: **Ensemble Strategy B (v28+v46) + 후처리 rule**
**TS 96.85% / KE 97.20% (이전 배포 93.53% → +2.23%p 개선)**

## 0. 배포 모델 선정

### 현재 권장: Ensemble v28+v46 (Strategy B) + iter9 후처리 rule
```
파일:     checkpoints/nlu_v28_v46_ensemble.onnx  (105MB FP32, 변경 없음)
후처리:   scripts/ensemble_inference_with_rules.py  (★ 10개 rule 적용)
Preprocess: scripts/preprocess.py  (★ 190+ entries, iter9 확장)
성능:     TS 96.85%, KE 97.20%, balanced 96.48
Latency:  0.67ms/query CPU (preprocess + inference + rules)
```

### 옵션 비교

| 옵션 | 크기 | TS | KE | Latency | 용도 |
|------|:---:|:---:|:---:|:---:|------|
| **Ensemble FP32 + iter9 rules** | 105MB | **96.85%** | **97.20%** | **0.67ms** | ★ 현재 배포 |
| Ensemble FP32 (no rules) | 105MB | 93.59% | 97.79% | 0.55ms | 참고 (rule 제외) |
| Ensemble FP16 | 52.5MB | 93.59% | 97.79% | 20.4ms | GPU/NPU 전용 |
| v46 단독 + rules | 100MB | 94.68% | 97.27% | 0.50ms | 더 가벼운 배포 |
| v28 단독 | 100MB | 95.53% | 75.59% | 0.37ms | GT 전용 (외부 일반화 약함) |

→ **FP16은 CPU에서 37x 느림** (onnxruntime CPU는 FP16 최적화 없음)
→ NPU/GPU 배포 시에만 FP16 유효

## 1. Pre-flight Checks

- [ ] **Regression test** 100% pass
  ```bash
  python3 scripts/regression_test.py
  # Golden 12/12, STT 10/10, TS ≥93%, KE ≥97%, Latency ≤5ms 모두 통과
  ```

- [ ] **Demo 시나리오** 정상 동작
  ```bash
  python3 scripts/demo_dialogs.py
  # 10 시나리오, 32 turn 모두 정상 응답
  ```

- [ ] **ONNX 파일** 유효성
  ```bash
  python3 scripts/verify_ensemble_onnx.py
  # TS combo 94%+, latency <1ms 확인
  ```

- [ ] **전처리 사전** 최신 버전 (190+개 STT 교정, iter9 확장)
  ```bash
  grep -c ':' scripts/preprocess.py | head  # STT_CORRECTION 개수 확인
  ```

- [ ] **후처리 rule** 10개 적용 확인 (iter8/9)
  ```bash
  grep -c "iter8\|iter9" scripts/ensemble_inference_with_rules.py
  # 최소 10+ 매치 (주석 + 규칙)
  ```

## 2. Android 앱 통합

### 파일 배포 위치
```
app/src/main/assets/
├── models/nlu/
│   └── nlu_v28_v46_ensemble.onnx  (105MB)
├── tokenizer/
│   ├── vocab.txt
│   └── tokenizer_config.json
└── preprocess_config.json  (STT 교정 사전)
```

### JNI/Java 인터페이스 필요
- [ ] `NLUEngine.java` — 추론 wrapper
- [ ] `Preprocessor.java` — 한글 숫자 + STT 교정 포팅
- [ ] `DST.java` — 멀티턴 상태 관리 포팅
- [ ] `ResponseGenerator.java` — 템플릿 기반 응답

### 성능 예산 (T527 기준)
- 토큰화: ≤0.5ms
- NLU 추론 (CPU): ≤1ms  
- Rule-based slots: ≤1ms
- DST lookup: ≤0.1ms
- 응답 생성: ≤1ms
- **NLU 총계**: ≤5ms

## 3. 런타임 후처리 필수

### 규칙 A — param_type 보정
```python
if param_direction in ('open', 'close', 'stop'):
    param_type = 'none'
if judge != 'none':
    param_type = 'none'
if exec_type in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
    param_type = 'none'
```

### 규칙 B — Confidence fallback
```python
fn_probs = softmax(fn_logits)
if fn_probs.max() < 0.5:
    fn = 'unknown'
    exec_type = 'direct_respond'
    # → 서버로 fallback
```

### 규칙 C — dir 교정 (v28 학습 오류 보정)
```python
if re.search(r'밝게', text) and param_direction == 'down':
    param_direction = 'up'
    param_type = 'brightness'
if re.search(r'어둡게', text) and param_direction in ('up', 'on'):
    param_direction = 'down'
    param_type = 'brightness'
```

### 규칙 D — 미지원 액션
```python
UNSUPPORTED_ACTIONS = {
    'medical_query': ['예약', '상담', '진료', '증상'],
    'market_query': ['추천', '매수', '매도'],
    'news_query': ['구독', '매일 아침', '브리핑'],
    'system_meta': ['비밀번호 바꿔'],
    # ...
}
# 매칭 시 → "죄송합니다. 해당 기능은 지원하지 않습니다."
```

## 4. 로깅 & 모니터링

### 필수 로그 (사용자 발화마다)
- [ ] Timestamp
- [ ] 원본 발화 (STT 결과)
- [ ] 전처리된 발화
- [ ] 예측 결과 (fn, exec, dir, param, judge, confidence)
- [ ] 룰 매칭 (UNSUPPORTED, Confidence fallback 등)
- [ ] 추출된 slots (room, value)
- [ ] DST 상태 (사용 시)
- [ ] 응답 템플릿 선택

### 알림 지표
- [ ] Confidence<0.5 비율 (≥5% 넘으면 경고)
- [ ] Unknown fn 비율 (도메인 외 발화 많으면 경고)
- [ ] Response latency p99 (>10ms 경고)
- [ ] Rule D 매칭 빈도 (미지원 기능 수요 파악)

### 샘플 로그 포맷 (JSON)
```json
{
  "ts": "2026-04-21T15:30:00Z",
  "session_id": "uuid-xxx",
  "turn": 3,
  "stt_text": "거실 에어콘 켜줘",
  "preprocessed": "거실 에어컨 켜줘",
  "preds": {
    "fn": "ac_control",
    "exec": "control_then_confirm",
    "dir": "on",
    "conf": 0.98
  },
  "slots": {"room": "living", "value": null},
  "unsupported": false,
  "response_template": "control_then_confirm.ac_control",
  "latency_ms": 3.2
}
```

## 5. A/B Test 권장

### 시나리오
- 50% 사용자: Ensemble (현재)
- 50% 사용자: v46 단독 (크기/속도 더 좋음, TS 0.23%p 낮음)

### 측정
- 사용자 만족도 (대화 완료율)
- 응답 적절성 (수동 샘플 리뷰)
- Edge case 처리 (복잡 발화, STT 오류)

### 결과로 결정
- Ensemble 우위 유지 → 그대로
- v46 단독이 실사용 동일하면 → 가벼운 v46으로 전환

## 6. Rollback 계획

### 문제 발생 시
1. **즉시 전환**: `nlu_v28_final.onnx` (TS 96.3%, KE 75.5%)
   - GT 패턴에서 가장 정확
   - 외부 일반화 약함
   - 한시적 사용

2. **완전 rollback**: 이전 버전 체크포인트
   - `checkpoints/cnn_multihead_v34.pt` (v34, TS 93.3%, KE 96.8%)

### 검증 기준 (rollback 트리거)
- 사용자 오작동 리포트 급증
- Unknown 비율 >15% (지속)
- 특정 fn 오분류 >10% 수준

## 7. 지속 개선 경로

### Phase 1 (즉시)
- [x] Ensemble ONNX 배포 준비
- [x] 후처리 rules 정리
- [x] 로깅 인프라 설계

### Phase 2 (1~2개월)
- [ ] **실사용 로그 수집 파이프라인**
  - Production 발화 수집
  - 익명화 + 저장
  - Weekly review queue
- [ ] **피드백 UI** (선택적)
  - 사용자가 오분류 수정 가능

### Phase 3 (3~6개월)
- [ ] **GT 수동 재구축**
  - 엑셀 219 시나리오 전문가 검수
  - 진짜 벤치마크 확보
- [ ] **학습 데이터 대폭 확장**
  - 실사용 로그 기반
  - 월간 재학습 cycle

## 8. 알려진 한계 (User 전달 필요)

1. **복합 명령 처리 불가** ("거실 23도, 안방 25도" 같은 경우 room 하나만)
2. **일부 schedule_manage 오분류** ("알람 맞춰줘" → system_meta 가능)
3. **자연어 숫자 preprocess 의존** ("이십삼도" 미지원 시 실패)
4. **Out-of-domain 발화** → unknown fallback (서버 처리)

## 9. 관련 문서

- `docs/MODEL_CARD.md` — 모델 공식 스펙
- `docs/DEPLOYMENT_GUIDE.md` — 상세 통합 가이드  
- `docs/KNOWN_FAILURES.md` — 실패 패턴 리스트
- `docs/ROADMAP.md` — 향후 개선 계획
- `docs/MODEL_LIMITATIONS.md` — 구조적 한계

## 10. 배포 승인 체크

- [ ] 기술 리뷰 완료 (Regression + Demo + ONNX 검증)
- [ ] 문서 최신화 (README, CHANGELOG, MODEL_CARD)
- [ ] 로깅 인프라 준비
- [ ] Rollback 절차 검증
- [ ] Stakeholder 브리핑 (한계 + 모니터링 계획)
- [ ] 첫 24시간 집중 모니터링 계획
