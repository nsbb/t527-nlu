# NLU 배포 가이드 (Production Deployment)

## 권장 구성

| 시나리오 | 모델 | 파일 | TS | KE | 크기 |
|---------|------|------|:---:|:---:|:---:|
| 배포 기본 | v28+v46 앙상블 | `nlu_v28_v46_ensemble.onnx` | **95.76%** | **97.20%** | 105MB |
| 단일 모델 | v46 | `nlu_v46_generalization.onnx` | 93.3% | 97.8% | 100MB |
| 기존 패턴 전용 | v28 | `nlu_v28_final.onnx` | 96.3% | 75.5% | 100MB |

## 입출력 스펙

### 입력
- **Name**: `input_ids`
- **Shape**: `[batch, 32]` (int64)
- **Padding**: 0-padded to max_len=32
- **Tokenizer**: ko-sbert-sts (`jhgan/ko-sbert-sts`) BertTokenizer

### 출력 (앙상블 기준, 5개)

| 출력명 | Shape | 클래스 | 설명 |
|--------|-------|:---:|------|
| `fn_logits` | [batch, 20] | 20 | 기능 분류 (v46에서) |
| `exec_logits` | [batch, 5] | 5 | 실행 타입 (v28에서) |
| `dir_logits` | [batch, 9] | 9 | 방향 (v28에서) |
| `param_logits` | [batch, 5] | 5 | 파라미터 타입 (v28에서) |
| `judge_logits` | [batch, 5] | 5 | 판단 유형 (v46에서) |

### 클래스 정의

**fn (20)**: `light_control`, `heat_control`, `ac_control`, `vent_control`, `gas_control`, `door_control`, `curtain_control`, `elevator_call`, `security_mode`, `schedule_manage`, `weather_query`, `news_query`, `traffic_query`, `energy_query`, `home_info`, `system_meta`, `market_query`, `medical_query`, `vehicle_manage`, `unknown`

**exec_type (5)**: `query_then_respond`, `control_then_confirm`, `query_then_judge`, `direct_respond`, `clarify`

**param_direction (9)**: `none`, `up`, `down`, `set`, `on`, `off`, `open`, `close`, `stop`

**param_type (5)**: `none`, `temperature`, `brightness`, `mode`, `speed`

**judge (5)**: `none`, `outdoor_activity`, `clothing`, `air_quality`, `cost_trend`

## 파이프라인 전체 구조

```
사용자 음성
    ↓ STT (ex: Citrinet)
텍스트 "거실 에어콘 23도로 맞춰쥐"
    ↓ preprocess.py
     - STT 교정 (에어콘→에어컨, 맞춰쥐→맞춰줘)
     - 한글숫자 변환 (이십삼도→23도)
     - 공백 정리
"거실 에어컨 23도로 맞춰줘"
    ↓ 토크나이저 (max_len=32)
input_ids: [2, 1234, ..., 0, 0, ...]
    ↓ ONNX 추론 (0.48ms on CPU)
5개 logits
    ↓ argmax + 규칙 보정
preds = {fn, exec_type, dir, param, judge}
    ↓ Rule-based slot 추출
    - room: 정규식 (거실→living, 안방→bedroom_main)
    - value: 숫자 + 단위 (23도, 30분, 50%)
    ↓ DST (멀티턴 상태 추적, 10초 timeout)
    - follow-up 처리 (안방도, 난방도)
    - correction 처리 (아니 꺼줘)
resolved = {fn, exec, dir, room, value}
    ↓ 응답 생성 (generate_response)
    - control_then_confirm → "네, 거실 에어컨을 온도를 23도로 설정합니다."
    - query_then_respond → "현재 에어컨은 자동 모드로..."
    - direct_respond → "정보를 확인합니다."
    - clarify → "어떤 공간의 기기를 제어할지..."
"네, 거실 에어컨을 23도로 설정합니다."
    ↓ TTS
음성 응답
```

## 후처리 규칙 (param_type 보정)

```python
if param_direction in ('open', 'close', 'stop'):
    param_type = 'none'
if judge != 'none':
    param_type = 'none'
if exec_type in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
    param_type = 'none'
```

## Confidence Fallback

```python
confidence = softmax(fn_logits).max()
if confidence < 0.5 and fn != 'unknown':
    fn = 'unknown'
    exec_type = 'direct_respond'
    # 서버로 전달
```

## Latency 예산 (T527 기준)

| 단계 | 시간 | 주석 |
|------|------|------|
| STT | ~120-250ms | Citrinet, 주력 부분 |
| 전처리 | < 1ms | Python 문자열 연산 |
| 토큰화 | 0.1ms | BertTokenizer fast |
| NLU 추론 (CPU) | 0.48ms | Ensemble ONNX |
| 규칙 처리 | < 1ms | 정규식 + DST |
| 응답 생성 | < 1ms | 템플릿 매칭 |
| TTS | ~200-500ms | 외부 모듈 |
| **NLU 총계** | **~3ms** | STT 제외 |

## 미지원 액션 체크

일부 fn은 예측 정확하지만 **실제 지원 불가**한 경우 있음:

```python
UNSUPPORTED_ACTIONS = {
    'medical_query': ['예약', '상담', '진료', '증상', '처방'],
    'traffic_query': ['택시', '대리', '카풀', '렌트'],
    'market_query': ['추천', '매수', '매도', '예측', ...],
    'news_query': ['구독', '매일 아침', '브리핑 해줘'],
    'system_meta': ['비밀번호 바꿔', '이름 바꾸'],
    'home_info': ['골프장 예약', '수영장 예약', '자동 밝기'],
    'vent_control': ['필터 주문', '고쳐', 'AS', '수리'],
    'ac_control': ['고장', 'AS', '수리'],
    'weather_query': ['다음 달 날씨', '몇 mm', '강수량'],
}
# 매칭 시: "죄송합니다. 해당 기능은 지원하지 않습니다."
```

## Android 앱 통합 (awnn VNN)

1. ONNX → TFLite 또는 NB 변환 (필요 시)
2. `app/src/main/assets/nlu/`에 복사:
   - `nlu_v28_v46_ensemble.onnx`
   - `tokenizer/` (vocab.txt 포함)
   - `preprocess_config.json`
3. JNI 래퍼: `NLUEngine.java`
4. 전처리는 Java/Kotlin에서 구현 (STT 교정 사전 포팅)

## 검증 방법

```bash
# ONNX 검증
python3 scripts/verify_ensemble_onnx.py
# Expected: TS 95.76%, KE 97.20%, 0.67ms (iter9)

# 대화형 테스트
python3 scripts/sap_inference_v2.py

# 배치 테스트
python3 scripts/run_test_suite.py

# 멀티턴 DST
python3 scripts/dialogue_state_tracker.py
```

## 알려진 한계

1. **TS vs KE 트레이드오프**: 단일 모델로는 양쪽 모두 최적 불가 → 앙상블 권장
2. **문맥 의존 발화**: "응", "그거" 등은 DST 필수 (10초 timeout)
3. **낮은 신뢰도 발화**: conf<0.5 → unknown → 서버 폴백
4. **미지원 액션**: 위 UNSUPPORTED_ACTIONS 리스트 → 안내 멘트
5. **방 레이블**: bedroom_main/bedroom_sub 구분은 정규식 기반 (침실 vs 안방)

## 파일 위치

- 모델: `checkpoints/nlu_v28_v46_ensemble.onnx`
- 토크나이저: `tokenizer/vocab.txt`
- 전처리: `scripts/preprocess.py`
- 추론: `scripts/sap_inference_v2.py`
- DST: `scripts/dialogue_state_tracker.py`
- 앙상블 ONNX 내보내기: `scripts/export_ensemble_onnx.py`
- 검증: `scripts/verify_ensemble_onnx.py`

## 향후 개선 방향

1. **더 큰 학습 데이터**: 현재 24.5K → 50K+ (수작업 라벨 포함)
2. **도메인 적응**: 월패드 사용 로그 기반 실제 발화 수집
3. **KoELECTRA pre-training**: 라벨 품질 개선 (현재 ~1.2% 오류 존재)
4. **더 정교한 DST**: 5턴 이상 문맥 + 슬롯 채움 (slot filling)
5. **Edge case 전용 데이터**: schedule_manage vs system_meta 등 모호 패턴
