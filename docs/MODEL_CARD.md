# NLU Model Card — 르엘 AI 월패드 NLU

## 모델 정보

- **이름**: t527-nlu v28+v46 Ensemble
- **버전**: v66 (Ensemble ONNX)
- **날짜**: 2026-04-21
- **개발**: HDC Labs (T527 NLU Team)
- **타입**: 멀티헤드 분류 (5-head)
- **파일**: `checkpoints/nlu_v28_v46_ensemble.onnx`
- **크기**: 104.9MB FP32
- **라이센스**: 내부 사용

## 의도된 사용

### 기본 용도
- 한국어 스마트홈 음성 명령 이해
- STT 출력 텍스트 → 구조화된 Action (fn/exec/dir/param/judge)
- 르엘 어퍼하우스 월패드 device control

### 사용 시나리오
- "거실 에어컨 23도로 맞춰줘" → ac_control + control + set + temperature:23 + room:living
- "오늘 날씨 어때" → weather_query + query_then_respond
- "안방도" → DST에서 이전 context 상속

### 부적절한 사용
- 감정 분석 (지원 안 함)
- 번역/요약 (지원 안 함)
- 비한국어 입력 (영어 일부만 혼용 가능)
- 보안/의료 판단 (disclaimer 필요)

## 입출력

### 입력
- `input_ids`: int64 [batch, 32]
- Tokenizer: ko-sbert-sts BertTokenizer
- 권장: preprocess.py로 STT 교정 선행

### 출력 (5 logits)

| 출력 | Shape | 클래스 수 |
|------|-------|:---:|
| fn_logits | [batch, 20] | 20 (19 known + unknown) |
| exec_logits | [batch, 5] | 5 |
| dir_logits | [batch, 9] | 9 |
| param_logits | [batch, 5] | 5 |
| judge_logits | [batch, 5] | 5 |

## 성능 지표 (iter9 업데이트, 2026-04-22)

### 벤치마크

| 지표 | Base Ensemble | + iter8/9 rules | 비고 |
|------|:---:|:---:|------|
| Test Suite combo | 93.59% | **95.76%** | +2.17%p (rule로만) |
| KoELECTRA fn | 97.79% | 97.20% | -0.59%p (알람 라벨 불일치) |
| GT 219 combo | 94.5% | **95.0%** | +0.5%p |
| STT 오류 내성 | 100% | 100% | 10/10 유지 |
| CPU latency | 0.55ms | 0.67ms | rule 오버헤드 미미 |
| 크기 | 104.9MB | 동일 | ONNX 변경 없음 |

### 헤드별 정확도 (Ensemble + rules)

| 헤드 | Test Suite | KoELECTRA |
|------|:---:|:---:|
| fn | 99.34% | 97.20% |
| exec_type | 98.39% | - |
| param_direction | 97.90% | - |
| param_type | 99.5% | - |
| judge | 99.7% | - |

## 학습 정보

### 데이터
- **규모**: 24,507 샘플 (v43 데이터셋)
- **출처**:
  - GT 시나리오 증강 (르엘 219 시나리오): 9,990
  - KoELECTRA pseudo-labeled: 9,361
  - MASSIVE OOD: 1,509
  - HA 번역: 714
  - 기타 fix/augment: ~3,000
- **Val**: 2,883 샘플

### 학습 레시피
- **모델**: CNN 4L residual (k=3,5,7,3) + 5 heads
- **d_model**: 256
- **임베딩**: ko-sbert-sts 768d frozen
- **학습**: 40 epochs, AdamW (lr=1e-3), Cosine LR schedule
- **Augmentation**: Mixup 30% (같은 fn 내 발화 교체)
- **Loss weights**: fn×2, exec×2, dir×1.5, param×1, judge×1.5

### Ensemble Strategy B
- fn ← v46 (mixup, 일반화 최적)
- exec_type ← v28 (GT only, 정확도 최적)
- param_direction ← v28 (동일)
- param_type ← v28
- judge ← v46

## 한계 및 편향

### 알려진 한계

1. **TS vs KE 트레이드오프**
   - 단일 모델로 양쪽 모두 최적 불가
   - 앙상블로 해결 (94.3% + 97.8%)

2. **Schedule 발화 약점**
   - "알람/타이머" → system_meta로 오분류 빈발
   - 학습 데이터 부족 (~500 예시)

3. **구어체 방향 어려움**
   - "따뜻하게" → dir:up 매핑 부족 (주로 dir:on)
   - "시원하게" → 같은 이슈

4. **극단 STT 오류**
   - 띄어쓰기 없는 긴 구절 어려움
   - "오늘날씨어때" 등은 preprocess로 해결

5. **문맥 의존 발화**
   - "응", "그거" 등 DST 필수
   - 10초 timeout 기반

### 편향

- **도메인**: 아파트/월패드 중심, 일반 주택 patterns 부족
- **지역**: 서울/수도권 기본 가정
- **연령**: 성인 발화 위주 (노인/아동 특수 표현 부족)
- **방언**: 표준어 기본, 경상/전라 방언 미포함

## 운영 권장사항

### 신뢰도 임계값
- conf < 0.5 → unknown 처리 → 서버 fallback
- conf < 0.3 → 확실한 거부

### 후처리 규칙 (필수)
```python
if param_direction in ('open', 'close', 'stop'):
    param_type = 'none'
if judge != 'none':
    param_type = 'none'
if exec_type in ('query_then_respond', 'direct_respond', 'clarify', 'query_then_judge'):
    param_type = 'none'
```

### 미지원 액션 안내
```python
UNSUPPORTED_ACTIONS = {
    'medical_query': ['예약', '상담', '진료', '증상'],
    'market_query': ['추천', '매수', '매도'],
    # ...
}
```

## 윤리적 고려

1. **개인정보**: 음성 데이터 저장 안 함, 서버 전송 시 암호화
2. **편향**: 표준어 중심 → 방언 사용자에게 성능 저하
3. **접근성**: 노인/장애인 사용자 경험 개선 필요
4. **오작동 리스크**: gas_control, door_control 등 안전 관련 기능은 확인 응답 필수

## 디버깅 및 모니터링

### 추천 지표
- fn confidence distribution (conf<0.5 비율)
- unknown 비율 (너무 높으면 학습 데이터 부족 신호)
- Unsupported 매칭 빈도
- DST timeout 빈도

### 로깅 포인트
- 입력 발화 (STT 결과)
- 전처리 전/후
- fn/exec/dir 예측 + confidence
- 최종 응답

## 버전 히스토리

| 버전 | 주요 변경 | 날짜 |
|------|----------|------|
| v1-v28 | 초기 CNN 5-head, GT 증강 | 2026-04-13~19 |
| v34 | Pseudo-labeling +21%p | 2026-04-19 |
| v46 | Mixup +1%p | 2026-04-20 |
| v55-v63 | 여러 기법 실험 (모두 failure) | 2026-04-21 |
| v64-v65 | Unfreeze/KLUE emb (failure) | 2026-04-21 |
| **v66** | **Ensemble ONNX (current)** | **2026-04-21** |

## 파일 위치

- 모델: `checkpoints/nlu_v28_v46_ensemble.onnx`
- 토크나이저: `tokenizer/`
- 추론: `scripts/sap_inference_v2.py`
- 전처리: `scripts/preprocess.py`
- DST: `scripts/dialogue_state_tracker.py`
- 검증: `scripts/verify_ensemble_onnx.py`

## 문서

- [`docs/CHANGELOG.md`](CHANGELOG.md) — 버전별 변경
- [`docs/VERSION_LOG.md`](VERSION_LOG.md) — 실험 상세
- [`docs/KNOWN_FAILURES.md`](KNOWN_FAILURES.md) — 실패 패턴 분석
- [`docs/MODEL_LIMITATIONS.md`](MODEL_LIMITATIONS.md) — 구조적 한계
- [`docs/DEPLOYMENT_GUIDE.md`](DEPLOYMENT_GUIDE.md) — 배포 가이드
- [`docs/HEAD_CLASSES.md`](HEAD_CLASSES.md) — 5-head 클래스 정의

## 인용 / 문의

HDC Labs T527 Team
