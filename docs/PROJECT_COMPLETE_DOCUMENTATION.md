# 르엘 월패드 NLU — 프로젝트 완전 문서

> **작성일**: 2026-04-28  
> **현재 배포 모델**: `checkpoints/nlu_v28_v72_ensemble.onnx`  
> **현재 성능**: GT-219 94.06% (combo), KoELECTRA fn 97.20%

---

## 1. 프로젝트 개요

르엘 아파트 월패드 AI 음성 어시스턴트의 NLU(Natural Language Understanding) 시스템.  
STT 출력 → 의도 분류 → 기기 제어 / 응답 생성까지 전 파이프라인을 담당.

### 핵심 설계 원칙

- **온디바이스 추론**: T527 ARM Cortex-A55 + Vivante NPU 환경. NPU는 실패(아래 참고), CPU ONNX로 확정.
- **응답 생성 규칙 기반**: LLM 없이 if/else 템플릿으로 "네, 거실 조명을 켭니다" 같은 일관된 응답 보장.
- **5-head 분류**: 단일 intent 대신 fn / exec_type / direction / param_type / judge를 독립적으로 분류 → 조합 폭발 없이 일반화.

---

## 2. 전체 파이프라인

```
사용자 발화 (STT 출력)
    │
    ▼
[preprocess.py]
  - 한글 숫자 → 아라비아 숫자 (이십삼도→23도, 열두시→12시)
  - STT 오류 사전 교정 (남방→난방, 에어콘→에어컨, 오전공기→오전공기질 등)
  - 특수문자/공백 정리
    │
    ▼
[ensemble_inference_with_rules.py]
  - ONNX 추론: nlu_v28_v72_ensemble.onnx (104.9MB)
    - fn head: v72 (간접표현 강화)
    - exec_type head: v28 (기존 패턴 보존)
    - param_direction head: v72 (어두침침 라벨 수정)
    - param_type head: v28
    - judge head: v72
  - apply_post_rules(): 룰 기반 교정
    - 밝게→dir=up, 어둡게→dir=down
    - 엘리베이터 호출→exec=control, dir=on
    - 냉방/제습/송풍 모드→dir=set
    - 알람/모닝콜→fn=schedule_manage
    - 비상 키워드→fn=security_mode
    - 인체상태(목말라/배고파)→fn=unknown
    │
    ▼
[deployment_pipeline_v2.py - extract_rooms()]
  - 발화에서 방 정보 추출 (거실→living, 안방→bedroom_main 등)
    │
    ▼
[dialogue_state_tracker.py - DST]
  - 10초 타임아웃 기반 멀티턴 상태 관리
  - follow-up 처리: "안방도" → 이전 fn/dir 유지, room만 변경
  - correction: "아니 꺼줘" → 이전 fn 유지, direction 변경
  - slot filling: "23도" value 추적, "1도 더 올려" 상대값 계산
    │
    ▼
[response_generator_v2.py]
  - 5-head 결과 + raw_text → 한국어 응답 문장 생성
  - SPECIFIC_PATTERNS: 비유/은유 간접 표현에 공감 응답
  - 모든 응답 규칙 기반 (LLM 없음)
    │
    ▼
최종 출력: {fn, exec_type, param_direction, room, value, response}
```

---

## 3. 모델 아키텍처

### 3.1 CNNMultiHead (`scripts/model_cnn_multihead.py`)

```
입력: token_ids [B, T] (max_len=32)
    │
    ▼
token_emb (ko-sbert-sts WordPiece, vocab ~32K, emb_dim 768, freeze=True)
    │
    ▼
proj: Linear(768 → 256)
drop_in: Dropout(0.1)
    │
    ▼ [B, T, 256] → permute → [B, 256, T]
    
conv1: Conv1d(256, 256, k=3) + BN + ReLU + Drop + residual
conv2: Conv1d(256, 256, k=5) + BN + ReLU + Drop + residual
conv3: Conv1d(256, 256, k=7) + BN + ReLU + Drop + residual
conv4: Conv1d(256, 256, k=3) + BN + ReLU + Drop + residual
    │
    ▼
Global Mean Pooling → [B, 256]
    │
    ▼
5 classification heads (각각 Linear(256→128) + ReLU + Drop + Linear(128→NC))
├── fn (20): light/heat/ac/vent/gas/door/curtain/elevator/security/schedule/weather/news/traffic/energy/home_info/system_meta/market/medical/vehicle/unknown
├── exec_type (5): query_then_respond / control_then_confirm / query_then_judge / direct_respond / clarify
├── param_direction (9): none/up/down/set/on/off/open/close/stop
├── param_type (5): none/temperature/brightness/mode/speed  [※현재 규칙으로 강제=none]
└── judge (5): none/outdoor_activity/clothing/air_quality/cost_trend  [※현재 미사용]
```

**파라미터**: ~1.5M trainable (임베딩 freeze) / ~50M total  
**추론 속도**: CPU 0.32ms/utterance (Intel i7)  
**모델 크기**: ~104MB (ONNX)

### 3.2 임베딩 선택 이유

ko-sbert-sts 임베딩이 우세한 이유:
- 한국어 의미 유사도 학습 → 비슷한 의미 발화의 임베딩이 가깝게 배치
- freeze해도 CNN이 충분히 분류 가능한 특징 추출
- KLUE-RoBERTa emb 실험(v65): 89.8% → ko-sbert(94.3%) 대비 열세
- unfreeze 실험(v64): 90.5% → frozen(94.3%) 대비 열세 (24.5K 데이터에서 overfitting)

---

## 4. 학습 데이터

### 4.1 데이터 구성 (train_final_v72.json, 32,809개)

| 출처 | 수량 | 설명 |
|------|------|------|
| GT 시나리오 증강 | ~19,000 | 르엘 219개 시나리오 × avg 94 변형 |
| KoELECTRA pseudo-labeled | ~13,000 | fn=KE원본, exec/dir=v28예측 |
| MASSIVE 한국어 (OOD) | ~4,800 | unknown 클래스 학습용 |
| HA 영어→한국어 번역 | ~583 | Home-Assistant-Requests |
| llm_paraphrases | ~1,846 | GPT 패러프레이즈 (v71에서 추가) |
| indirect_expressions_v2 | ~3,961 | 비유/은유 간접 표현 (v72에서 추가) |

### 4.2 학습 설정 (v46 recipe, v71/v72에도 동일 적용)

```python
# CNN 4L, d=256, dropout=0.1
# Mixup: 같은 fn의 두 발화를 30% 확률로 선형 보간
# 30 epochs, batch=64, lr=1e-3 (AdamW)
# Cosine scheduler
# Early stop: val combo acc
```

### 4.3 데이터 구성 주의사항

- **param_type / judge head**: 학습 데이터에 라벨 있지만 파이프라인에서 rule로 강제 override
  - param_type: open/close/stop → none, query/direct → none
  - judge: 모든 케이스 none으로 강제 (규칙 기반 exec_type으로 대체)
- **Test Suite (ts) vs KoELECTRA (KE) 트레이드오프**: 구조적 한계
  - GT(르엘) 데이터: exec/dir 패턴 정확 → TS 높음
  - KoELECTRA: 다양한 fn 표현 → KE 높음
  - 두 데이터의 exec/dir 라벨 체계가 미묘하게 다름 → 동시 최적화 불가

---

## 5. 앙상블 전략

### 현재 배포 앙상블 (nlu_v28_v72_ensemble.onnx)

| Head | 사용 모델 | 이유 |
|------|-----------|------|
| fn | v72 | 간접/비유 표현 강화 |
| exec_type | v28 | GT 패턴 보존, 안정적 |
| param_direction | v72 | 어두침침 dir=on 수정 |
| param_type | v28 | (어차피 rule override) |
| judge | v72 | (어차피 rule override) |

### 앙상블 방식

`scripts/export_ensemble_onnx.py`로 생성. 두 ONNX를 단일 그래프로 통합, head별로 서로 다른 모델 출력을 선택.

---

## 6. 버전 이력 — 상세

### 6.1 Phase 1: 초기 개발 (2026-04-16~17)

| 날짜 | 내용 |
|------|------|
| 04-16 | 르엘 GT 219개 시나리오 파싱 (260330 엑셀) |
| 04-16 | 아키텍처 결정: flat intent(136+) vs multi-head(fn19×dir9) → **multi-head 채택** |
| 04-16 | 외부 데이터 조사: MASSIVE ko-KR, HA-Requests, Fluent Speech, SNIPS |
| 04-17 | CNN 4L + 5-head 설계 확정 |

### 6.2 Phase 2: v1~v9 — 초기 학습 및 데이터 보강 (04-18)

| 버전 | 핵심 | GT fn | combo |
|------|------|-------|-------|
| v1 | CNN 5-head 최초 학습 (19K 데이터) | 96.5% | 90.8% |
| v2 | MASSIVE 한국어 5K + judge/clarify 데이터 추가 | 95.1% | 89.1% |
| v3 | unknown class 도입, GT 증강 avg 25개/시나리오 | — | 89.5% |
| v4 | 219개 원본 GT 기준 평가 시작 (이전: 142개 파생) | 96.3% | 86.3% |
| v5 | unknown weight 0.3 조정 → 거부율 하락. 폐기 | — | — |
| v6 | targeted hard negatives 추가 | 97.3% | 86.5% |
| **v7** | 고다양성 증강 (avg 94 변형/시나리오), 24K 데이터 | **100%** | **94.5%** |
| v9 | 부족 fn 집중 증강 (elevator, gas, door) | 99.1% | 94.9% |

### 6.3 Phase 3: v10~v28 — 반복 개선 및 Test Suite 구축 (04-19)

| 버전 | 핵심 |
|------|------|
| v10~v19 | 96개→78개→92개 오류 전수 수정. "거실 불 꺼줘"→ac 오분류 수정 등. val combo 94.9% (v19 최고) |
| v20~v21 | judge/param head 수정 (반팔→clothing, 미세먼지→air_quality) |
| v22~v27 | 실험 → regression 발생, v28에서 안정화 |
| **v28** | **Test Suite 3,043개 combo 100% 달성. TS 전용 최적 모델** |
| v29~v33 | 추가 실험 전부 v28 대비 regression → v28 확정 |

**v28 핵심 변경:**
- Test Suite 3,043개 확정 (GT 시나리오 × 변형)
- STT 전처리 사전 54개 (`preprocess.py`)
- confidence fallback (conf<0.5 → unknown)
- ONNX export: `nlu_v28_final.onnx`

### 6.4 Phase 4: v34~v53 — 일반화 개선 실험 (04-19~20)

| 버전 | 기법 | TS | KE fn | 결론 |
|------|------|:--:|:-----:|------|
| **v34** | **pseudo-labeling** (fn=KE원본, exec/dir=v28예측) | 93.6% | **96.8%** | **핵심 돌파구 (+21.3%p)** |
| v36~v39 | fix 추가, loss weight 조정 | 90.9% | 97.3% | 소규모 패치 항상 regression |
| v40 | Knowledge Distillation from ensemble | 92.3% | 97.2% | — |
| v41~v44 | agreement filter, 2-phase, 어휘 추가, dropout 감소 | 91~92% | 97% | v34보다 나빠짐 |
| v45 | Label Smoothing 0.1 | 90.5% | 97.4% | — |
| **v46** | **Mixup** (같은 fn 30%) | 91.1% | **97.8%** | **KE 일반화 최고** |
| v47 | Mixup+LS | 91.6% | 97.3% | v46보다 나빠짐 |
| v48 | 어휘추가+Mixup | 90.7% | 97.4% | v46보다 나빠짐 |
| v49 | CNN+Attention | 91.5% | 97.5% | attention 추가 효과 미미 |
| v50 | R-Drop | 46.9% | — | **붕괴** |
| v51 | SupCon+Mixup | 91.4% | 97.6% | v46 대비 미미 |
| v52 | CutMix | 90.4% | 97.5% | 토큰 교체 의미 파괴 |
| v53 | Focal+Mixup | 90.5% | 97.7% | TS 하락 |

**앙상블 최적 (이 시점):**
- fn=v46, exec/dir=v28 (전략 B): **TS 94.3%, KE 97.8%** → `nlu_v28_v46_ensemble.onnx`

### 6.5 Phase 5: v54~v68 — 단일 모델 추가 실험 (04-21)

| 버전 | 기법 | 결과 |
|------|------|------|
| v54 | Self-training R2 | v28↔v46 차이 2%뿐, ROI 없음 |
| v55 | KD from ensemble | 92.1%/97.7%, soft label 재현 불가 |
| v56 | Two-stage (fn/judge freeze) | 91.2%/96.9%, backbone 변경이 fn 훼손 |
| v57 | Wider d=384 | 90.2%/97.2%, 24.5K 데이터 대비 과다 |
| v58 | Targeted augmentation | 91.4%/97.5%, 소규모 패치 분포 왜곡 |
| v59 | Head masking | 83.0%/97.2%, backbone 학습 저해 |
| v60 | Model Soup | α=0에서 최적 (=순수 v46), weight space 비볼록 |
| v61 | Warm-start from v28 | 92.4%/97.3%, 초기화 무관 동일 수렴 |
| v62 | Multi-seed avg | 92.0%/97.6%, v46 seed=42가 lucky |
| **v63** | Conformer 2L | **79.6%/94.8%, CNN>Conformer at 24.5K** |
| v64 | Unfreeze embeddings | 90.5%/97.3%, frozen이 더 나음 |
| v65 | KLUE-RoBERTa emb | 89.8%/97.3%, ko-sbert 우세 |
| **v68** | 라벨 오류 46건 수정 후 재학습 | **90.7%/97.5%, 효과 제한적** |

**결론**: v46이 단일 모델 최적. 앙상블 v28+v46 유지.

### 6.6 Phase 6: iter1~iter9 — 후처리 룰 기반 개선 (04-21)

| iter | 기법 | TS combo | 변화 |
|------|------|:--------:|------|
| iter1 | Retrieval Hybrid (GT pool cosine) | 91.49% | 실패 (pool sparse) |
| iter2 | 라벨 정제 v2 (24건) + 전략 재비교 | 93.59% | v28 "밝게→down" 오학습 확인 |
| iter3 | v28b (patch retrain) + rules | 92.74% | 밝게 교정, 전체 -3.6%p regression |
| **iter8** | **후처리 rule + preprocess 확장** | **94.38%** | **모델 변경 없이 +0.85%p ★** |
| iter9 | rule 14개 + DST slot filling | 95.76% | +2.23%p (DST 포함 시) |

**iter8 핵심 rules:**
- 알람/모닝콜 → fn=schedule_manage (device kw 없을 때)
- 날씨/뉴스/병원 키워드 → 해당 fn으로 교정 (unknown 탈출)
- 인체상태(목말라/배고파) → unknown
- 비상 상황 키워드 → security_mode

### 6.7 Phase 7: v70~v72 — 데이터 품질 개선 + 간접 표현 (04-21~28)

| 버전 | 날짜 | 내용 | 결과 |
|------|------|------|------|
| v70 | 04-21 | 대규모 1,180건 라벨 수정 후 재학습 | 90.04% — 부분 수정으로 데이터 일관성 파괴 |
| **v71** | **04-28** | **어두침침 52개 dir=set→on 수정 + llm_paraphrases 1,846개 + indirect 52개 추가** | **GT 94.06% (+0.45%p)** |
| **v72** | **04-28** | **비유/은유 3,961개 종합 간접 표현 데이터 추가** | **간접 표현 20/21 통과** |

**v72 간접 표현 커버리지 (`data/indirect_expressions_v2.json`, 3,961개):**
- 추위 비유 (heat/on): 얼어 죽겠다, 냉동실 같아, 이가 딱딱 부딪혀, 시베리아 같아
- 더위 비유 (ac/on): 쪄 죽겠어, 사우나, 찜통이야, 땀이 뻘뻘, 더위 먹겠다
- 어두움 (light/on): 앞이 안 보여, 동굴 같아, 암흑이야
- 눈부심 (light/off): 눈이 부셔, 눈이 따가워, 눈을 못 뜨겠어
- 공기 나쁨 (vent/on): 머리가 띵해, 공기가 탁해, 밀폐된 것 같아
- 방 prefix 5개 × 어미 변형 4~5개로 증강

**앙상블 최종:** `nlu_v28_v72_ensemble.onnx`

---

## 7. 성능 현황

### 7.1 최종 성능

| 지표 | v28+v46 | v28+v71 | v28+v72 |
|------|:-------:|:-------:|:-------:|
| GT-219 combo | 93.61% | **94.06%** | ≈94% |
| Test Suite combo (후처리 포함) | 94.38% | ~94.38% | ~94.38% |
| KoELECTRA fn | 97.79% | 97.20% | 97.20%~97.33% |
| 간접 표현 | 9/21 | 20/21 | **20/21+** |
| 어두침침→on | ✗ | ✓ | ✓ |

### 7.2 알려진 한계 및 실패 케이스

1. **"더위 먹겠다"** → ac_control + dir=none (간혹 on으로 못 분류)
2. **TS↔KE 트레이드오프**: 구조적 한계 (데이터 분포 차이)
3. **param_type/judge**: 모델이 예측하지만 파이프라인에서 무시
4. **복합 발화** 분리는 regex (`_split_compound`)로만 처리, 모델 미관여
5. **NPU 배포 불가**: CNN+ko-sbert 구조가 Acuity/Vivante NPU 양자화 시 정확도 0

---

## 8. 배포 파일 구조

```
t527-nlu/
├── checkpoints/
│   ├── nlu_v28_v72_ensemble.onnx   ← 현재 배포 모델 (104.9MB)
│   ├── cnn_multihead_v28.pt         ← GT 전용 모델
│   ├── cnn_multihead_v46.pt         ← 일반화 최고 단일 모델
│   ├── cnn_multihead_v71.pt         ← 간접표현 v1
│   ├── cnn_multihead_v72.pt         ← 간접표현 v2 (현재 최신)
│   └── nlu_v28_v46_ensemble.onnx   ← 이전 배포 (백업)
├── tokenizer/                        ← ko-sbert WordPiece tokenizer
├── scripts/
│   ├── model_cnn_multihead.py        ← 모델 정의 + HEAD_CLASSES
│   ├── preprocess.py                 ← STT 전처리
│   ├── ensemble_inference_with_rules.py  ← ONNX 추론 + 후처리 룰
│   ├── dialogue_state_tracker.py     ← DST 멀티턴
│   ├── deployment_pipeline_v2.py    ← 전체 파이프라인 클래스
│   ├── response_generator_v2.py     ← 응답 생성 (규칙 기반)
│   ├── build_indirect_expressions_v2.py ← 간접 표현 데이터 생성기
│   ├── build_train_v71.py           ← v71 데이터 빌드
│   └── train_v71.py / train_v72.py  ← 학습 스크립트
├── data/
│   ├── train_final_v72.json          ← 현재 학습 데이터 (32,809개)
│   ├── train_final_v71.json          ← v71 (29,015개)
│   ├── indirect_expressions_v2.json  ← 비유 표현 3,961개
│   ├── test_suite.json               ← TS 평가셋 (3,043개)
│   ├── gt_known_scenarios_v2.json    ← GT 219개
│   └── ruel_scenarios_final.csv      ← 원본 르엘 시나리오 CSV
├── test_interactive.py               ← 통합 테스트 도구 (API 포함)
└── docs/
    ├── CHANGELOG.md                  ← 버전별 변경 이력
    ├── VERSION_LOG.md               ← 상세 실험 기록
    └── PROJECT_COMPLETE_DOCUMENTATION.md  ← 이 파일
```

---

## 9. 응답 생성 시스템

### 9.1 구조

`response_generator_v2.py` — 완전 규칙 기반, LLM 없음.

```python
generate_response_v2(multihead, raw_text=None)
# multihead: {fn, exec_type, param_direction, room, value, old_value}
# raw_text: 원본 발화 (SPECIFIC_PATTERNS 매칭용)
```

### 9.2 응답 결정 흐름

1. **SPECIFIC_PATTERNS 우선** (비유/은유 공감 응답)
   - 예: "얼어 죽겠어" → "많이 추우시군요. 난방을 켜드릴게요."
2. **fn × exec_type × direction 조합 매핑**
   - 예: fn=light_control, dir=on, room=living → "네, 거실 조명을 켭니다."
3. **room 추가 포맷팅**
   - room='bedroom_main' → 응답에 "안방" 삽입

### 9.3 비유/은유 SPECIFIC_PATTERNS (response_generator_v2.py)

```python
# 추위 비유 → 난방 켜기
(r'얼어\s*죽겠|몸이\s*꽁꽁|이가\s*딱딱|시베리아\s*같|냉동실\s*같', '많이 추우시군요. 난방을 켜드릴게요.'),
# 더위 비유 → 에어컨 켜기  
(r'쪄\s*죽겠|사우나|찜통|땀이\s*(?:뻘뻘|철철)', '많이 더우시군요. 에어컨을 켜드릴게요.'),
# 어두움 → 조명 켜기
(r'동굴\s*같|암흑|앞이\s*안\s*보여', '어두우시군요. 조명을 켜드릴게요.'),
# 눈부심 → 조명 끄기/낮추기
(r'눈이\s*(?:부셔|따가워|아파)', '눈이 많이 부시시군요. 조명을 낮춰드릴게요.'),
# 공기 나쁨 → 환기
(r'공기가\s*(?:탁|나빠|죽어)', '공기가 좋지 않으시군요. 환기시스템을 켜드릴게요.'),
```

---

## 10. API 연동 (test_interactive.py)

### 10.1 날씨/대기질 — Open-Meteo (무료, 키 없음)

- 위치: 서초구 (LAT=37.4837, LON=127.0324)
- 현재 온도/체감/습도/풍속/날씨코드
- 일별 최저/최고/강수확률/UV
- 대기질: PM10, PM2.5

### 10.2 뉴스 — 경향신문 RSS (무료, 키 없음)

- `https://www.khan.co.kr/rss/rssdata/total_news.xml`
- 최신 3개 뉴스 타이틀 추출

### 10.3 OO/00 플레이스홀더 채우기

`fill_placeholders(resp, weather, air)`: 응답 템플릿의 OO/00 → 실제 API 값으로 치환

### 10.4 ApiCache (60초 TTL)

날씨/대기질 API 호출을 60초 캐싱 → 반복 요청 시 API 재호출 방지

---

## 11. 멀티턴 대화 (DialogueStateTracker)

10초 타임아웃 기반. 주요 처리 패턴:

| 패턴 | 예시 | 처리 |
|------|------|------|
| room follow-up | "안방도" | 이전 fn/exec/dir 유지, room 변경 |
| device follow-up | "난방도" | fn=NLU예측, exec/dir 이전 것 상속 |
| confirm | "응", "해줘" | 이전 fn/dir 유지, exec=control 승격 |
| correction | "아니 꺼줘" | fn 유지, direction만 변경 |
| there-too | "거기도" | 이전 fn/exec/dir/room 전체 유지 |
| bare verb | "켜", "꺼줘" | fn 상속, exec=control, dir=해당 방향 |
| relative value | "1도 더 올려" | prev_value + delta 계산 |
| bare direction | "더", "조금" | fn 상속, direction=up/down, value delta |

---

## 12. NPU 배포 시도 및 실패 기록

T527 NPU(Vivante VIP9000NANOSI+)에 NLU 모델을 올리려 시도했으나 전부 실패:

| 방식 | 증상 | 원인 |
|------|------|------|
| Pure Transformer (BERT류) | 양자화 후 정확도 0 | Vivante NPU Softmax/GeLU 지원 제한 |
| TextConformer (Conformer+KD) | val 98.7%, NPU 불일치 | 동적 shape, attention 연산 |
| CNN+ko-sbert (v7~v28) | PTQ/QAT 전부 NPU 불일치 | 임베딩 lookup 양자화 오류 |
| CNN 순수 (임베딩 없이) | 정확도 자체가 낮음 | 임베딩 없이 분류 불가 |

**결론: CPU ONNX 추론으로 확정. 0.32ms/utterance로 월패드 요구사항 충족.**

---

## 13. 테스트 방법

```bash
cd /home/nsbb/travail/claude/T527/t527-nlu

# 대화형 테스트 (API 포함)
python3 test_interactive.py

# verbose 모드 (exec_type, rooms, DST 출력)
python3 test_interactive.py -v

# 복합 발화 처리
python3 test_interactive.py -c

# 배치 테스트
echo "거실 불 켜줘" | python3 test_interactive.py
cat test_cases.txt | python3 test_interactive.py

# GT-219 평가
cd scripts && python3 eval_v2_ruel_scenarios.py

# Test Suite 평가
cd scripts && python3 run_test_suite.py
```

---

## 14. 향후 과제

### 단기 (다음 버전)

1. **v73**: "더위 먹겠다" → ac_control + dir=on 고정 (여전히 간헐적 dir=none)
2. 간접 표현 커버리지 추가 (계절별, 상황별)
3. rule 기반 응답에서 room 매핑 누락 케이스 수정

### 중기

4. **Android 포팅** (현재 논의 중):
   - ONNX Runtime for Android (`onnxruntime-android` AAR)
   - WordPiece tokenizer: HuggingFace tokenizers-android 또는 직접 포팅
   - response_generator_v2.py → Kotlin 변환
   - API 연동: Retrofit/OkHttp

5. **DST 개선**: 5턴 이상 긴 대화, 다른 방의 같은 기기 제어

### 장기

6. SAP (Semantic Action Parser) — 별도 `plan` 파일 참조
7. API 확장: 에너지 사용량, 단지 내 정보

---

## 15. 핵심 교훈

1. **pseudo-labeling이 직접 라벨 매핑보다 21%p 효과적** (v33 vs v34)
2. **소규모 패치 데이터는 항상 regression** — v29~v33, v58 모두 동일
3. **Mixup이 NLP에서 ROI 최고** — SupCon, Focal, CutMix 대비 압도적
4. **Model Soup 실패** — v28↔v46 weight space 비볼록, prediction-level 앙상블만 가능
5. **CNN > Conformer (24.5K 데이터)** — 데이터 부족 시 간단한 구조가 유리
6. **후처리 룰이 0.85%p 무료 향상** — 모델 재학습 없이 iter8에서 달성
7. **TS↔KE 트레이드오프는 구조적** — 두 데이터의 exec/dir 라벨 체계가 근본적으로 다름
8. **임베딩 freeze + CNN이 최적** — unfreeze, KLUE-RoBERTa emb 모두 열세
