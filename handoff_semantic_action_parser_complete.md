# On-Device Semantic Action Parser — 기술 핸드오프 문서 (Complete)

> **프로젝트**: 르엘 어퍼하우스 AI 월패드 서비스  
> **목적**: 사용자 자연어 발화 → 실행 가능한 구조체(Action Struct) 변환 → 디바이스 제어/정보 조회/조건 판단  
> **타겟 하드웨어**: Allwinner T527 NPU (Pegasus toolchain, NB format)  
> **작성일**: 2026-04-09

---

## 1. 프로젝트 배경 및 문제 정의

### 1.1 프로젝트 개요

르엘 어퍼하우스 AI 월패드 서비스는 사용자가 음성으로 스마트홈 기기를 제어하고, 날씨/뉴스/교통 등 생활 정보를 조회하며, 상황 판단("세차해도 되나?")까지 받을 수 있는 온디바이스 AI 시스템이다.

전체 시나리오는 233개 발화-응답 쌍으로 정의되어 있으며(엑셀 시나리오 문서 참조), 다음 도메인을 포함한다:

- **디바이스 제어**: 조명, 난방, 에어컨, 환기, 가스, 도어락, 전동커튼, 엘리베이터
- **정보 조회**: 날씨/미세먼지, 뉴스, 교통, 에너지 사용량, 증시, 유가, 의료/병원, 단지정보
- **설정/방범**: 환경설정(볼륨, 밝기), 외출/재택 모드, 모닝콜, 비상
- **판단형 응답**: 야외활동 적합성, 옷차림 추천, 공기질 판단, 비용 추세

### 1.2 제약 조건

- **하드웨어**: Allwinner T527 NPU
- **변환 툴체인**: Pegasus (ONNX → INT8/UINT8 → NB)
- **서버 사용 불가**: 전체 파이프라인이 온디바이스에서 동작해야 함
- **NPU 호환 op만 사용 가능**: Conv, Linear, ReLU 계열 확인됨. LayerNorm, Softmax, Multi-Head Attention은 ASR Conformer NB 변환 성공으로 호환 확인됨. Sigmoid(SE block)는 INT8 동적 범위 이슈 있음.
- **BERT 계열 NB 변환 불가**: BERT/KoELECTRA 전체 모델은 Pegasus에서 NB 변환이 되지 않음. 그러나 **ASR Conformer는 NB 변환 성공** — 이것이 TextConformer 접근의 근거.

### 1.3 설계 과정에서의 주요 의사결정 (대화 기반)

이 구조는 다음과 같은 단계적 논의를 통해 도출되었다.

#### 1.3.1 출발점: CNN 기반 NLU

초기에는 프리트레인 임베딩(KoELECTRA/KLUE-RoBERTa에서 추출) + CNN 1레이어 구조로 인텐트 분류를 시도했다. 이 구조는:
- 짧은 명령문("거실 불 꺼줘")에서는 충분한 성능
- T527 NPU에서 가볍게 구동 가능
- 프리트레인 임베딩이 의미 표현력을 제공하므로 학습 데이터가 적어도 일반화 가능

그러나 테스트 결과 CNN 단독으로는 **성능이 부족**했다.

#### 1.3.2 Transformer 대안 검토

- **BERT/KoELECTRA**: NLU 성능은 좋지만 T527 Pegasus에서 **NB 변환 불가**
- **RNN/LSTM**: 순차 연산이라 NPU 병렬화 불가, 양자화 시 게이트 연산에서 INT8 정밀도 이슈
- **Conformer**: ASR용이지만 **NB 변환 성공 확인됨** → 텍스트용으로 변형 가능

결론: **TextConformer** — ASR Conformer의 구조를 유지하되 입력을 스펙트로그램→토큰 임베딩으로 변경, Conformer 블록 수 축소(23→3), 출력을 CTC→분류 head로 교체.

#### 1.3.3 멀티턴 처리

멀티턴 대화 처리를 위해 LSTM을 별도로 두는 대신, **이전 턴의 인텐트를 스페셜 토큰으로 현재 입력에 prepend**하는 방식을 채택했다.

- `[NONE] 거실 불 켜줘` → 싱글턴
- `[LIGHT_ON] 안방도` → 멀티턴 (이전 컨텍스트 참조)
- `[LIGHT_ON] 에어컨도 켜줘` → 멀티턴이지만 컨텍스트 무시 (모델이 학습)

싱글턴/멀티턴 구분 로직을 별도로 두지 않고, **항상 이전 fn 결과를 넘기되 세션 타임아웃(10초) 후 `[NONE]`으로 리셋**한다. 모델이 컨텍스트 참조 여부를 자체적으로 학습한다.

#### 1.3.4 인텐트-슬롯 NLU의 근본적 한계 인식

233개 시나리오를 전부 인텐트로 정의하면 사실상 **거대한 룰베이스**가 된다. 모델이 하는 건 "어떤 룰을 적용할지 고르는 것"뿐이고, 학습하지 않은 새로운 조합에 대응할 수 없다.

핵심 질문: "이게 AI인가, 그냥 룰베이스 아닌가? AI는 안 본 걸 잘해야 AI인데."

#### 1.3.5 해결: Compositional Semantic Action Parser

인텐트 분류 대신, 의미 구조를 구성하는 **독립적인 축(axis)을 병렬로 예측**하는 방식으로 전환.

- 인텐트 60개를 하나의 분류 문제로 풀지 않고
- device(10) × action(5) × param(8) 등 각 축을 독립적으로 예측
- 학습에 없던 조합도 각 축의 독립 예측으로 일반화 가능
- Prior art: Facebook **MTOP** (Multi-domain Task-Oriented Parsing, 2020)

MTOP과의 차이: MTOP은 오토리그레시브 디코더로 트리 구조를 순차 생성하지만, 본 구조는 **병렬 FC head**로 플랫 구조체를 동시 예측한다. 스마트홈 도메인에서는 트리 깊이가 1~2이므로 플랫 구조로 표현력 손실이 없으며, NPU에서 구동 가능한 형태가 된다.

#### 1.3.6 Function Call 통합

모델 출력에 **실행 타입(exec_type)**을 추가하여 온디바이스 function calling을 구현한다:

- `query_then_respond`: API 조회 → 값 삽입 → 응답 ("지금 온도 어때?")
- `control_then_confirm`: 디바이스 제어 → 확인 응답 ("거실 불 꺼줘")
- `query_then_judge`: API 조회 → 조건 판단 → 조건 응답 ("세차해도 되나?")
- `direct_respond`: 고정 응답 ("너 이름이 뭐야?")
- `clarify`: 정보 부족 → 재질문 ("불 켜줘" - 공간 미지정)

이를 통해 LLM 서버 없이도 자연어 → 함수 호출 → 응답 생성의 전체 파이프라인을 온디바이스에서 처리한다.

---

## 2. 시스템 아키텍처

### 2.1 전체 파이프라인

```
┌──────────────────────────────────────────────────────┐
│                    음성 입력                           │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│         STT (Conformer, T527 NPU)                    │
│         음성 → 텍스트 변환                              │
└──────────────┬───────────────────────────────────────┘
               ▼
┌──────────────────────────────────────────────────────┐
│    Semantic Action Parser (TextConformer, T527 NPU)  │
│    텍스트 → Action Struct 변환                         │
│                                                      │
│    ┌─────────────────────────────────┐               │
│    │   Pretrained Embedding Layer    │               │
│    │   (KLUE-RoBERTa base에서 추출)   │               │
│    └──────────────┬──────────────────┘               │
│                   ▼                                  │
│    ┌─────────────────────────────────┐               │
│    │   TextConformer Encoder         │               │
│    │   (Conformer Block × 3)         │               │
│    └──────────────┬──────────────────┘               │
│                   ▼                                  │
│    ┌─────────────────────────────────┐               │
│    │   Multi-Head FC Classifier      │               │
│    │   (7개 독립 Linear head)         │               │
│    └──────────────┬──────────────────┘               │
└───────────────────┬──────────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────────┐
│              Action Executor (CPU, 룰 엔진)           │
│                                                      │
│   exec_type에 따라 분기:                               │
│                                                      │
│   query_then_respond ──→ fn 호출 → 값 → 템플릿 → TTS │
│   control_then_confirm ──→ fn 호출 → 확인 응답 → TTS  │
│   query_then_judge ──→ fn 호출 → 판단 → 조건 응답 → TTS│
│   direct_respond ──→ 고정 응답 → TTS                  │
│   clarify ──→ 재질문 응답 → TTS                       │
└──────────────────────────────────────────────────────┘
```

### 2.2 멀티턴 컨텍스트 처리

```
1턴: "거실 불 켜줘"
  → 입력: [NONE] 거실 불 켜줘
  → 출력: {fn: light_control, room: living, ...}
  → 세션에 fn=LIGHT_CONTROL 저장

2턴 (5초 내): "안방도"
  → 입력: [LIGHT_CONTROL] 안방도
  → 출력: {fn: light_control, room: bedroom, ...}

3턴 (5초 내): "에어컨도 켜줘" (새 맥락)
  → 입력: [LIGHT_CONTROL] 에어컨도 켜줘
  → 출력: {fn: ac_control, room: living, ...}
  → 모델이 이전 컨텍스트 무시하도록 학습됨

4턴 (10초 이후): "날씨 어때?"
  → 입력: [NONE] 날씨 어때?   (타임아웃으로 리셋)
```

---

## 3. 모델 상세 설계

### 3.1 TextConformer Encoder

ASR Conformer에서 입력 레이어만 변경한 구조. ASR Conformer가 T527 NB 변환에 성공했으므로, 동일 op 구성을 유지하면 변환 가능.

```python
class TextConformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=3):
        self.embedding = nn.Embedding(vocab_size, d_model)  # 프리트레인 가중치 로드
        self.context_embedding = nn.Embedding(17, d_model)  # 16 fn + NONE
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                num_heads=4,
                ff_dim=1024,
                kernel_size=31,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, token_ids, context_id):
        x = self.embedding(token_ids)                      # [B, seq, 256]
        ctx = self.context_embedding(context_id)           # [B, 256]
        cls = self.cls_token.expand(x.size(0), -1, -1)     # [B, 1, 256]
        x = torch.cat([cls, ctx.unsqueeze(1), x], dim=1)   # [B, 2+seq, 256]
        x = self.pos_enc(x)
        for block in self.conformer_blocks:
            x = block(x)
        return x  # [B, 2+seq, 256], x[:, 0, :] = CLS
```

**Conformer Block 내부 구조** (ASR과 동일):
```
Feed Forward Module (1/2)
    ↓
Multi-Head Self-Attention Module
    ↓
Convolution Module (Depthwise + Pointwise)
    ↓
Feed Forward Module (1/2)
    ↓
LayerNorm
```

### 3.2 Multi-Head Classifier

```python
class MultiHeadClassifier(nn.Module):
    def __init__(self, d_model=256):
        self.exec_head   = nn.Linear(d_model, 5)   # 실행 타입
        self.fn_head     = nn.Linear(d_model, 15)   # 함수 선택
        self.room_head   = nn.Linear(d_model, 8)    # 공간
        self.param_head  = nn.Linear(d_model, 8)    # 파라미터 타입
        self.api_head    = nn.Linear(d_model, 7)    # API 종류
        self.judge_head  = nn.Linear(d_model, 5)    # 판단 유형
        self.multi_head  = nn.Linear(d_model, 2)    # 단일/복합 액션
        
    def forward(self, cls_embedding):
        return {
            "exec_type":    self.exec_head(cls_embedding),
            "fn":           self.fn_head(cls_embedding),
            "room":         self.room_head(cls_embedding),
            "param_type":   self.param_head(cls_embedding),
            "api":          self.api_head(cls_embedding),
            "judge":        self.judge_head(cls_embedding),
            "multi_action": self.multi_head(cls_embedding),
        }
```

### 3.3 Value Pointer Head

수치 추출("23도", "오후 3시")은 시퀀스에서 해당 토큰 위치를 가리키는 pointer 방식.

```python
class ValuePointerHead(nn.Module):
    def __init__(self, d_model=256):
        self.has_value = nn.Linear(d_model, 2)
        self.start_pointer = nn.Linear(d_model, 1)
        self.end_pointer = nn.Linear(d_model, 1)
    
    def forward(self, cls_embedding, encoder_output):
        has_val = self.has_value(cls_embedding)
        start_scores = self.start_pointer(encoder_output).squeeze(-1)
        end_scores = self.end_pointer(encoder_output).squeeze(-1)
        return has_val, F.softmax(start_scores, dim=-1), F.softmax(end_scores, dim=-1)
```

### 3.4 전체 모델 + 사이즈 추정

```python
class SemanticActionParser(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, num_layers=3):
        self.encoder = TextConformerEncoder(vocab_size, d_model, num_layers)
        self.classifier = MultiHeadClassifier(d_model)
        self.value_pointer = ValuePointerHead(d_model)
    
    def forward(self, token_ids, context_id):
        full_output = self.encoder(token_ids, context_id)
        cls_vec = full_output[:, 0, :]
        heads = self.classifier(cls_vec)
        has_val, start_ptr, end_ptr = self.value_pointer(cls_vec, full_output[:, 2:, :])
        heads["has_value"] = has_val
        heads["start_pointer"] = start_ptr
        heads["end_pointer"] = end_ptr
        return heads
```

| 컴포넌트 | 파라미터 수 | 비고 |
|---|---|---|
| 토큰 임베딩 (32000 × 256) | 8.2M | 프리트레인 가중치, freeze 가능 |
| 컨텍스트 임베딩 (17 × 256) | 4.4K | |
| Conformer Block × 3 | ~3M | ASR과 동일 op |
| Classification Heads (7개) | ~15K | Linear only |
| Value Pointer Head | ~1.5K | Linear only |
| **합계 (임베딩 포함)** | **~11.2M** | |
| **합계 (임베딩 freeze 시)** | **~3M** | NPU에 올라가는 부분 |

INT8 양자화 시 약 **3~11MB**. T527 NPU에서 충분히 구동 가능.

---

## 4. Head 클래스 정의

### 4.1 exec_type_head (5 classes)

| ID | 라벨 | 설명 | 예시 |
|---|---|---|---|
| 0 | `query_then_respond` | 조회 → 값 삽입 → 응답 | "지금 온도 어때?" |
| 1 | `control_then_confirm` | 제어 실행 → 확인 응답 | "거실 불 꺼줘" |
| 2 | `query_then_judge` | 조회 → 조건 판단 → 응답 | "세차해도 되나?" |
| 3 | `direct_respond` | 고정 응답 | "너 이름이 뭐야?" |
| 4 | `clarify` | 정보 부족 → 재질문 | "불 켜줘" (공간 미지정) |

### 4.2 fn_head (15 classes)

| ID | 함수명 | 도메인 |
|---|---|---|
| 0 | `light_control` | 조명 ON/OFF/DIM/색온도 |
| 1 | `heat_control` | 난방 ON/OFF/온도조절 |
| 2 | `ac_control` | 에어컨 ON/OFF/온도/모드/풍량/풍향 |
| 3 | `vent_control` | 환기 ON/OFF/풍량/모드 |
| 4 | `gas_control` | 가스 밸브 열기/닫기 |
| 5 | `door_control` | 도어락/공동현관 열기 |
| 6 | `curtain_control` | 전동커튼 열기/닫기/정지 |
| 7 | `elevator_call` | 엘리베이터 호출 |
| 8 | `security_mode` | 외출/재택 모드 |
| 9 | `schedule_manage` | 예약/모닝콜/스케줄 |
| 10 | `weather_query` | 날씨/미세먼지 조회 |
| 11 | `news_query` | 뉴스 검색/브리핑 |
| 12 | `traffic_query` | 교통정보/버스/소요시간 |
| 13 | `energy_query` | 에너지 사용량/원격검침 |
| 14 | `info_query` | 단지정보/의료/유가/증시/차량/설정/매뉴얼 |

### 4.3 room_head (8 classes)

| ID | 라벨 | 매핑 |
|---|---|---|
| 0 | `living` | 거실 |
| 1 | `kitchen` | 주방/식탁 |
| 2 | `bedroom_main` | 안방/침실 |
| 3 | `bedroom_sub` | 각실/작은방 |
| 4 | `all` | 전체 |
| 5 | `external` | 외부/밖 (날씨 등) |
| 6 | `none` | 해당 없음 |
| 7 | `ambiguous` | 공간 미지정 → clarify |

### 4.4 param_type_head (8 classes)

| ID | 라벨 | 예시 |
|---|---|---|
| 0 | `none` | 파라미터 없음 |
| 1 | `temperature` | 23도, 올려, 내려 |
| 2 | `brightness` | 밝게, 어둡게, 1단계, 은은하게 |
| 3 | `mode` | 자동, 제습, 송풍, 외출, 취침 |
| 4 | `speed` | 강풍, 약풍, 세게, 줄여 |
| 5 | `direction` | 고정, 회전 |
| 6 | `time` | 오전 10시, 30초, 내일, 매일 |
| 7 | `keyword` | 경제, 삼성전자, 소아과, 차량번호 |

### 4.5 api_head (7 classes)

| ID | 라벨 | 소스 |
|---|---|---|
| 0 | `none` | API 불필요 (고정 응답) |
| 1 | `inbase_device` | inbase 디바이스 상태/제어 |
| 2 | `weather_api` | 날씨/미세먼지 |
| 3 | `news_api` | 네이버 뉴스 검색 |
| 4 | `traffic_api` | 교통/대중교통 |
| 5 | `energy_api` | 원격검침/에너지 |
| 6 | `local_info_api` | 의료/유가/증시/단지정보/차량 |

### 4.6 judge_head (5 classes)

| ID | 라벨 | 판단 기준 | 예시 |
|---|---|---|---|
| 0 | `none` | 판단 불필요 | 대부분의 제어/조회 |
| 1 | `outdoor_activity` | 강수, 미세먼지, 풍속 | "세차해도 돼?", "캠핑 괜찮아?", "소풍 가도 돼?" |
| 2 | `clothing` | 기온, 일교차 | "뭐 입고 나가?", "겉옷 필요해?", "아이 옷 어떻게?" |
| 3 | `air_quality` | 미세먼지 등급 | "창문 열어도 돼?", "환기해도 돼?" |
| 4 | `cost_trend` | 가격 추이 | "주유해도 되나?", "전기요금 많이 나올까?" |

### 4.7 multi_action_head (2 classes)

| ID | 라벨 | 설명 | 예시 |
|---|---|---|---|
| 0 | `single` | 단일 액션 | 대부분 |
| 1 | `composite` | 복합 액션 → 매크로 | "나 나갈건데" (소등+엘베+외출) |

---

## 5. 임베딩 모델 선택

### 5.1 후보 모델 비교

| 모델 | 파라미터 | Vocab | 학습 데이터 | KLUE 벤치마크 성능 |
|---|---|---|---|---|
| **KLUE-RoBERTa base** | 111M | 32K | 62GB (모두의 말뭉치, CC-100, 나무위키, 뉴스) | 전반적 최고 (large 기준) |
| KoELECTRA base | 110M | 35K | 34GB | STS, NLI에서 강점 |
| KLUE-BERT base | 111M | 32K | 62GB | YNAT 최고 |
| KoBERT (SKT) | 92M | 8K | 위키피디아 | 구형, 성능 열위 |

### 5.2 추천: KLUE-RoBERTa base

**이유:**

1. **KLUE 벤치마크 전반적 최고 성능**: KLUE-RoBERTa large는 8개 태스크 중 6개에서 1위. base 모델도 KoELECTRA와 호각이며, 일부 태스크(RE, DP, MRC)에서 우세.

2. **형태소 기반 서브워드 토크나이저**: 한국어 교착어 특성을 고려한 morpheme-based subword tokenization 사용. "거실 조명을 껐습니다"를 형태소 단위로 분해하므로 조사/어미 변형에 강건.

3. **임베딩만 추출하여 사용**: 전체 모델을 올리는 게 아니라 `model.embeddings.word_embeddings.weight`만 추출하여 TextConformer의 룩업 테이블로 사용. 임베딩 레이어는 CPU에서 룩업 연산만 하므로 NPU 호환성 문제 없음.

4. **vocab size 32K**: 적절한 크기. 임베딩 테이블 32000 × 256 = 8.2M 파라미터 (FP16으로 약 16MB).

### 5.3 임베딩 추출 및 적용 방법

```python
from transformers import AutoModel

# KLUE-RoBERTa base 로드
roberta = AutoModel.from_pretrained("klue/roberta-base")

# 임베딩 가중치 추출
pretrained_embeddings = roberta.embeddings.word_embeddings.weight.data  # [32000, 768]

# TextConformer의 d_model=256에 맞게 차원 축소
# 방법 1: PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=256)
reduced_embeddings = pca.fit_transform(pretrained_embeddings.numpy())

# 방법 2: Linear projection (학습 가능)
projection = nn.Linear(768, 256, bias=False)
reduced_embeddings = projection(pretrained_embeddings)

# TextConformer에 로드
model.encoder.embedding.weight.data = torch.tensor(reduced_embeddings)
```

**차원 축소 방법 선택:**
- **PCA**: 빠르고 간단. 학습 없이 적용 가능. 분산 보존율 확인 필요.
- **Linear Projection**: 태스크에 맞게 fine-tuning 가능. 약간 더 좋은 성능 기대.
- **추천: Linear Projection** (projection layer를 모델에 포함시켜 함께 학습)

### 5.4 토크나이저 호환성

KLUE-RoBERTa의 토크나이저를 그대로 사용한다. 형태소 기반 서브워드이므로 스마트홈 발화의 조사 변형("불을/불 좀/불이")에 강건하다.

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

# 추가 토큰 등록
special_tokens = ["[NONE]", "[LIGHT_CONTROL]", "[HEAT_CONTROL]", "[AC_CONTROL]", 
                  "[VENT_CONTROL]", "[GAS_CONTROL]", "[DOOR_CONTROL]", 
                  "[CURTAIN_CONTROL]", "[ELEVATOR_CALL]", "[SECURITY_MODE]",
                  "[SCHEDULE_MANAGE]", "[WEATHER_QUERY]", "[NEWS_QUERY]",
                  "[TRAFFIC_QUERY]", "[ENERGY_QUERY]", "[INFO_QUERY]"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
```

---

## 6. 학습 데이터 전략 (핵심)

### 6.1 문제 인식

현재 인텐트 모델의 성능 부족 원인은 **학습 데이터 부족**이 가장 크다. 233개 시나리오만으로는 자연어 변형 커버리지가 심각하게 부족하다.

### 6.2 기존 시도와 실패 원인 분석

#### 6.2.1 시도한 외부 데이터셋

| 데이터셋 | 내용 | 결과 |
|---|---|---|
| **kochat** | 한국어 챗봇 데이터, 일상대화 인텐트 (인사, 욕설, 감사, 날씨 등) | ❌ 스마트홈 인텐트 없음 |
| **amazon_massive_intent_ko-KR** | Amazon MASSIVE의 한국어 인텐트 분류 부분 | ❌ IoT 도메인 소수, 라벨 체계 불일치 |
| **songys/Chatbot_data** | 한국어 감성 대화 데이터 | ❌ 스마트홈과 무관 |

세 데이터셋 모두 **스마트홈 도메인의 인텐트-슬롯 라벨이 없거나 부족**하여 기존 인텐트 분류 모델 학습에 직접 사용할 수 없었다.

#### 6.2.2 "STT는 됐는데 NLU는 왜 안 됐는가?"

같은 팀에서 STT는 AIHub의 일반 한국어 음성 데이터로 학습하여 잘 동작했다. 그런데 NLU는 외부 데이터셋으로 안 됐다. 이상하게 보이지만, 원리적으로 당연한 결과다.

**STT와 NLU의 근본적 차이:**

```
STT: 한국어 소리 → 한국어 글자 (도메인 무관)
     "거실 불 켜줘" → 한국어 음소 패턴 → 글자
     "오늘 주식 어때" → 한국어 음소 패턴 → 글자
     
     → 도메인이 뭐든 음향 모델링은 동일. 
       AIHub 일상대화 음성으로도 "거실 불 켜줘"의 음소를 충분히 학습.

NLU: 한국어 글자 → 의미 구조 (도메인 종속)
     "거실 불 켜줘" → {fn: light_control, room: living}
     "오늘 주식 어때" → {fn: info_query, keyword: 주식}
     
     → 의미 구조가 프로젝트 도메인에 완전히 종속.
       kochat의 "인사"/"감사" 인텐트는 스마트홈 제어와 아무 관련 없음.
```

즉, STT에서 도메인 무관 데이터가 통한 이유는 **출력(글자)이 범용적**이기 때문이고, NLU에서 외부 데이터가 안 통한 이유는 **출력(라벨)이 도메인 특화**이기 때문이다.

#### 6.2.3 근데 외부 데이터가 진짜 쓸모없는 건 아니다

핵심 인사이트: **외부 데이터의 발화 텍스트 자체는 가치가 있다.** 라벨이 안 맞는 거지, 한국어 자연어 패턴을 이해하는 데는 도움이 된다.

```
기존 시도 (실패):
  외부 데이터의 [발화 + 라벨]을 그대로 사용 → 라벨 체계 불일치 → 실패

올바른 접근 (2단계 분리):
  1단계 - 인코더 프리트레인 (도메인 무관)
    → KLUE-RoBERTa 임베딩이 이 역할을 이미 수행
    → 62GB 한국어 데이터로 학습된 임베딩을 가져오므로
      "은은하게", "어두침침한데", "답답하네" 같은 
      간접 표현의 의미 벡터가 이미 양호
    → STT에서 AIHub 데이터가 효과 있었던 것과 동일 원리

  2단계 - Head 학습 (도메인 특화)
    → 233개 시나리오 기반 자체 생성 데이터로 학습
    → 이 부분에서 데이터 부족 문제가 발생
```

#### 6.2.4 Compositional 방식이 데이터 부족 문제를 완화하는 이유

기존 인텐트 방식과 Compositional 방식은 **같은 양의 데이터로도 커버리지가 완전히 다르다.**

```
인텐트 방식 (곱셈적 데이터 요구):
  "거실 에어컨 23도" → AC_TEMP_SET (인텐트 #1)
  "안방 에어컨 25도" → AC_TEMP_SET (같은 인텐트, 다른 슬롯)
  "거실 난방 23도"   → HEAT_TEMP_SET (다른 인텐트)
  "안방 난방 25도"   → HEAT_TEMP_SET (같은 인텐트, 다른 슬롯)
  
  → 디바이스 5개 × 공간 5개 × 액션 5개 = 125개 조합
  → 각 조합당 10개 표현 = 1,250개 데이터 필요
  → 새 디바이스 추가 시 125→150개 조합으로 증가

Compositional 방식 (덧셈적 데이터 요구):
  "거실 에어컨 23도" → fn=ac, room=거실, param=temp, value=23
  "안방 난방 25도"   → fn=heat, room=안방, param=temp, value=25
  
  → fn=ac는 에어컨 데이터에서 학습
  → room=안방은 난방 데이터에서 학습  
  → "안방 에어컨 20도"는 학습 안 해도 각 축이 독립 예측
  
  → fn 15개 × 10표현 + room 8개 × 10표현 + param 8개 × 10표현 = 310개 데이터
  → 이론상 15 × 8 × 8 = 960가지 조합 커버
  → 새 디바이스 추가 시 10개 발화만 추가
```

**이전 실패의 진짜 원인 정리:**

| 원인 | 기존 접근 | Compositional 접근 |
|---|---|---|
| 외부 데이터 라벨 불일치 | 라벨을 그대로 써서 실패 | 임베딩만 활용 (KLUE-RoBERTa) |
| 스마트홈 데이터 부족 | 인텐트 60개 × 100표현 = 6,000개 필요 | head당 50표현이면 조합으로 커버 |
| 새 조합 대응 불가 | 인텐트에 없으면 예측 불가 | 각 축 독립 예측으로 일반화 |
| 간접 표현 대응 | 인텐트별로 간접 표현 학습 필요 | 임베딩이 의미 유사성 처리 |

### 6.3 데이터 소스 전략

#### 6.3.1 핵심 공개 데이터셋: Amazon MASSIVE

**MASSIVE (Multilingual Amazon SLURP for Slot-filling, Intent classification, and Virtual assistant Evaluation)**

- **규모**: 51개 언어, 100만+ 발화, **한국어(ko-KR) 포함**
- **도메인**: 18개 (iot, weather, music, news, general, transport, email, calendar, alarm, audio, social, recommendation, cooking, qa, datetime, play, lists, takeaway)
- **인텐트**: 60개 (iot_hue_lightchange, iot_hue_lightoff, iot_hue_lighton, weather_query, news_query, transport_query 등)
- **슬롯**: 55개 (device_type, room, temperature, time, location 등)
- **라이센스**: Apache 2.0 (상업 사용 가능)
- **출처**: https://github.com/alexa/massive

**스마트홈 관련 인텐트 매핑 (MASSIVE → 프로젝트):**

| MASSIVE 인텐트 | 프로젝트 fn_head 매핑 |
|---|---|
| iot_hue_lightchange | light_control |
| iot_hue_lightoff | light_control |
| iot_hue_lighton | light_control |
| iot_hue_lightdim | light_control |
| iot_hue_lightup | light_control |
| iot_cleaning | vent_control (유사) |
| weather_query | weather_query |
| news_query | news_query |
| transport_query | traffic_query |
| general_quirky | direct_respond |
| qa_factoid | info_query |
| datetime_query | direct_respond |
| alarm_set | schedule_manage |

MASSIVE의 한국어 데이터에서 IoT/날씨/뉴스/교통 도메인만 추출하면 약 **5,000~8,000개의 자연스러운 한국어 발화**를 확보할 수 있다. 이를 프로젝트의 Action Struct 포맷으로 재라벨링한다.

#### 6.3.2 AIHub 활용 가능 데이터

| 데이터셋 | 활용 방안 |
|---|---|
| 차량 내 대화 및 명령어 음성 (aihub.or.kr, SN:112) | 명령형 발화 패턴 참고, 날씨/내비/음악 제어 도메인 |
| 한국어 대화 데이터셋 (SN:272) | 오피스/응급상황 멀티턴 대화 구조 참고 |
| 주제별 텍스트 일상 대화 (SN:543) | 주거/생활, 교통, 날씨 주제 발화 패턴 |
| 한국어 대화 (SN:85) | 소상공인/공공민원 의도-엔티티 구조 참고 |

#### 6.3.3 LLM 기반 데이터 증강 (가장 중요)

엑셀 233개 시나리오를 seed로, LLM(Claude/GPT)을 사용하여 대규모 패러프레이즈 생성.

**프롬프트 예시:**
```
다음 스마트홈 음성 명령의 다양한 한국어 표현을 20개 생성해주세요.
표현은 구어체, 반말, 존댓말, 간접 표현, 감정 표현을 포함해야 합니다.

원본: "거실 에어컨 23도로 맞춰줘"
의미: {fn: ac_control, room: living, param: temperature, value: 23}

생성 결과:
1. 거실 냉방 23도로 해줘
2. 거실 좀 시원하게 23도
3. 에어컨 거실 23도 부탁
4. 거실이 더운데 23도로 좀 낮춰
5. 거실 에어컨 온도 23도로 설정해
6. 에어컨 23도로 틀어줘 거실
7. 거실 23도면 되겠다
8. 거실 시원하게 좀 23도
9. 거실 냉방 23도 ㄱㄱ (비표준이지만 STT 오류 대비)
10. 거실 에어컨 이십삼도로 맞춰 (STT 숫자-한글 변환 대비)
...
```

**간접 표현 생성 (중요):**
```
원본: "환기 켜줘"
간접 표현:
- "공기가 답답하네"
- "좀 환기 시키면 좋겠다"
- "공기가 안 좋은데"
- "냄새 좀 빼줘"
- "바람 좀 통하게 해줘"
```

#### 6.3.4 조합 증강 (Compositional Augmentation)

각 축의 값을 교차 조합하여 자동 생성:

```python
rooms = ["거실", "주방", "안방", "각실", "전체"]
devices = ["조명", "에어컨", "난방", "환기"]
actions = ["켜줘", "꺼줘", "올려줘", "내려줘", "상태"]

for room in rooms:
    for device in devices:
        for action in actions:
            utterance = f"{room} {device} {action}"
            labels = {
                "fn": device_to_fn[device],
                "room": room_to_id[room],
                "exec_type": action_to_exec[action],
                ...
            }
```

#### 6.3.5 Negative / Out-of-Domain 데이터

모델이 "처리할 수 없는 요청"을 인식하도록 학습 데이터에 포함:

```json
{"utterance": "오늘 로또 번호 뭐야?", "labels": {"exec_type": "direct_respond", "fn": "info_query"}}
{"utterance": "피자 주문해줘", "labels": {"exec_type": "direct_respond", "fn": "info_query"}}
{"utterance": "노래 틀어줘", "labels": {"exec_type": "direct_respond", "fn": "info_query"}}
```

### 6.4 최종 데이터 구성 목표

| 소스 | 예상 수량 | 용도 |
|---|---|---|
| 엑셀 시나리오 233개 | 233 | seed, 정확한 라벨 |
| MASSIVE 한국어 (IoT/날씨/뉴스/교통) | 5,000~8,000 | 자연스러운 발화, 재라벨링 필요 |
| LLM 패러프레이즈 (시나리오당 20개) | ~4,600 | 표현 다양성 |
| LLM 간접 표현 생성 | ~1,000 | "답답하네" → 환기, "어두침침한데" → 조명 |
| 조합 증강 (room × fn × param) | ~3,000 | 조합 일반화 |
| 멀티턴 페어 | ~1,000 | 컨텍스트 학습 |
| Negative/OOD | ~500 | 미지원 요청 대응 |
| **합계** | **~15,000~18,000** | |

### 6.5 데이터 포맷

```json
{
  "id": "train_00001",
  "utterance": "안방 에어컨 23도로 맞춰줘",
  "context_token": "NONE",
  "tokens": ["안방", "에어컨", "23", "도", "로", "맞춰줘"],
  "labels": {
    "exec_type": 1,
    "fn": 2,
    "room": 2,
    "param_type": 1,
    "has_value": 1,
    "value_start": 2,
    "value_end": 3,
    "api": 1,
    "judge": 0,
    "multi_action": 0
  },
  "source": "scenario_R39",
  "augmentation": "paraphrase"
}
```

### 6.6 데이터 품질 관리

1. **LLM 생성 데이터 검수**: 생성된 패러프레이즈를 사람이 검수하여 부자연스러운 표현 제거
2. **STT 오류 시뮬레이션**: 실제 STT 출력에서 자주 발생하는 오류 패턴을 학습 데이터에 반영
   - "이십삼도" ↔ "23도"
   - "에어컨" ↔ "에어콘"
   - "거실" ↔ "거시" (인식 오류)
3. **클래스 밸런싱**: exec_type, fn 등 각 head별 클래스 분포를 확인하고, 부족한 클래스에 대해 추가 증강

---

## 7. Loss 함수 및 학습 전략

### 7.1 Multi-Task Loss

```python
def compute_loss(predictions, labels):
    loss = 0
    weights = {
        "exec_type": 2.0,      # 실행 분기 → 가장 중요
        "fn": 2.0,             # 함수 선택 → 가장 중요
        "room": 1.5,
        "param_type": 1.0,
        "api": 1.0,
        "judge": 1.5,          # 판단 유형 정확도 중요
        "multi_action": 1.0,
        "has_value": 1.0,
        "value_start": 1.5,
        "value_end": 1.5,
    }
    
    for head_name, weight in weights.items():
        if head_name in ["value_start", "value_end"]:
            mask = (labels["has_value"] == 1)
            if mask.any():
                loss += weight * F.cross_entropy(
                    predictions[head_name][mask],
                    labels[head_name][mask]
                )
        else:
            loss += weight * F.cross_entropy(
                predictions[head_name], labels[head_name]
            )
    return loss
```

### 7.2 학습 하이퍼파라미터

| 항목 | 값 | 비고 |
|---|---|---|
| Optimizer | AdamW | |
| Learning Rate | 5e-4 (encoder), 1e-3 (heads) | Differential LR |
| Warmup | 500 steps | |
| Scheduler | Cosine with warmup | |
| Batch Size | 32 | |
| Epochs | 30 | Early stopping (patience=5) |
| Dropout | 0.1 | |
| 임베딩 Freeze | 처음 10 epoch freeze, 이후 unfreeze | |

---

## 8. Action Executor 설계

### 8.1 실행 분기 (Python pseudocode)

```python
class ActionExecutor:
    def execute(self, action_struct):
        t = action_struct["exec_type"]
        
        if t == "control_then_confirm":
            result = self.device_api.call(action_struct)
            return self.templates.control_response(action_struct, result)
        
        elif t == "query_then_respond":
            data = self.call_api(action_struct)
            return self.templates.query_response(action_struct, data)
        
        elif t == "query_then_judge":
            data = self.call_api(action_struct)
            judgment = self.judge_engine.evaluate(action_struct["judge"], data)
            return self.templates.judge_response(action_struct, judgment, data)
        
        elif t == "direct_respond":
            return self.templates.static_response(action_struct["fn"])
        
        elif t == "clarify":
            return self.templates.clarify_response(action_struct)
```

### 8.2 판단 엔진 (Judge Engine)

```python
JUDGE_RULES = {
    "outdoor_activity": {
        "conditions": [
            {"field": "precip_prob", "op": ">", "value": 40, "reason": "강수확률 {v}%"},
            {"field": "pm10_grade", "op": ">=", "value": 3, "reason": "미세먼지 나쁨"},
            {"field": "wind_speed", "op": ">", "value": 14, "reason": "강풍 예보"},
        ],
        "ok_response": "기온과 대기질 모두 양호합니다. 외출하기 무리 없습니다.",
        "fail_template": "{reason}이므로 야외활동은 권장하지 않습니다."
    },
    "clothing": {
        "ranges": [
            {"min": 28, "max": 99, "clothing": "반팔/반바지"},
            {"min": 20, "max": 27, "clothing": "얇은 긴팔"},
            {"min": 12, "max": 19, "clothing": "가벼운 겉옷"},
            {"min": -99, "max": 11, "clothing": "두꺼운 외투"},
        ],
        "gap_threshold": 10,
        "gap_note": "일교차가 크니 겉옷을 함께 준비하세요."
    },
    "air_quality": {
        "conditions": [
            {"field": "pm25_grade", "op": ">=", "value": 3, "reason": "초미세먼지 나쁨"},
            {"field": "pm10_grade", "op": ">=", "value": 3, "reason": "미세먼지 나쁨"},
        ],
        "ok_response": "미세먼지 양호 수준으로 창문을 열어 환기하기 적절합니다.",
        "fail_template": "{reason} 수준으로 창문 개방은 권장하지 않습니다."
    },
    "cost_trend": {
        "rising": "상승 추세입니다. 참고하시기 바랍니다.",
        "stable": "안정적인 수준입니다.",
        "falling": "하락 추세입니다."
    }
}
```

### 8.3 복합 액션 매크로

```python
COMPOSITE_MACROS = {
    "security_mode:away": {
        "name": "외출 모드",
        "actions": [
            {"fn": "light_control", "room": "all", "action": "off"},
            {"fn": "vent_control", "action": "off"},
            {"fn": "elevator_call"},
            {"fn": "security_mode", "mode": "away"},
        ],
        "response": "외출 감지 {delay}초 후 일괄 소등, 환기 끄기, 엘리베이터 호출을 실행합니다."
    },
    "schedule_manage:morning": {
        "name": "모닝콜 연동",
        "actions": [
            {"fn": "light_control", "room": "bedroom_main", "action": "on"},
            {"fn": "vent_control", "action": "on"},
            {"fn": "curtain_control", "room": "all", "action": "open"},
        ],
        "response": "모닝콜이 설정되었습니다. 조명, 환기, 전동커튼이 함께 작동합니다."
    }
}
```

---

## 9. NPU 배포 (T527)

### 9.1 변환 파이프라인

```
PyTorch 학습 완료
    ↓
ONNX export (opset 13)
    ↓
Pegasus toolchain
    ↓
INT8/UINT8 양자화 (Calibration: 학습 데이터 10%)
    ↓
NB 파일 생성
    ↓
T527 NPU 배포
```

### 9.2 양자화 전략

| 레이어 | 양자화 | 이유 |
|---|---|---|
| 토큰 임베딩 | FP32 (CPU) | 룩업 테이블, NPU 미사용 |
| 컨텍스트 임베딩 | FP32 (CPU) | 룩업 테이블 |
| Conformer Conv 모듈 | INT8 (NPU) | Conv op은 INT8에서 정밀도 유지 |
| Conformer FF 모듈 | INT8 (NPU) | Linear op |
| Conformer Attention | INT8 (NPU) | ASR에서 검증됨 |
| Classification Heads | INT8 (NPU) | 클래스 수 적어 정밀도 이슈 낮음 |
| Value Pointer Head | FP16 (혼합정밀도) | 위치 정밀도 필요 |

### 9.3 NPU 호환성 체크리스트

| Op | 사용 여부 | NPU 호환 | 비고 |
|---|---|---|---|
| Embedding (lookup) | ✅ | CPU 처리 | |
| Conv1D (depthwise) | ✅ | ✅ | Conformer conv |
| Conv1D (pointwise) | ✅ | ✅ | |
| Linear (FC) | ✅ | ✅ | FF + heads |
| ReLU / Swish | ✅ | ✅ | Swish→ReLU 대체 검토 |
| LayerNorm | ✅ | ⚠️ | ASR에서 변환 성공 |
| Multi-Head Attention | ✅ | ⚠️ | ASR에서 변환 성공 |
| Softmax | ✅ | ⚠️ | ASR에서 변환 성공 |
| Sigmoid | ❌ | - | 사용 안 함 (SE block 제외) |

---

## 10. 평가 지표

### 10.1 Head별 평가

| Head | 지표 | 목표 |
|---|---|---|
| exec_type | Accuracy | ≥ 95% |
| fn | Accuracy | ≥ 93% |
| room | Accuracy | ≥ 95% |
| param_type | Accuracy | ≥ 93% |
| api | Accuracy | ≥ 95% |
| judge | Accuracy | ≥ 90% |
| multi_action | Accuracy | ≥ 95% |
| value_pointer | Position Accuracy | ≥ 90% |

### 10.2 End-to-End 평가

| 지표 | 설명 | 목표 |
|---|---|---|
| Exact Match | 모든 head 동시 정답 | ≥ 85% |
| Execution Accuracy | 올바른 함수 호출 + 올바른 응답 | ≥ 90% |
| Compositional Accuracy | 학습에 없던 조합 정확도 | ≥ 80% |
| Latency (NPU) | 텍스트 입력→구조체 출력 | ≤ 50ms |

### 10.3 테스트셋 구성

- **In-domain** (40%): 학습 데이터와 동일 조합, 다른 표현
- **Compositional** (30%): 학습에 없던 fn×room×param 조합
- **Adversarial** (15%): 모호한 발화, 예외 케이스, STT 오류 발화
- **Multi-turn** (15%): 컨텍스트 의존 발화 쌍

---

## 11. 놓치기 쉬운 추가 고려사항

### 11.1 STT 오류 전파

STT에서 오인식된 텍스트가 Parser에 입력되므로, STT 오류 패턴을 학습 데이터에 반영해야 한다.

- 숫자 표기: "이십삼도" / "23도" / "스물세도" 모두 처리
- 유사 발음: "에어컨" / "에어콘" / "에어컨을" (조사 결합)
- 띄어쓰기 변형: "거실불꺼줘" / "거실 불 꺼줘"
- 노이즈 삽입: "어... 거실 불 좀 꺼줘"

### 11.2 시간 표현 파싱

"오전 10시부터 오후 4시까지", "내일", "매일 아침 6시", "30초 후" 등 시간 표현이 다양하다. Value Pointer만으로는 시간 범위(start~end)를 표현하기 어려우므로:

- **time_start_pointer**와 **time_end_pointer**를 추가하거나
- 시간 표현 전체를 추출한 후 별도 파서(규칙 기반)로 구조화하는 방식 검토

### 11.3 차량 번호 / 고유명사 추출

"0000000 차량 등록해줘", "삼성전자 주가" 등 고유명사/번호는 Value Pointer로 범위를 추출한 후, 원본 텍스트에서 해당 토큰들을 연결하여 문자열로 복원.

### 11.4 예약/스케줄의 복합 구조

"매일 밤 10시 취침모드로 설정" → fn=schedule_manage이지만, 내부적으로 light_control + time 파라미터가 필요. 이 경우 schedule_manage가 내부에 또 다른 Action Struct를 포함하는 **중첩 구조**가 필요할 수 있다. 일단은 schedule_manage로 분류 후, 룰 엔진에서 연관 파라미터를 추출하는 방식으로 처리.

### 11.5 에러 핸들링

- API 호출 실패 시 fallback 응답
- NPU 추론 timeout 시 기본 응답
- 모든 head의 confidence가 낮은 경우 "다시 말씀해주세요" 응답
- `clarify` 후 재발화에서도 정보 부족 시 기본 안내 ("월패드 화면에서 직접 설정해주세요")

### 11.6 TTS 응답 자연스러움

템플릿 기반 응답이 기계적으로 느껴지지 않도록:
- 동일 의미의 응답 템플릿을 2~3개씩 준비하여 랜덤 선택
- 시간대별 인사 ("좋은 아침이에요", "안녕하세요")
- 확인 응답 변형 ("네", "알겠습니다", "처리했습니다")

### 11.7 보안 관련 발화

비밀번호, 가스 밸브 열기 등 보안 민감 발화는 모델 레벨이 아닌 **Action Executor 레벨에서 차단**:

```python
BLOCKED_ACTIONS = [
    {"fn": "gas_control", "action": "open"},     # 가스 열기 차단
    {"fn": "door_control", "condition": "외부"},   # 외부에서 도어락 열기 제한
]
```

### 11.8 모델 업데이트

새로운 디바이스나 시나리오 추가 시:
- fn_head에 클래스 추가 → head만 재학습 (encoder freeze)
- 새 디바이스에 대한 발화 데이터 추가
- **증분 학습(incremental learning)** 가능하도록 모델 구조 설계

---

## 12. 개발 로드맵

### Phase 1: 데이터 준비 (3주)

- [ ] 엑셀 233개 → Action Struct JSON 변환
- [ ] MASSIVE 한국어 데이터 다운로드 + IoT/날씨/뉴스/교통 필터링 + 재라벨링
- [ ] LLM 패러프레이즈 생성 (시나리오당 20개)
- [ ] LLM 간접 표현 생성
- [ ] 조합 증강 스크립트
- [ ] 멀티턴 페어 데이터 생성
- [ ] STT 오류 시뮬레이션 데이터
- [ ] Negative/OOD 데이터
- [ ] 데이터 검수 및 클래스 밸런싱
- [ ] train/val/test 분할 (7:1.5:1.5)

### Phase 2: 모델 구현 및 학습 (2주)

- [ ] KLUE-RoBERTa 임베딩 추출 + 차원 축소
- [ ] TextConformer 인코더 구현 (ASR Conformer 코드 기반)
- [ ] Multi-Head Classifier + Value Pointer 구현
- [ ] 토크나이저 설정 + 스페셜 토큰 추가
- [ ] Loss 함수 + 학습 파이프라인
- [ ] 학습 + 하이퍼파라미터 튜닝
- [ ] 컴포지셔널 테스트셋으로 일반화 성능 검증

### Phase 3: NPU 변환 및 검증 (1주)

- [ ] ONNX export
- [ ] Pegasus NB 변환
- [ ] INT8 양자화 + calibration
- [ ] NPU vs FP32 정확도 비교
- [ ] 레이턴시 측정

### Phase 4: Action Executor + 템플릿 (2주)

- [ ] 실행 분기 로직
- [ ] 템플릿 엔진 (233개 시나리오 전체 커버)
- [ ] 판단 엔진 (Judge) 구현 + 룰 테이블
- [ ] 복합 액션 매크로 정의
- [ ] inbase API 연동
- [ ] 에러 핸들링

### Phase 5: 통합 테스트 (1주)

- [ ] STT → Parser → Executor → TTS 전체 파이프라인
- [ ] 233개 시나리오 전수 테스트
- [ ] End-to-End 정확도 측정
- [ ] 멀티턴 시나리오 테스트
- [ ] 엣지 케이스/보안 테스트

---

## 13. 참고 문헌

- **MTOP**: Li et al., "MTOP: A Comprehensive Benchmark for Multi-domain Task-Oriented Semantic Parsing" (Facebook, 2020)
- **MASSIVE**: FitzGerald et al., "MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages" (Amazon, ACL 2023)
- **Conformer**: Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (Google, 2020)
- **KLUE**: Park et al., "KLUE: Korean Language Understanding Evaluation" (NeurIPS 2021)
- **JointBERT**: Chen et al., "BERT for Joint Intent Classification and Slot Filling" (2019)
- **Compositional Generalization**: Keysers et al., "Measuring Compositional Generalization: A Comprehensive Method on Realistic Data" (Google, ICLR 2020)

---

## 부록 A: 엑셀 시나리오 → Action Struct 변환 예시 (확장)

```json
[
  {
    "id": "R2", "category": "메인/상태/조회",
    "utterance": "지금 집 상태 어때?",
    "labels": {"exec_type": "query_then_respond", "fn": "info_query", "room": "all", "param_type": "none", "api": "inbase_device", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R5", "category": "메인/시간/조회",
    "utterance": "지금 몇 시야?",
    "labels": {"exec_type": "direct_respond", "fn": "info_query", "room": "none", "param_type": "none", "api": "none", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R16", "category": "제어/조명/전원",
    "utterance": "불 좀 켜줘",
    "labels": {"exec_type": "clarify", "fn": "light_control", "room": "ambiguous", "param_type": "none", "api": "inbase_device", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R17", "category": "제어/조명/조절",
    "utterance": "거실 조명 은은하게 해줘",
    "labels": {"exec_type": "control_then_confirm", "fn": "light_control", "room": "living", "param_type": "brightness", "value": 1, "api": "inbase_device", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R21", "category": "제어/조명/조절",
    "utterance": "거실이 좀 어두침침한데?",
    "labels": {"exec_type": "control_then_confirm", "fn": "light_control", "room": "living", "param_type": "brightness", "value": "max", "api": "inbase_device", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R39", "category": "제어/에어컨/조절",
    "utterance": "거실 에어컨 23도로 맞춰줘",
    "labels": {"exec_type": "control_then_confirm", "fn": "ac_control", "room": "living", "param_type": "temperature", "value": 23, "api": "inbase_device", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R48", "category": "제어/에어컨/예외",
    "utterance": "전체 다 켜",
    "labels": {"exec_type": "clarify", "fn": "info_query", "room": "all", "param_type": "none", "api": "none", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R56", "category": "제어/환기/전원",
    "utterance": "공기가 답답하네",
    "labels": {"exec_type": "control_then_confirm", "fn": "vent_control", "room": "none", "param_type": "none", "api": "inbase_device", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R86", "category": "방범/외출/예외",
    "utterance": "나 지금 나갈건데",
    "labels": {"exec_type": "control_then_confirm", "fn": "security_mode", "room": "all", "param_type": "mode", "value": "away", "api": "inbase_device", "judge": "none", "multi_action": "composite"}
  },
  {
    "id": "R129", "category": "설정/비밀번호/조회",
    "utterance": "현재 월패드 비밀번호 뭐야?",
    "labels": {"exec_type": "direct_respond", "fn": "info_query", "room": "none", "param_type": "none", "api": "none", "judge": "none", "multi_action": "single"}
  },
  {
    "id": "R161", "category": "부가/날씨/조회",
    "utterance": "오늘 한강 바람쐬기 괜찮나?",
    "labels": {"exec_type": "query_then_judge", "fn": "weather_query", "room": "external", "param_type": "none", "api": "weather_api", "judge": "outdoor_activity", "multi_action": "single"}
  },
  {
    "id": "R170", "category": "부가/미세먼지/조회",
    "utterance": "창문 열어도 괜찮아?",
    "labels": {"exec_type": "query_then_judge", "fn": "weather_query", "room": "external", "param_type": "none", "api": "weather_api", "judge": "air_quality", "multi_action": "single"}
  },
  {
    "id": "R173", "category": "부가/날씨/조회",
    "utterance": "오늘 뭐 입고 나가야 돼?",
    "labels": {"exec_type": "query_then_judge", "fn": "weather_query", "room": "external", "param_type": "none", "api": "weather_api", "judge": "clothing", "multi_action": "single"}
  },
  {
    "id": "R214", "category": "부가/유가/조회",
    "utterance": "오늘 주유해도 되나?",
    "labels": {"exec_type": "query_then_judge", "fn": "info_query", "room": "none", "param_type": "none", "api": "local_info_api", "judge": "cost_trend", "multi_action": "single"}
  }
]
```

---

## 부록 B: 인텐트 방식 vs Compositional 방식 비교

| 항목 | Intent-Slot NLU | Semantic Action Parser |
|---|---|---|
| 인텐트 수 | 60~80개 (시나리오 추가 시 증가) | 해당 없음 |
| 분류 클래스 합계 | 60~80 | 43 (5+15+8+8+7+5+2) |
| 새 디바이스 추가 | 인텐트 N개 + 재학습 | fn_head에 1개 추가 |
| 안 본 조합 대응 | 불가 | 각 축 독립 예측으로 가능 |
| 멀티 액션 | 별도 인텐트 필요 | multi_action head + 매크로 |
| 판단형 발화 | 인텐트마다 별도 정의 | judge_head로 통합 |
| 응답 생성 | 인텐트별 1:1 매핑 | 구조체→템플릿 조합 |
| 확장성 | 낮음 | 높음 |
| NPU 연산량 | 비슷 | 비슷 (head가 Linear) |

---

## 부록 C: MASSIVE 데이터셋 활용 가이드

### 다운로드

```bash
git clone https://github.com/alexa/massive.git
cd massive
# ko-KR 데이터 위치: data/ko-KR/
```

### 한국어 IoT 도메인 필터링

```python
import json

with open("data/ko-KR/train.jsonl") as f:
    data = [json.loads(line) for line in f]

# IoT + 날씨 + 뉴스 + 교통 도메인 필터
target_domains = ["iot", "weather", "news", "transport", "general", "qa", "datetime"]
filtered = [d for d in data if d["scenario"].split("_")[0] in target_domains]

print(f"전체: {len(data)}, 필터링 후: {len(filtered)}")
```

### 재라벨링 예시

MASSIVE의 인텐트/슬롯 → 프로젝트 Action Struct로 변환:

```python
INTENT_MAP = {
    "iot_hue_lighton": {"fn": "light_control", "exec_type": "control_then_confirm"},
    "iot_hue_lightoff": {"fn": "light_control", "exec_type": "control_then_confirm"},
    "iot_hue_lightchange": {"fn": "light_control", "exec_type": "control_then_confirm"},
    "weather_query": {"fn": "weather_query", "exec_type": "query_then_respond"},
    "news_query": {"fn": "news_query", "exec_type": "query_then_respond"},
    "transport_query": {"fn": "traffic_query", "exec_type": "query_then_respond"},
    # ...
}

SLOT_MAP = {
    "device_type": "fn",        # -> fn_head 참고
    "house_place": "room",     # -> room_head
    "temperature": "value",    # -> value_pointer
    "weather_descriptor": "param_type",
    # ...
}
```
