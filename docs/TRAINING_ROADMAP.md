# TextConformer NLU 학습 로드맵

## 목표

T527 NPU에서 돌아가는 한국어 의도 분류 모델.
Conformer(CNN+Attention) 구조로 양자화 에러 보정.
간접 표현("너무추워"→heating_on) 포함 처리.

---

## 1. 모델 아키텍처: TextConformer-18L

Conformer STT(18L, 122.5M, 102MB NB, 250ms)와 동일 규모.

```
TextConformer Block (× 18):
  FFN(½) → MultiheadAttention → DepthwiseConv1D → FFN(½)
  ↑ Macaron 구조: Conv가 Attention 양자화 에러 보정
```

| 항목 | 값 |
|------|-----|
| Layer | 18 |
| d_model | 512 |
| nhead | 8 |
| conv_kernel | 15 |
| ff_dim | 2048 |
| vocab_size | 32000 (BERT tokenizer) |
| max_len | 64 |
| num_intents | 51 |
| 예상 params | ~80-120M |
| 예상 NB | 50-100MB |
| 예상 NPU 추론 | 30-100ms |

### 참고: Conformer STT vs TextConformer

| | Conformer STT | TextConformer |
|---|---|---|
| 입력 | mel [1,80,301] float | token_ids [1,64] int |
| 출력 | logits [1,76,2049] | intent [1,51] |
| Layer | 18 | 18 |
| 용도 | 음성 → 텍스트 | 텍스트 → 의도 |
| NPU | 250ms, 102MB NB | 예상 30-100ms |

---

## 2. 학습 데이터

### 2-1. 공개 한국어 데이터

| 데이터 | 소스 | 크기 | 용도 |
|--------|------|------|------|
| [songys/Chatbot_data](https://github.com/songys/Chatbot_data) | GitHub | ~11k | 한국어 챗봇 Q&A |
| [KorNLI/KorSTS](https://github.com/kakaobrain/kor-nlu-datasets) | KakaoBrain | ~900k | 한국어 NLU 벤치마크 |
| [kochat](https://github.com/hyunwoongko/kochat) | GitHub | intent+slot | 한국어 챗봇 프레임워크 |
| AIHub 한국어 대화 | aihub.or.kr | 다양 | 음성/텍스트 대화 |
| [3i4k](https://github.com/warnikchow/3i4k) | GitHub | ~60k | 한국어 의도 분류 (7 class) |

### 2-2. 스마트홈 도메인 데이터 (자체 생성)

엑셀 219개 시나리오 + 간접 표현 변형:

```
intent: heating_on
직접: "난방 켜줘", "보일러 켜줘", "난방 틀어줘", "전체 난방 켜줘"
간접: "너무 추워", "춥다", "얼어 죽겠다", "쌀쌀해", "집이 너무 춥다"
변형: "거실 난방 좀 켜줄래", "보일러 좀 올려줘", "방이 추운데"
```

- 51개 intent × 각 50-100개 발화문 = **2,500 ~ 5,100개**
- GPT로 변형 생성하면 빠르게 확보 가능

### 2-3. 사전 학습 데이터 (일반 한국어)

TextConformer를 처음부터 학습하려면 대량 한국어 텍스트 필요:
- [나무위키 말뭉치](https://github.com/lovit/namuwikitext) ~5GB
- [AIHub 한국어 말뭉치](https://aihub.or.kr) 
- [모두의 말뭉치](https://corpus.korean.go.kr/) — 국립국어원

**방법 2가지:**
1. **처음부터 학습 (pre-train + fine-tune)** — 시간 많이 필요 (수일~수주)
2. **기존 한국어 모델 weight 활용 + Conformer 구조로 변환** — 빠르지만 복잡

---

## 3. 학습 파이프라인

### 3-1. 환경

```bash
# NeMo Docker (PyTorch + CUDA 포함)
docker run --gpus all -it \
  -v /home/nsbb/travail/claude/T527/t527-nlu:/workspace/nlu \
  nvcr.io/nvidia/nemo:23.06 bash
```

### 3-2. Phase 1: 도메인 데이터로 직접 학습 (빠른 검증)

사전 학습 건너뛰고, 스마트홈 intent 데이터만으로 학습.
모델이 작으면 (tiny/small) 이것만으로도 가능.

```python
# 학습 스크립트 (PyTorch)
model = TextConformer(vocab_size=32000, d_model=512, num_classes=51, num_layers=18)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in dataloader:
        logits = model(batch["input_ids"])
        loss = criterion(logits, batch["intent"])
        loss.backward()
        optimizer.step()
```

### 3-3. Phase 2: 사전 학습 + Fine-tune (성능 최적화)

1. 대량 한국어 텍스트로 MLM(Masked Language Model) 사전 학습
2. 스마트홈 intent 데이터로 fine-tune

---

## 4. 양자화 + NB 변환

```bash
# ONNX export
torch.onnx.export(model, dummy, "textconformer_18l.onnx")

# Acuity pipeline (Docker)
pegasus import onnx --model textconformer_18l.onnx
pegasus quantize --qtype uint8 --algorithm kl_divergence
pegasus export ovxlib --optimize VIP9000NANOSI_PLUS_PID0X10000016
```

### NPU 검증 포인트

| 체크 | 방법 |
|------|------|
| NB 변환 성공 | `pegasus export` Error 0 |
| NPU 추론 성공 | `vpm_run ret=0` |
| **양자화 정확도** | **NPU 출력 vs CPU 출력 비교** |
| intent 정확도 | 테스트셋 F1 score |

---

## 5. 일정 (예상)

| 단계 | 소요 | 산출물 |
|------|------|--------|
| 학습 데이터 생성 (51 intent × 50개) | 1일 | intent_train.json |
| TextConformer-small (2L) 학습 + NB 변환 | 1일 | 빠른 검증 |
| TextConformer-18L 학습 | 2-3일 | 본 모델 |
| NB 변환 + NPU 정확도 검증 | 1일 | network_binary.nb |
| 앱 통합 | 2일 | t527_vad_service + NLU |

**총 예상: 1-2주**

---

## 6. 대안 (TextConformer 학습 실패 시)

1. **CPU ONNX ko-sbert-sts** — 이미 95.7% 정확도, 12ms, 학습 불필요
2. **TextCNN (NPU)** — 간접 표현 약하지만 학습 데이터로 보완
3. **KoELECTRA (CPU)** — 기존 방식, 간접 표현 학습 데이터 필요

---

## 참고 자료

- [Conformer 논문](https://arxiv.org/abs/2005.08100)
- [songys/Chatbot_data](https://github.com/songys/Chatbot_data)
- [kakaobrain/kor-nlu-datasets](https://github.com/kakaobrain/kor-nlu-datasets)
- [kochat](https://github.com/hyunwoongko/kochat)
- [3i4k 의도 분류](https://github.com/warnikchow/3i4k)
