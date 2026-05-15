#!/usr/bin/env python3
"""v72 단독 ONNX export — cnn_multihead_v72.pt → nlu_v72_generalization.onnx.

목적: ensemble에서 v72만 빼내기. 그 후 extract_cnn_body.py로 cnn_body 추출 → Acuity NB.
"""
import sys, os
sys.path.insert(0, 'scripts')
import torch
from model_cnn_multihead import CNNMultiHead, HEAD_CLASSES, HEAD_NAMES

CHK = 'checkpoints'
PT_PATH = f'{CHK}/cnn_multihead_v72.pt'
ONNX_PATH = f'{CHK}/nlu_v72_generalization.onnx'

# token_emb weight 추출 (v28과 동일 ko-sbert이지만 안전하게 v72 자체에서 추출)
import torch.nn as nn

# pt 파일 로드 — 우리 ckpt는 {epoch, state, combo, ke_fn, balanced, desc}
ckpt = torch.load(PT_PATH, map_location='cpu', weights_only=False)
print(f'ckpt epoch={ckpt["epoch"]}, combo={ckpt["combo"]:.2f}, ke_fn={ckpt["ke_fn"]:.2f}, desc={ckpt.get("desc","")}')
state = ckpt['state']

# token_emb.weight 추출 (32000 × 768)
emb_w = state['token_emb.weight']
print(f'emb weight: {emb_w.shape}')

# CNNMultiHead 모델 생성
model = CNNMultiHead(pretrained_emb_weights=emb_w)

# state_dict load
missing, unexpected = model.load_state_dict(state, strict=False)
print(f'missing: {len(missing)}, unexpected: {len(unexpected)}')
if missing[:3]: print('  missing sample:', missing[:3])
if unexpected[:3]: print('  unexpected sample:', unexpected[:3])

model.eval()

# 더미 입력으로 ONNX export
dummy = torch.randint(0, 32000, (1, 32), dtype=torch.long)
torch.onnx.export(
    model,
    (dummy,),
    ONNX_PATH,
    input_names=['token_ids'],
    output_names=['fn', 'exec_type', 'param_direction', 'param_type', 'judge'],
    dynamic_axes={
        'token_ids': {0: 'batch'},
        'fn': {0: 'batch'}, 'exec_type': {0: 'batch'},
        'param_direction': {0: 'batch'}, 'param_type': {0: 'batch'},
        'judge': {0: 'batch'},
    },
    opset_version=14,
    do_constant_folding=True,
)
print(f'\nsaved {ONNX_PATH}')

# 검증
import onnx
m = onnx.load(ONNX_PATH)
print(f'ops: {len(m.graph.node)}')
for inp in m.graph.input:
    print(f'  IN  {inp.name}: {[d.dim_value if d.dim_value>0 else d.dim_param for d in inp.type.tensor_type.shape.dim]}')
for out in m.graph.output:
    print(f'  OUT {out.name}: {[d.dim_value if d.dim_value>0 else d.dim_param for d in out.type.tensor_type.shape.dim]}')
