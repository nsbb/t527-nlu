#!/usr/bin/env python3
"""Export v28+v46 Ensemble as single ONNX model
Strategy B: fn from v46, exec/dir from v28, judge from v46, param_type from v28

Produces one deployable ONNX file with ensemble logic built-in.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, 'scripts')

from model_cnn_multihead import *
from transformers import AutoModel, AutoTokenizer


class EnsembleModel(nn.Module):
    """Combined v28+v46 model that produces ensemble predictions in one forward pass"""
    def __init__(self, pw):
        super().__init__()
        self.m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)
        self.m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)

    def forward(self, input_ids):
        l28 = self.m28(input_ids)
        l46 = self.m46(input_ids)

        # Strategy B ensemble:
        # fn = v46 (better generalization)
        # exec, dir, param = v28 (better accuracy on known patterns)
        # judge = v46 (better with external data)
        return {
            'fn': l46['fn'],
            'exec_type': l28['exec_type'],
            'param_direction': l28['param_direction'],
            'param_type': l28['param_type'],
            'judge': l46['judge'],
        }


class EnsembleTupleModel(nn.Module):
    """Same as EnsembleModel but returns tuple for ONNX export"""
    def __init__(self, pw):
        super().__init__()
        self.m28 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)
        self.m46 = CNNMultiHead(pw, d_model=256, max_len=32, dropout=0.0)

    def forward(self, input_ids):
        l28 = self.m28(input_ids)
        l46 = self.m46(input_ids)
        # Return in order: fn, exec_type, param_direction, param_type, judge
        return (l46['fn'], l28['exec_type'], l28['param_direction'],
                l28['param_type'], l46['judge'])


def main():
    sbert = AutoModel.from_pretrained('jhgan/ko-sbert-sts')
    pw = sbert.embeddings.word_embeddings.weight.detach()
    del sbert

    model = EnsembleTupleModel(pw)

    # Load weights
    ckpt28 = torch.load('checkpoints/cnn_multihead_v28.pt', map_location='cpu', weights_only=False)
    ckpt46 = torch.load('checkpoints/cnn_multihead_v46.pt', map_location='cpu', weights_only=False)
    model.m28.load_state_dict(ckpt28['state'])
    model.m46.load_state_dict(ckpt46['state'])
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Ensemble total: {total/1e6:.1f}M params")

    # Dummy input
    dummy_input = torch.zeros(1, 32, dtype=torch.long)

    # Test forward
    with torch.no_grad():
        outputs = model(dummy_input)
    for i, name in enumerate(['fn', 'exec_type', 'param_direction', 'param_type', 'judge']):
        print(f"  {name}: {outputs[i].shape}")

    # ONNX export
    output_names = ['fn_logits', 'exec_logits', 'dir_logits', 'param_logits', 'judge_logits']
    onnx_path = 'checkpoints/nlu_v28_v46_ensemble.onnx'

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input_ids'],
        output_names=output_names,
        dynamic_axes={'input_ids': {0: 'batch_size'},
                      **{n: {0: 'batch_size'} for n in output_names}},
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nONNX saved: {onnx_path} ({size_mb:.1f}MB)")

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # Test inference
        import numpy as np
        tok = AutoTokenizer.from_pretrained('tokenizer/')

        test_texts = [
            "거실 에어컨 23도로 맞춰줘",
            "내일 날씨 어때",
            "안방 불 꺼줘",
            "오늘 뉴스 알려줘",
        ]

        import time
        print(f"\n=== ONNX 추론 테스트 ===")
        for text in test_texts:
            tk = tok(text, padding='max_length', truncation=True, max_length=32, return_tensors='np')
            start = time.time()
            outs = sess.run(None, {'input_ids': tk['input_ids'].astype(np.int64)})
            elapsed = (time.time() - start) * 1000
            fn = HEAD_I2L['fn'][outs[0][0].argmax()]
            exec_t = HEAD_I2L['exec_type'][outs[1][0].argmax()]
            dir_t = HEAD_I2L['param_direction'][outs[2][0].argmax()]
            print(f"  \"{text}\" → fn={fn}, exec={exec_t}, dir={dir_t} ({elapsed:.1f}ms)")

    except ImportError:
        print("onnxruntime not available, skipping verification")


if __name__ == '__main__':
    main()
