#!/usr/bin/env python3
"""Extract CNN body subgraph from full NLU ONNX (v28 or v46).

Input ONNX: token_ids → ... → 5 logits
Output ONNX: embedded[1,32,768] → 5 logits (CNN body only, NPU-targetable)
"""
import sys, onnx
from onnx.utils import Extractor

def extract(full_path, out_path):
    m = onnx.load(full_path)
    embedded_tensor = '/m/token_emb/Gather_output_0'
    out_names = [o.name for o in m.graph.output]
    print(f'Outputs: {out_names}')

    ex = Extractor(m)
    sub = ex.extract_model([embedded_tensor], out_names)

    # Rename input from /m/token_emb/Gather_output_0 → "embedded"
    for inp in sub.graph.input:
        if inp.name == embedded_tensor:
            inp.name = 'embedded'
    for n in sub.graph.node:
        for i, x in enumerate(n.input):
            if x == embedded_tensor:
                n.input[i] = 'embedded'

    onnx.save(sub, out_path)
    print(f'Saved {out_path} ({len(sub.graph.node)} nodes)')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: extract_cnn_body.py <full_onnx> <out_onnx>')
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
