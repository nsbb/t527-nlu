#!/usr/bin/env python3
"""골든셋 → NPU 추론용 .bin 일괄 생성 + ONNX 레퍼런스 logits 저장.

생성:
  /tmp/npu_eval_99/inputs.bin    (N * 24576 bytes, concat)
  /tmp/npu_eval_99/refs.json     (N개의 정답 + ONNX ensemble logits + argmax)

디바이스에서 NPU로 .bin 일괄 추론 후 refs.json와 비교.
"""
import sys, os, json
import numpy as np
import onnx, onnxruntime as ort
from onnx.utils import Extractor

CHK = '/home/nsbb/travail/claude/T527/t527-nlu/checkpoints'
TOK = '/home/nsbb/travail/claude/T527/t527-nlu/tokenizer/'
INPUT_SCALE = 0.003369783
INPUT_ZP = 155
HEAD_FN = ['light_control','heat_control','ac_control','vent_control','gas_control',
           'door_control','curtain_control','elevator_call','security_mode',
           'schedule_manage','weather_query','news_query','traffic_query',
           'energy_query','home_info','system_meta','market_query',
           'medical_query','vehicle_manage','unknown']
HEAD_DIR = ['none','up','down','set','on','off','open','close','stop']
HEAD_EXEC = ['query_then_respond','control_then_confirm','query_then_judge','direct_respond','clarify']

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else '/home/nsbb/travail/claude/T527/t527-nlu/data/golden_test_100.json'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/npu_eval_99'
    os.makedirs(out_dir, exist_ok=True)

    sys.path.insert(0, '/home/nsbb/travail/claude/T527/t527-nlu/scripts')
    from preprocess import preprocess
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)

    # Sub-model: token_ids → embedded
    full = onnx.load(f'{CHK}/nlu_v46_generalization.onnx')
    sub = Extractor(full).extract_model(['token_ids'], ['/m/token_emb/Gather_output_0'])
    sess_emb = ort.InferenceSession(sub.SerializeToString(), providers=['CPUExecutionProvider'])

    # Full ensemble
    sess_ens = ort.InferenceSession(f'{CHK}/nlu_v28_v46_ensemble.onnx', providers=['CPUExecutionProvider'])

    data = json.load(open(in_path))
    N = len(data)
    print(f'Processing {N} samples from {in_path}...')

    bin_data = bytearray()
    refs = []
    for i, item in enumerate(data):
        utt = item['utterance']
        pp = preprocess(utt)
        enc = tok(pp, padding='max_length', max_length=32, truncation=True, return_tensors='np')
        ids = enc['input_ids'].astype(np.int64)

        embedded = sess_emb.run(None, {'token_ids': ids})[0]
        embed_q = np.round(embedded / INPUT_SCALE + INPUT_ZP).clip(0, 255).astype(np.uint8)
        bin_data += embed_q.tobytes()

        ens_outs = sess_ens.run(None, {'input_ids': ids})
        ens = dict(zip([o.name for o in sess_ens.get_outputs()], ens_outs))

        refs.append({
            'idx': i,
            'utterance': utt,
            'gt_fn': item.get('fn'),
            'gt_exec': item.get('exec'),
            'gt_dir': item.get('dir'),
            'onnx_fn': HEAD_FN[int(ens['fn_logits'].argmax())],
            'onnx_exec': HEAD_EXEC[int(ens['exec_logits'].argmax())],
            'onnx_dir': HEAD_DIR[int(ens['dir_logits'].argmax())],
            'onnx_fn_logits': ens['fn_logits'][0].tolist(),
            'onnx_dir_logits': ens['dir_logits'][0].tolist(),
        })
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{N} done')

    bin_path = f'{out_dir}/inputs.bin'
    ref_path = f'{out_dir}/refs.json'
    with open(bin_path, 'wb') as f:
        f.write(bytes(bin_data))
    with open(ref_path, 'w') as f:
        json.dump({'count': N, 'samples': refs}, f, ensure_ascii=False)

    # Quick ONNX accuracy
    fn_match = sum(1 for r in refs if r['onnx_fn'] == r['gt_fn'])
    exec_match = sum(1 for r in refs if r['onnx_exec'] == r['gt_exec'])
    dir_match = sum(1 for r in refs if r['onnx_dir'] == r['gt_dir'])
    combo = sum(1 for r in refs if r['onnx_fn']==r['gt_fn'] and r['onnx_exec']==r['gt_exec'] and r['onnx_dir']==r['gt_dir'])
    print(f'\n=== ONNX Ensemble (CPU) on {N} samples ===')
    print(f'fn:    {fn_match}/{N} ({100*fn_match/N:.1f}%)')
    print(f'exec:  {exec_match}/{N} ({100*exec_match/N:.1f}%)')
    print(f'dir:   {dir_match}/{N} ({100*dir_match/N:.1f}%)')
    print(f'combo: {combo}/{N} ({100*combo/N:.1f}%)')
    print(f'\nSaved {bin_path} ({len(bin_data)} bytes), {ref_path}')

if __name__ == '__main__':
    main()
