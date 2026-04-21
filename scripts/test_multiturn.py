#!/usr/bin/env python3
"""Multi-turn DST 벤치마크 — DST 있음 vs 없음 비교.

이전 벤치마크(TS/KE/GT)는 단일 턴 기반. DST 가치가 측정 안 됨.
이 스크립트는 멀티턴 시나리오에서 DST on/off 성능 차이 측정.
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deployment_pipeline import DeploymentPipeline


# 각 시나리오: 여러 턴, 마지막 턴의 예상 결과
SCENARIOS = [
    # (name, turns, expected_last_fn, expected_last_exec, expected_last_dir)
    ('room follow-up',
     [('거실 불 켜줘', 'light_control', 'on'),
      ('안방도', 'light_control', 'on')],
     'light_control', 'control_then_confirm', 'on'),

    ('device follow-up',
     [('안방 에어컨 켜줘', 'ac_control', 'on'),
      ('난방도', 'heat_control', 'on')],
     'heat_control', 'control_then_confirm', 'on'),

    ('correction',
     [('거실 불 켜줘', 'light_control', 'on'),
      ('아니 꺼줘', 'light_control', 'off')],
     'light_control', 'control_then_confirm', 'off'),

    ('slot fill up',
     [('난방 25도로 맞춰줘', 'heat_control', 'set'),
      ('더 올려줘', 'heat_control', 'up')],
     'heat_control', 'control_then_confirm', 'up'),

    ('slot fill down',
     [('에어컨 20도로', 'ac_control', 'set'),
      ('조금 내려줘', 'ac_control', 'down')],
     'ac_control', 'control_then_confirm', 'down'),

    ('confirm',
     [('안방 에어컨 켤까요?', 'ac_control', 'on'),
      ('응', 'ac_control', 'on')],
     'ac_control', 'control_then_confirm', 'on'),

    ('there too',
     [('거실 조명 밝게', 'light_control', 'up'),
      ('거기도', 'light_control', 'up')],
     'light_control', 'control_then_confirm', 'up'),
]


def run_scenario(pipeline, turns, use_dst=True):
    pipeline.reset_dst()
    last = None
    for text, _, _ in turns:
        last = pipeline.process(text, use_dst=use_dst)
    return last


def main():
    print("Loading pipeline...")
    p = DeploymentPipeline()
    print("Loaded.\n")
    print("=" * 70)
    print("  Multi-Turn 벤치마크 — DST on vs off")
    print("=" * 70)

    dst_on_pass = 0
    dst_off_pass = 0
    n = len(SCENARIOS)

    for name, turns, exp_fn, exp_exec, exp_dir in SCENARIOS:
        r_on = run_scenario(p, turns, use_dst=True)
        r_off = run_scenario(p, turns, use_dst=False)

        on_ok = (r_on['fn'] == exp_fn and r_on['exec_type'] == exp_exec and r_on['param_direction'] == exp_dir)
        off_ok = (r_off['fn'] == exp_fn and r_off['exec_type'] == exp_exec and r_off['param_direction'] == exp_dir)

        if on_ok: dst_on_pass += 1
        if off_ok: dst_off_pass += 1

        print(f"\n{'─'*70}")
        print(f"시나리오: {name}")
        for t, _, _ in turns:
            print(f'  입력: "{t}"')
        print(f'  기대: fn={exp_fn}, exec={exp_exec}, dir={exp_dir}')
        print(f'  DST on  [{"✓" if on_ok else "✗"}]: fn={r_on["fn"]}, exec={r_on["exec_type"]}, dir={r_on["param_direction"]}')
        print(f'  DST off [{"✓" if off_ok else "✗"}]: fn={r_off["fn"]}, exec={r_off["exec_type"]}, dir={r_off["param_direction"]}')

    print()
    print("=" * 70)
    print(f"  최종 결과: DST on {dst_on_pass}/{n}, DST off {dst_off_pass}/{n}")
    print(f"  DST 가치: +{(dst_on_pass - dst_off_pass)/n*100:.1f}%p on multi-turn")
    print("=" * 70)


if __name__ == '__main__':
    main()
