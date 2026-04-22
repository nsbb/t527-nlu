#!/usr/bin/env python3
"""v2 파이프라인 대화형 테스트.

사용법:
    python3 scripts/test_v2_interactive.py              # 인터랙티브 모드
    python3 scripts/test_v2_interactive.py "거실 불 켜줘"  # 단일 입력
    python3 scripts/test_v2_interactive.py --demo        # 데모 (22가지 시나리오)
    python3 scripts/test_v2_interactive.py --multiturn   # 멀티턴 DST 데모
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment_pipeline_v2 import DeploymentPipelineV2


def format_result(r, show_debug=True):
    """결과를 읽기 좋게 포맷"""
    lines = []
    lines.append(f"  📥 입력:     \"{r['raw']}\"")
    if r['preprocessed'] != r['raw']:
        lines.append(f"  🔧 전처리:    \"{r['preprocessed']}\"")
    if show_debug:
        lines.append(f"  🧠 멀티헤드: fn={r['fn']}, exec={r['exec_type']}, dir={r['param_direction']}")
        room_str = r['room'] if r['room'] != 'none' else '-'
        value_str = str(r['value']) if r['value'] else '-'
        lines.append(f"              room={room_str}, value={value_str}")
        if r.get('dst_applied'):
            lines.append(f"              [DST 적용]")
    lines.append(f"  💬 응답:     {r['response']}")
    return '\n'.join(lines)


def demo_single():
    """22가지 다양한 시나리오 데모"""
    p = DeploymentPipelineV2()

    scenarios = [
        # 제어
        "거실 불 켜줘",
        "안방 에어컨 23도로 맞춰줘",
        "전체 난방 켜 줘",
        "가스 밸브 잠금해",
        "거실 전동커튼 열어줘",
        "30분 후에 거실 에어컨 꺼줘",

        # 조회
        "지금 집 상태 어때?",
        "지금 몇 시야?",
        "에어컨 설정 온도는?",
        "주방 불 켜져있어?",

        # 정보 (부가)
        "오늘 날씨 어때?",
        "강남역까지 얼마나 걸려?",
        "삼성전자 주가 얼마야?",
        "근처 병원 어디 있어?",

        # 판단
        "오늘 뭐 입고 나가야 돼?",
        "오늘 세차해도 되나?",
        "창문 열어도 괜찮아?",

        # 비상/시스템
        "가스 냄새 나",
        "알람 7시",
        "외출모드 실행해 줘",

        # Unknown 조립 (v2 우위)
        "거실 뭐 좀 켜",
        "너무 답답해",
    ]

    print("\n" + "=" * 70)
    print("v2 파이프라인 데모 — 22가지 시나리오")
    print("=" * 70 + "\n")

    for i, utt in enumerate(scenarios, 1):
        p.reset_dst()
        r = p.process(utt, use_dst=True)
        print(f"[{i:2d}] {format_result(r, show_debug=False)}")
        print()


def demo_multiturn():
    """멀티턴 DST 시나리오 — 이전 맥락 상속"""
    p = DeploymentPipelineV2()

    scenarios = [
        # 시나리오 1: 온도 조절 체인
        [
            "거실 난방 25도로 맞춰줘",
            "1도 더 올려",        # → 26도 (value inheritance)
            "2도 낮춰",           # → 24도
            "꺼",                 # → 난방 꺼 (fn inheritance)
        ],

        # 시나리오 2: 방 follow-up
        [
            "거실 불 켜줘",
            "안방도",              # → 안방 불 켜
            "주방도",              # → 주방 불 켜
        ],

        # 시나리오 3: 교정 (아니)
        [
            "난방 켜",
            "아니 에어컨",         # → 에어컨 켜 (dir inherit)
            "다시 난방",           # → 난방 켜
        ],

        # 시나리오 4: 타이머 + 취소
        [
            "30분 후 난방 꺼",
            "취소",               # → schedule cancel
        ],
    ]

    print("\n" + "=" * 70)
    print("v2 파이프라인 멀티턴 DST 데모")
    print("=" * 70)

    for i, scenario in enumerate(scenarios, 1):
        p.reset_dst()
        print(f"\n--- 시나리오 {i} ---")
        for turn, utt in enumerate(scenario, 1):
            r = p.process(utt, use_dst=True)
            marker = ' [DST]' if r['dst_applied'] else ''
            print(f"  턴 {turn}: \"{utt}\"{marker}")
            print(f"         fn={r['fn']}/{r['param_direction']}, value={r['value']}")
            print(f"         → {r['response']}")


def interactive():
    """대화형 모드"""
    p = DeploymentPipelineV2()

    print("\n" + "=" * 70)
    print("v2 파이프라인 대화형 테스트")
    print("=" * 70)
    print("  명령어:")
    print("    q/quit/exit : 종료")
    print("    reset       : DST 세션 초기화")
    print("    debug       : 디버그 정보 토글")
    print("    dst         : DST on/off 토글")
    print()

    show_debug = True
    use_dst = True

    while True:
        try:
            prompt_str = f"\n[DST={'on' if use_dst else 'off'}, debug={'on' if show_debug else 'off'}] 입력> "
            utt = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료.")
            break

        if not utt:
            continue
        if utt in ('q', 'quit', 'exit'):
            print("종료.")
            break
        if utt == 'reset':
            p.reset_dst()
            print("  [DST 세션 초기화됨]")
            continue
        if utt == 'debug':
            show_debug = not show_debug
            print(f"  [debug={'on' if show_debug else 'off'}]")
            continue
        if utt == 'dst':
            use_dst = not use_dst
            print(f"  [use_dst={'on' if use_dst else 'off'}]")
            continue

        # 실제 처리
        r = p.process(utt, use_dst=use_dst)
        print(format_result(r, show_debug=show_debug))


def main():
    args = sys.argv[1:]

    if not args:
        interactive()
        return

    if args[0] == '--demo':
        demo_single()
        return

    if args[0] == '--multiturn':
        demo_multiturn()
        return

    if args[0] in ('-h', '--help'):
        print(__doc__)
        return

    # 단일 입력
    p = DeploymentPipelineV2()
    utt = ' '.join(args)
    r = p.process(utt, use_dst=True)
    print(format_result(r))


if __name__ == '__main__':
    main()
