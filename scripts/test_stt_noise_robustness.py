#!/usr/bin/env python3
"""STT 노이즈 강건성 벤치마크.

실제 STT 출력이 갖는 특성 시뮬레이션:
1. Character confusion (ㅕ → ㅣ, ㅗ → ㅜ)
2. Syllable insertion (띄어쓰기 오류)
3. Phonetic similarity (미세먼지 → 미세문지)
4. Ending variation (줘 → 쥬 → 주오)

각 원본 발화에 대해 노이즈 변형 5개 생성 → 모두 같은 fn/dir 예측하는지.
"""
import os, sys
sys.path.insert(0, 'scripts')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
from deployment_pipeline import DeploymentPipeline


# Clean test cases with expected results
CLEAN_CASES = [
    ('거실 불 켜줘', 'light_control', 'on'),
    ('안방 에어컨 25도로', 'ac_control', 'set'),
    ('난방 켜줘', 'heat_control', 'on'),
    ('가스 잠가', 'gas_control', 'close'),
    ('도어락 열어', 'door_control', 'open'),
    ('환기 켜', 'vent_control', 'on'),
    ('커튼 닫아', 'curtain_control', 'close'),
    ('엘리베이터 불러', 'elevator_call', 'on'),
    ('오늘 날씨', 'weather_query', 'none'),
    ('뉴스 알려줘', 'news_query', 'none'),
]


def apply_noise(text, noise_type):
    """Apply STT-like noise."""
    if noise_type == 'typo':
        # ㅕ→ㅣ, 미세먼지→미세문지 스타일
        substitutions = {
            '먼지': '문지', '에어컨': '에어건', '난방': '남방',
            '가스': '까스', '도어락': '도얼락', '켜': '켜', '줘': '쥬',
            '환기': '한기', '커튼': '커턴', '엘리베이터': '엘레베이터',
        }
        for orig, sub in substitutions.items():
            if orig in text:
                text = text.replace(orig, sub, 1)
                break
    elif noise_type == 'no_space':
        # 모든 공백 제거
        text = text.replace(' ', '')
    elif noise_type == 'extra_space':
        # 글자 사이 공백 삽입
        text = ' '.join(text.replace(' ', ''))
    elif noise_type == 'ending_drop':
        # "줘" 제거
        text = text.rstrip('줘').rstrip()
    elif noise_type == 'verb_variant':
        # 켜줘 → 켜주세요 / 꺼줘 → 꺼주세요
        text = text.replace('켜줘', '켜주세요').replace('꺼줘', '꺼주세요')
    return text


def main():
    p = DeploymentPipeline()
    print("STT Noise Robustness Test\n")

    noise_types = ['typo', 'no_space', 'extra_space', 'ending_drop', 'verb_variant']

    total_cases = 0
    consistent_cases = 0
    failures = []

    for clean, exp_fn, exp_dir in CLEAN_CASES:
        p.reset_dst()
        r_clean = p.process(clean)
        if r_clean['fn'] != exp_fn:
            print(f"SKIP (clean wrong): \"{clean}\" → {r_clean['fn']} != {exp_fn}")
            continue

        noisy_matches = 0
        total_noise = 0
        for noise in noise_types:
            noisy = apply_noise(clean, noise)
            if noisy == clean:  # no change
                continue
            p.reset_dst()
            r_noisy = p.process(noisy)
            total_noise += 1
            total_cases += 1
            if r_noisy['fn'] == exp_fn and (r_noisy['param_direction'] == exp_dir or exp_dir == 'none'):
                noisy_matches += 1
                consistent_cases += 1
            else:
                failures.append((clean, noisy, noise, r_noisy['fn'], r_noisy['param_direction'], exp_fn, exp_dir))

        print(f"  \"{clean:<22}\" → noise 통과 {noisy_matches}/{total_noise}")

    rate = consistent_cases / total_cases * 100 if total_cases else 0
    print(f"\nSTT noise robustness: {consistent_cases}/{total_cases} = {rate:.1f}%")
    print(f"\n실패 예시 (처음 10):")
    for clean, noisy, noise, pred_fn, pred_dir, exp_fn, exp_dir in failures[:10]:
        print(f"  [{noise}] \"{clean}\" → \"{noisy}\" → {pred_fn}/{pred_dir} (expected {exp_fn}/{exp_dir})")


if __name__ == '__main__':
    main()
