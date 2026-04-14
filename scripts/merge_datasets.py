#!/usr/bin/env python3
"""전체 데이터 병합 + 밸런싱 + train/val/test 분리"""
import json, random, hashlib
from collections import Counter

random.seed(42)

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def dedup(data):
    """발화 기준 중복 제거 (첫 번째 것 유지)"""
    seen = set()
    result = []
    for d in data:
        key = d['utterance'].strip().lower()
        if key not in seen and len(key) > 1:
            seen.add(key)
            result.append(d)
    return result

def main():
    # 1. CNN 재매핑 데이터 (16,879)
    cnn = load_json('data/train_multihead_v1.json')
    print(f"CNN 재매핑: {len(cnn)}")

    # 2. 증강 데이터 (8,045)
    aug = load_json('data/train_augmented_v1.json')
    print(f"증강 데이터: {len(aug)}")

    # 3. 기존 SAP 데이터 (7,719) — fn이 19개 체계와 다르므로 info_query 매핑 필요
    sap = load_json('data/train_sap_v15.json')
    sap_mapped = []
    for d in sap:
        fn = d['labels']['fn']
        # info_query → 키워드로 분리
        if fn == 'info_query':
            utt_lo = d['utterance'].lower()
            if any(w in utt_lo for w in ['할 수 있', '기능', '뭐 할']):
                fn = 'system_meta'
            elif any(w in utt_lo for w in ['만들', '개발', '누구']):
                fn = 'system_meta'
            elif any(w in utt_lo for w in ['이름', '뭐라고 불']):
                fn = 'system_meta'
            elif any(w in utt_lo for w in ['사용법', '어떻게 써', '어케']):
                fn = 'system_meta'
            elif any(w in utt_lo for w in ['몇 시', '시간', '시각']):
                fn = 'home_info'
            elif any(w in utt_lo for w in ['볼륨', '소리', '음량']):
                fn = 'home_info'
            elif any(w in utt_lo for w in ['밝기', '화면', '디스플레이']):
                fn = 'home_info'
            elif any(w in utt_lo for w in ['주식', '코스피', '코스닥', '종목']):
                fn = 'market_query'
            elif any(w in utt_lo for w in ['유가', '주유', '기름값', '휘발유']):
                fn = 'market_query'
            elif any(w in utt_lo for w in ['병원', '약국', '의료', '소아과']):
                fn = 'medical_query'
            elif any(w in utt_lo for w in ['주차', '차량', '충전']):
                fn = 'vehicle_manage'
            elif any(w in utt_lo for w in ['단지', '관리', '커뮤니티', '헬스']):
                fn = 'home_info'
            else:
                fn = 'home_info'  # default

        entry = {
            'utterance': d['utterance'],
            'flat_intent': d.get('flat_intent', fn),
            'labels': {
                'fn': fn,
                'exec_type': d['labels'].get('exec_type', 'query_then_respond'),
                'param_direction': d['labels'].get('param_direction', 'none'),
                'param_type': d['labels'].get('param_type', 'none'),
                'judge': d['labels'].get('judge', 'none'),
            },
            'source': 'sap_v15'
        }
        sap_mapped.append(entry)
    print(f"SAP 매핑: {len(sap_mapped)}")

    # 병합
    all_data = cnn + aug + sap_mapped
    print(f"\n병합 전: {len(all_data)}")

    # 중복 제거
    all_data = dedup(all_data)
    print(f"중복 제거 후: {len(all_data)}")

    # weather_query 다운샘플링 (10,000+ → 1,500)
    weather = [d for d in all_data if d['labels']['fn'] == 'weather_query']
    non_weather = [d for d in all_data if d['labels']['fn'] != 'weather_query']
    random.shuffle(weather)
    weather_sampled = weather[:1500]
    all_data = non_weather + weather_sampled
    print(f"weather 다운샘플링: {len(weather)} → {len(weather_sampled)}")

    # 셔플
    random.shuffle(all_data)

    # Train / Val / Test 분리 (85 / 10 / 5)
    n = len(all_data)
    train_end = int(n * 0.85)
    val_end = int(n * 0.95)

    train = all_data[:train_end]
    val = all_data[train_end:val_end]
    test = all_data[val_end:]

    # 저장
    for split, data in [('train', train), ('val', val), ('test', test)]:
        path = f'data/{split}_final.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 통계
    print(f"\n=== 최종 분포 ===")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}, Total: {n}")

    fn_c = Counter(d['labels']['fn'] for d in all_data)
    exec_c = Counter(d['labels']['exec_type'] for d in all_data)
    dir_c = Counter(d['labels']['param_direction'] for d in all_data)

    print(f"\n--- fn ({len(fn_c)}개) ---")
    for k, v in fn_c.most_common():
        pct = v / n * 100
        bar = '█' * int(pct / 2)
        print(f"  {k:20s}: {v:5d} ({pct:4.1f}%) {bar}")

    print(f"\n--- exec_type ---")
    for k, v in exec_c.most_common():
        print(f"  {k:25s}: {v:5d}")

    print(f"\n--- param_direction ---")
    for k, v in dir_c.most_common():
        print(f"  {k:10s}: {v:5d}")


if __name__ == '__main__':
    main()
