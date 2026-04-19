#!/usr/bin/env python3
"""변환된 KoELECTRA 데이터를 기존 멀티헤드 학습데이터에 합침

사용법:
    python3 scripts/merge_koelectra_data.py --base data/train_final_v32.json --version 33

입력:
    --base: 기존 멀티헤드 학습데이터 (e.g. train_final_v32.json)
    data/koelectra_converted_train.json  (13,540개, 자동 로드)

출력:
    data/train_final_v{version}.json     — 합친 학습데이터
    data/val_final_v{version}.json       — 기존 val 복사 (koelectra val은 별도 테스트용)

중복 제거: 발화문 정규화 후 exact match로 중복 제거
"""
import json, argparse, re, unicodedata
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    """발화문 정규화 (중복 판정용)"""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', '', text)  # 공백 제거
    text = text.lower()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='기존 train 파일 경로')
    parser.add_argument('--version', required=True, help='출력 버전 번호')
    parser.add_argument('--koelectra', default='data/koelectra_converted_train.json',
                        help='변환된 KoELECTRA 데이터 (기본: data/koelectra_converted_train.json)')
    args = parser.parse_args()

    base_path = Path(args.base)
    ke_path = Path(args.koelectra)

    with open(base_path, 'r', encoding='utf-8') as f:
        base_data = json.load(f)
    with open(ke_path, 'r', encoding='utf-8') as f:
        ke_data = json.load(f)

    print(f"기존 데이터: {len(base_data)}개 ({base_path.name})")
    print(f"KoELECTRA:   {len(ke_data)}개")

    # 기존 데이터의 발화문 set (중복 판정)
    existing = set(normalize(d['utterance']) for d in base_data)

    # 중복 제거하면서 추가
    added = 0
    dup = 0
    for entry in ke_data:
        norm = normalize(entry['utterance'])
        if norm in existing:
            dup += 1
            continue
        existing.add(norm)
        base_data.append(entry)
        added += 1

    print(f"\n중복: {dup}개 제거")
    print(f"추가: {added}개")
    print(f"최종: {len(base_data)}개")

    # fn 분포
    fn_c = Counter(d['labels']['fn'] for d in base_data)
    print(f"\n--- 합친 fn 분포 ({len(fn_c)}개 카테고리) ---")
    for k, v in fn_c.most_common():
        print(f"  {k:20s}: {v:5d}")

    # source 분포
    src_c = Counter(d.get('source', 'none') for d in base_data)
    print(f"\n--- source 분포 ---")
    for k, v in src_c.most_common():
        print(f"  {k:25s}: {v:5d}")

    # 저장
    out_dir = Path('data')
    train_out = out_dir / f'train_final_v{args.version}.json'
    with open(train_out, 'w', encoding='utf-8') as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {train_out}")

    # val은 기존 것 복사 (koelectra val은 별도 교차검증용으로 보존)
    val_src = base_path.parent / base_path.name.replace('train_', 'val_')
    if val_src.exists():
        val_out = out_dir / f'val_final_v{args.version}.json'
        with open(val_src) as f:
            val_data = json.load(f)
        with open(val_out, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        print(f"       {val_out} (기존 val 복사, {len(val_data)}개)")

    print(f"\n★ KoELECTRA val (1,536개)은 data/koelectra_converted_val.json에 별도 보존")
    print(f"  → 교차검증 테스트셋으로 활용 가능")


if __name__ == '__main__':
    main()
