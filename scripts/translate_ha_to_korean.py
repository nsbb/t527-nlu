#!/usr/bin/env python3
"""Home-Assistant 영어 → 한국어 규칙 기반 번역"""
import json, random, re
from collections import Counter

random.seed(42)

# Room 번역
ROOM_EN_TO_KR = {
    'kitchen': '주방', 'living room': '거실', 'bedroom': '안방', 'bathroom': '화장실',
    'master bedroom': '안방', 'kids room': '아이방', "kids' room": '아이방',
    'nursery': '아이방', 'garage': '차고', 'hallway': '복도', 'office': '서재',
    'dining room': '식당', 'laundry': '세탁실', 'basement': '지하', 'attic': '다락',
    'porch': '현관', 'balcony': '발코니', 'garden': '정원', 'foyer': '현관',
    'guest room': '게스트룸', 'sunroom': '거실', 'library': '서재',
    'storage room': '창고', 'wine cellar': '창고', 'home theater': '거실',
    'staircase': '계단', 'entryway': '현관', 'front': '현관', 'back': '후문',
    'side': '옆', 'main': '메인', 'upstairs': '윗층', 'downstairs': '아래층',
    'right-side': '오른쪽', 'left-side': '왼쪽',
}

# 디바이스 번역
DEVICE_KR = {
    'door_control': '도어락',
    'curtain_control': '커튼',
    'heat_control': '난방',
    'ac_control': '에어컨',
    'vent_control': '환기',
    'schedule_manage': '타이머',
}

# 어미 풀
ENDINGS_CONTROL = ['켜줘','켜봐','켜주세요','켜줄래','켜','좀 켜줘']
ENDINGS_OFF = ['꺼줘','꺼봐','꺼주세요','꺼','좀 꺼줘']
ENDINGS_OPEN = ['열어줘','열어봐','열어주세요','열어','좀 열어줘']
ENDINGS_CLOSE = ['닫아줘','닫아봐','닫아주세요','닫아','좀 닫아줘']
ENDINGS_LOCK = ['잠가줘','잠가봐','잠가주세요','잠가','잠금해줘']
ENDINGS_UNLOCK = ['열어줘','열어봐','열어주세요','열어']
ENDINGS_SET = ['맞춰줘','설정해줘','해줘','맞춰','설정해']
ENDINGS_UP = ['올려줘','높여줘','세게 해줘','올려','강하게']
ENDINGS_DOWN = ['낮춰줘','줄여줘','약하게 해줘','낮춰','약하게']
ENDINGS_STOP = ['멈춰줘','멈춰','중지해줘','그만']


def extract_room(text):
    """영어 텍스트에서 room 추출"""
    text_lo = text.lower()
    for en, kr in sorted(ROOM_EN_TO_KR.items(), key=lambda x: -len(x[0])):
        if en in text_lo:
            return kr
    return ''


def translate_door(text, direction):
    room = extract_room(text)
    room_prefix = f"{room} " if room else ''

    if direction == 'close':  # lock
        ending = random.choice(ENDINGS_LOCK)
        templates = [
            f"{room_prefix}도어락 {ending}",
            f"{room_prefix}현관문 {ending}",
            f"{room_prefix}문 잠가줘",
        ]
    else:  # unlock
        ending = random.choice(ENDINGS_UNLOCK)
        templates = [
            f"{room_prefix}도어락 {ending}",
            f"{room_prefix}현관문 {ending}",
            f"{room_prefix}문 열어줘",
        ]
    return random.choice(templates)


def translate_curtain(text, direction):
    room = extract_room(text)
    room_prefix = f"{room} " if room else ''

    if direction == 'open':
        ending = random.choice(ENDINGS_OPEN)
        return f"{room_prefix}커튼 {ending}"
    elif direction == 'close':
        ending = random.choice(ENDINGS_CLOSE)
        return f"{room_prefix}커튼 {ending}"
    else:  # stop
        ending = random.choice(ENDINGS_STOP)
        return f"{room_prefix}커튼 {ending}"


def translate_climate(text, direction, fn):
    room = extract_room(text)
    room_prefix = f"{room} " if room else ''

    if fn == 'heat_control':
        # set_temperature
        m = re.search(r'(\d+)\s*degrees?', text)
        if m:
            temp = m.group(1)
            # Fahrenheit → Celsius 대략 변환
            try:
                f_temp = int(temp)
                c_temp = round((f_temp - 32) * 5 / 9)
                c_temp = max(15, min(30, c_temp))
            except:
                c_temp = 23
            ending = random.choice(ENDINGS_SET)
            templates = [
                f"{room_prefix}난방 {c_temp}도로 {ending}",
                f"{room_prefix}온도 {c_temp}도로 {ending}",
                f"{room_prefix}난방 온도 {c_temp}도",
            ]
            return random.choice(templates)
        ending = random.choice(ENDINGS_SET)
        return f"{room_prefix}난방 온도 {ending}"

    else:  # ac_control
        if direction == 'on':
            ending = random.choice(ENDINGS_CONTROL)
            return f"{room_prefix}에어컨 {ending}"
        elif direction == 'off':
            ending = random.choice(ENDINGS_OFF)
            return f"{room_prefix}에어컨 {ending}"
        else:  # set mode/fan
            modes = ['자동','송풍','제습','냉방','약풍','강풍']
            mode = random.choice(modes)
            ending = random.choice(ENDINGS_SET)
            return f"{room_prefix}에어컨 {mode} 모드로 {ending}"


def translate_fan(text, direction):
    room = extract_room(text)
    room_prefix = f"{room} " if room else ''

    if direction == 'on':
        ending = random.choice(ENDINGS_CONTROL)
        templates = [
            f"{room_prefix}환기 {ending}",
            f"{room_prefix}환풍기 {ending}",
            f"{room_prefix}환기시스템 {ending}",
        ]
    elif direction == 'off':
        ending = random.choice(ENDINGS_OFF)
        templates = [
            f"{room_prefix}환기 {ending}",
            f"{room_prefix}환풍기 {ending}",
        ]
    elif direction == 'up':
        ending = random.choice(ENDINGS_UP)
        templates = [
            f"{room_prefix}환기 {ending}",
            f"{room_prefix}환기 바람 세게",
            f"{room_prefix}환풍기 {ending}",
        ]
    else:  # down
        ending = random.choice(ENDINGS_DOWN)
        templates = [
            f"{room_prefix}환기 {ending}",
            f"{room_prefix}환기 바람 약하게",
            f"{room_prefix}환풍기 {ending}",
        ]
    return random.choice(templates)


def translate_timer(text, direction):
    # 타이머 → 예약/알람
    m = re.search(r'(\d+)\s*(minutes?|hours?|seconds?)', text)
    if m:
        num = m.group(1)
        unit = m.group(2)
        if 'hour' in unit: kr_unit = '시간'
        elif 'minute' in unit: kr_unit = '분'
        else: kr_unit = '초'

        ending = random.choice(ENDINGS_SET)
        templates = [
            f"{num}{kr_unit} 후 알려줘",
            f"{num}{kr_unit} 타이머 {ending}",
            f"{num}{kr_unit} 뒤에 알람 맞춰줘",
        ]
        return random.choice(templates)

    if direction == 'off':
        return random.choice(["타이머 취소해줘", "알람 취소", "예약 취소해줘"])
    return random.choice(["타이머 설정해줘", "알람 맞춰줘", "예약해줘"])


def translate(item, fn):
    text = item['en']
    direction = item['labels']['param_direction']

    if fn == 'door_control':
        return translate_door(text, direction)
    elif fn == 'curtain_control':
        return translate_curtain(text, direction)
    elif fn in ('heat_control', 'ac_control'):
        return translate_climate(text, direction, fn)
    elif fn == 'vent_control':
        return translate_fan(text, direction)
    elif fn == 'schedule_manage':
        return translate_timer(text, direction)
    return None


def main():
    with open('data/ha_english_clean.json') as f:
        eng_data = json.load(f)

    all_translated = []
    for fn, items in eng_data.items():
        ok = 0
        for item in items:
            kr = translate(item, fn)
            if kr:
                kr = re.sub(r'\s+', ' ', kr).strip()
                if len(kr) > 2:
                    entry = {
                        'utterance': kr,
                        'labels': item['labels'],
                        'source': f'ha_translated_{fn}',
                        'en_original': item['en'][:50],
                    }
                    all_translated.append(entry)
                    ok += 1
        print(f"{fn}: {ok}/{len(items)} 번역 성공")

    # 중복 제거
    seen = set()
    deduped = []
    for d in all_translated:
        key = d['utterance'].strip()
        if key not in seen:
            seen.add(key)
            deduped.append(d)

    with open('data/ha_translated_kr.json', 'w', encoding='utf-8') as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    fn_c = Counter(d['labels']['fn'] for d in deduped)
    print(f"\n번역 완료: {len(deduped)}개 (중복제거 후)")
    for k, v in fn_c.most_common():
        print(f"  {k}: {v}")

    print(f"\n--- 샘플 ---")
    for fn in fn_c:
        samples = [d for d in deduped if d['labels']['fn'] == fn][:3]
        print(f"\n[{fn}]")
        for s in samples:
            print(f"  \"{s['utterance']}\" ← \"{s['en_original']}\"")


if __name__ == '__main__':
    main()
