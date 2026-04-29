#!/usr/bin/env python3
"""NLU + 실제 API 연결 테스트 — 키 없이 동작.

날씨/대기질: Open-Meteo (무료, 키 없음)
뉴스: 경향신문 RSS (무료, 키 없음)

OO/00 placeholder → 실데이터로 자동 치환

사용:
    python3 test_api.py          # 대화형
    python3 test_api.py -v       # verbose (exec/rooms 출력)
"""

import os, sys, re, json, time, urllib.request, urllib.parse, argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

LAT, LON  = 37.4837, 127.0324
DISTRICT  = '서초구'

RESET  = '\033[0m'; BOLD = '\033[1m'
GREEN  = '\033[92m'; YELLOW = '\033[93m'
CYAN   = '\033[96m'; GRAY = '\033[90m'
RED    = '\033[91m'

DAYS_KR = ['월', '화', '수', '목', '금', '토', '일']


def _get_json(url, timeout=10):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {}


def _get_raw(url, timeout=6):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode('utf-8', errors='replace')
    except Exception:
        return ''


# ─────────────────────────────────────────────────
# WMO 날씨 코드 → 한국어
# ─────────────────────────────────────────────────
WMO_KR = {
    0: '맑음', 1: '대체로 맑음', 2: '구름 많음', 3: '흐림',
    45: '안개', 48: '안개',
    51: '이슬비', 53: '이슬비', 55: '이슬비',
    61: '비', 63: '비', 65: '강한 비',
    71: '눈', 73: '눈', 75: '강한 눈',
    80: '소나기', 81: '소나기', 82: '강한 소나기',
    85: '눈 소나기', 86: '눈 소나기',
    95: '천둥번개', 96: '천둥번개', 99: '천둥번개',
}
RAIN_CODES = {61, 63, 65, 80, 81, 82, 95, 96, 99, 51, 53, 55}
SNOW_CODES = {71, 73, 75, 85, 86}
THUNDER_CODES = {95, 96, 99}


def _uv_grade(uv):
    if uv is None: return '보통'
    if uv < 3:    return '낮음'
    if uv < 6:    return '보통'
    if uv < 8:    return '높음'
    if uv < 11:   return '매우높음'
    return '위험'


def _pm_grade(pm10, pm25):
    g10 = ('좋음' if pm10 <= 30 else '보통' if pm10 <= 80 else '나쁨' if pm10 <= 150 else '매우나쁨')
    g25 = ('좋음' if pm25 <= 15 else '보통' if pm25 <= 35 else '나쁨' if pm25 <= 75 else '매우나쁨')
    return g10, g25


def _weather_cond_slot(w):
    """르엘 슬롯 선택: 맑고/흐리고/비가 오고/춥고/덥고 중 하나."""
    code = w['code']
    temp = w['temp']
    if code in SNOW_CODES:
        return '눈이 오고'
    if code in RAIN_CODES:
        return '비가 오고'
    if temp >= 28:
        return '덥고'
    if temp <= 5:
        return '춥고'
    if code >= 3:   # 구름많음/흐림/안개
        return '흐리고'
    return '맑고'


# ─────────────────────────────────────────────────
# API 호출
# ─────────────────────────────────────────────────
def get_weather():
    url = (f'https://api.open-meteo.com/v1/forecast'
           f'?latitude={LAT}&longitude={LON}'
           f'&current=temperature_2m,apparent_temperature,relative_humidity_2m,'
           f'wind_speed_10m,precipitation,weather_code'
           f'&daily=temperature_2m_min,temperature_2m_max,'
           f'precipitation_probability_max,uv_index_max,sunrise,sunset'
           f'&timezone=Asia%2FSeoul')
    d = _get_json(url)
    if not d or 'current' not in d:
        return None

    c = d['current']
    daily = d.get('daily', {})

    code  = c.get('weather_code', 0)
    cond  = WMO_KR.get(code, '흐림')
    prec  = c.get('precipitation', 0) or 0
    rain_prob = daily.get('precipitation_probability_max', [None])[0]
    if rain_prob is None:
        rain_prob = 85 if prec > 0 else (35 if code in range(1, 4) else 5)

    # 일출/일몰 파싱 (ISO 8601: "2026-04-27T05:32")
    sunrise_raw = (daily.get('sunrise') or [''])[0]
    sunset_raw  = (daily.get('sunset')  or [''])[0]
    def _hhmm(s):
        m = re.search(r'T(\d+):(\d+)', s)
        if not m: return None
        h, mn = int(m.group(1)), int(m.group(2))
        ampm = '오전' if h < 12 else '오후'
        return f'{ampm} {h if h <= 12 else h-12}시 {mn:02d}분'

    uv_raw = (daily.get('uv_index_max') or [None])[0]
    min_temp = (daily.get('temperature_2m_min') or [None])[0]
    max_temp = (daily.get('temperature_2m_max') or [None])[0]

    return {
        'temp':      round(c.get('temperature_2m', 0), 1),
        'feels':     round(c.get('apparent_temperature', 0), 1),
        'humidity':  c.get('relative_humidity_2m', 0),
        'wind':      round(c.get('wind_speed_10m', 0), 1),
        'condition': cond,
        'code':      code,
        'rain_prob': int(rain_prob),
        'prec':      prec,
        'min_temp':  round(min_temp, 1) if min_temp is not None else None,
        'max_temp':  round(max_temp, 1) if max_temp is not None else None,
        'uv':        uv_raw,
        'uv_grade':  _uv_grade(uv_raw),
        'sunrise':   _hhmm(sunrise_raw),
        'sunset':    _hhmm(sunset_raw),
    }


def get_air():
    url = (f'https://air-quality-api.open-meteo.com/v1/air-quality'
           f'?latitude={LAT}&longitude={LON}&current=pm10,pm2_5')
    d = _get_json(url)
    if not d or 'current' not in d:
        return None
    c = d['current']
    pm10 = c.get('pm10') or 0
    pm25 = c.get('pm2_5') or 0
    g10, g25 = _pm_grade(pm10, pm25)
    return {
        'pm10': round(pm10, 1), 'pm25': round(pm25, 1),
        'pm10_grade': g10, 'pm25_grade': g25,
    }


def get_news(n=3):
    xml = _get_raw('https://www.khan.co.kr/rss/rssdata/total_news.xml')
    if not xml:
        return None
    titles = re.findall(r'<title><!\[CDATA\[(.+?)\]\]></title>', xml)
    if not titles:
        titles = re.findall(r'<title>([^<]{5,})</title>', xml)
    titles = [t.strip() for t in titles if len(t.strip()) > 5][:n]
    return titles or None


# ─────────────────────────────────────────────────
# OO placeholder → 실데이터 치환
# ─────────────────────────────────────────────────
def fill_placeholders(resp, w, a):
    """NLU 응답의 OO/00 placeholder를 실데이터로 치환."""
    if not resp:
        return resp

    now = datetime.now()

    # 날짜/요일
    date_str = f'{now.month}월 {now.day}일 {DAYS_KR[now.weekday()]}요일'
    resp = re.sub(r'N월\s*M일\s*OO요일', date_str, resp)

    # 지역명
    resp = re.sub(r'OO구\s*OO동', DISTRICT, resp)

    if w:
        t     = w['temp']
        feels = w['feels']
        hum   = w['humidity']
        rain  = w['rain_prob']
        cond  = w['condition']
        code  = w['code']

        # 르엘 슬롯: 맑고/흐리고/비가 오고/춥고/덥고 → 하나 선택
        slot = _weather_cond_slot(w)
        resp = re.sub(r'맑고/흐리고/비가\s*오고/춥고/덥고', slot, resp)

        # 기온
        resp = re.sub(r'기온은\s*OO도', f'기온은 {t}도', resp)
        resp = re.sub(r'기온은\s*00도', f'기온은 {t}도', resp)
        resp = re.sub(r'체감온도는\s*OO도', f'체감온도는 {feels}도', resp)

        # OO도 (지역+기온 패턴)
        resp = re.sub(r'은\s*OO도입니다', f'은 {t}도입니다', resp)

        # 최저/최고
        if w['min_temp'] is not None:
            resp = re.sub(r'최저 기온은\s*00도', f'최저 기온은 {w["min_temp"]}도', resp)
            resp = re.sub(r'최저\s*00도', f'최저 {round(w["min_temp"])}도', resp)
        if w['max_temp'] is not None:
            resp = re.sub(r'최고 기온은\s*00도', f'최고 기온은 {w["max_temp"]}도', resp)
            resp = re.sub(r'최고\s*00도', f'최고 {round(w["max_temp"])}도', resp)

        # 강수 확률
        resp = re.sub(r'강수 확률은\s*0%', f'강수 확률은 {rain}%', resp)
        resp = re.sub(r'강수 확률은\s*OO%', f'강수 확률은 {rain}%', resp)

        # 눈/비/천둥 예보 — 실제 날씨 코드 기반으로 문장 교체
        if '강설 예보는 없습니다' in resp:
            if code in SNOW_CODES:
                resp = resp.replace('강설 예보는 없습니다. 맑은 날씨가 예상됩니다.',
                                    f'오늘 눈이 내릴 예정입니다. 미끄럼에 주의하세요.')
            else:
                resp = resp.replace('강설 예보는 없습니다. 맑은 날씨가 예상됩니다.',
                                    f'강설 예보는 없습니다. 현재 {cond}입니다.')

        if '천둥·번개 예보는 없습니다' in resp:
            if code in THUNDER_CODES:
                resp = resp.replace('천둥·번개 예보는 없습니다.',
                                    '천둥번개 예보가 있습니다. 외출을 자제하세요.')

        # 야외 활동 — 날씨 기반
        if '야외 활동하기 적합합니다' in resp:
            if code in RAIN_CODES or rain >= 50:
                resp = resp.replace('날씨는 맑고 야외 활동하기 적합합니다.',
                                    f'현재 {cond}로 야외 활동이 어려울 수 있습니다. 우산을 준비하세요.')
            else:
                resp = resp.replace('날씨는 맑고 야외 활동하기 적합합니다.',
                                    f'현재 {cond}, {t}도로 야외 활동하기 좋습니다.')

        # 자외선
        if '자외선 지수' in resp:
            resp = re.sub(r'자외선 지수는 높음 수준', f'자외선 지수는 {w["uv_grade"]} 수준', resp)
            if w['uv_grade'] in ('낮음', '보통'):
                resp = resp.replace('외출 시 자외선 차단제를 바르세요.',
                                    '자외선 차단제는 선택 사항입니다.')

        # 일출/일몰
        if w['sunrise']:
            resp = re.sub(r'일출 시각은 오전 \d+시 \d+분', f'일출 시각은 {w["sunrise"]}', resp)
        if w['sunset']:
            resp = re.sub(r'일몰 시각은 오후 \d+시 \d+분', f'일몰 시각은 {w["sunset"]}', resp)

        # 습도
        resp = re.sub(r'습도는\s*OO%', f'습도는 {hum}%', resp)
        resp = re.sub(r'실내 습도는\s*OO%', f'외부 습도는 {hum}%', resp)

        # 어제보다 00도 — 근사치 대입 (오늘 최저-최고 중간)
        if '어제보다 00도' in resp and w['max_temp']:
            diff = round(w['max_temp'] - t, 1)
            sign = '+' if diff >= 0 else ''
            resp = resp.replace('어제보다 00도 높습니다', f'최고 {w["max_temp"]}도 예상됩니다')

        # 황사 — PM10 기반
        if '황사 농도는 보통 수준' in resp and a:
            resp = re.sub(r'황사 농도는 보통 수준', f'미세먼지는 {a["pm10_grade"]} 수준', resp)

    if a:
        g10, g25 = a['pm10_grade'], a['pm25_grade']
        pm10, pm25 = a['pm10'], a['pm25']

        # 르엘 슬롯: 나쁨/보통/좋음 → 하나 선택 (PM10 기준)
        resp = re.sub(r'나쁨/보통/좋음', g10, resp)

        # 미세먼지 등급 — 정적 텍스트 교체
        for grade in ('좋음', '보통', '나쁨', '매우나쁨'):
            resp = resp.replace(f'미세먼지는 {grade} 수준으로',
                                f'미세먼지(PM10 {pm10}㎍/m³)는 {g10} 수준으로')
            resp = resp.replace(f'미세먼지 {grade} 수준으로',
                                f'미세먼지(PM10 {pm10}㎍/m³) {g10} 수준으로')

        resp = resp.replace('초미세먼지가 나쁨 수준으로',
                            f'초미세먼지(PM2.5 {pm25}㎍/m³)가 {g25} 수준으로')
        resp = resp.replace('미세먼지는 보통 수준입니다.',
                            f'미세먼지(PM10) {pm10}㎍/m³ {g10}, 초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다.')

        # 환기 권장 문 동적 교체
        if '창문 개방은 권장하지 않습니다' in resp:
            worst = g10 if g10 in ('나쁨', '매우나쁨') else g25
            if worst not in ('나쁨', '매우나쁨'):
                resp = resp.replace('창문 개방은 권장하지 않습니다.',
                                    '환기하기 좋은 날씨입니다.')
        if '창문을 열어 환기하기 적절합니다' in resp:
            worst = g10 if g10 in ('나쁨', '매우나쁨') else g25
            if worst in ('나쁨', '매우나쁨'):
                resp = resp.replace('창문을 열어 환기하기 적절합니다.',
                                    '대기질이 나빠 환기를 자제하세요.')

        # 마스크
        if '마스크 착용을 권장합니다' in resp:
            worst = g10 if g10 in ('나쁨', '매우나쁨') else g25
            if worst not in ('나쁨', '매우나쁨'):
                resp = resp.replace('마스크 착용을 권장합니다.',
                                    f'현재 대기질 {g10}으로 마스크 없이 괜찮습니다.')

    return resp


# ─────────────────────────────────────────────────
# 날씨/대기질 → 자연어 응답 (판단형 / 일반)
# ─────────────────────────────────────────────────
def weather_response(raw_text, w):
    """판단형 쿼리 응답. None이면 fill_placeholders로 대체."""
    if w is None:
        return None
    t, cond, hum, rain = w['temp'], w['condition'], w['humidity'], w['rain_prob']
    code = w['code']

    if re.search(r'빨래|세탁', raw_text):
        ok = code not in RAIN_CODES and rain < 20 and hum < 70
        return (f'현재 {t}도, {cond}이고 습도 {hum}%입니다. '
                + ('빨래 널기에 좋은 날씨입니다.' if ok else '빨래 건조가 어려울 수 있습니다.'))
    if re.search(r'운동|조깅|달리기|자전거|드라이브|나들이', raw_text):
        ok = code not in RAIN_CODES and rain < 30
        return (f'현재 {t}도, {cond}입니다. '
                + ('활동하기 좋은 날씨입니다.' if ok else '비 예보가 있어 실내 활동을 권장합니다.'))
    if re.search(r'나가|외출|산책', raw_text):
        ok = rain < 40
        return (f'현재 {t}도, {cond}입니다. '
                + ('외출하기 괜찮습니다.' if ok else f'강수 확률 {rain}%로 우산을 준비하세요.'))
    if re.search(r'우산|비\s*(올까|와|맞)', raw_text):
        return (f'강수 확률 {rain}%입니다. '
                + ('우산을 챙기세요.' if rain >= 40 else '우산은 필요 없을 것 같습니다.'))
    if re.search(r'반팔|옷|뭐\s*입', raw_text):
        if t >= 25:   tip = '반팔이 적당합니다.'
        elif t >= 18: tip = '가벼운 긴팔이 좋습니다.'
        elif t >= 10: tip = '얇은 겉옷을 챙기세요.'
        else:          tip = '따뜻하게 입고 나가세요.'
        return f'현재 {t}도, {cond}입니다. {tip}'
    if re.search(r'눈\s*(올까|와|내려)', raw_text):
        snow = code in SNOW_CODES
        return (f'현재 날씨 코드상 {"눈이 내리고 있습니다." if snow else "강설 예보는 없습니다."} '
                f'현재 {t}도입니다.')
    if re.search(r'세차', raw_text):
        ok = code not in RAIN_CODES and rain < 20
        return ('세차하기 좋은 날씨입니다.' if ok else f'강수 확률 {rain}%로 세차를 미루는 것이 좋습니다.')
    if re.search(r'이불|난방.*필요|히터', raw_text):
        mt = w['min_temp']
        if mt is not None:
            return (f'오늘 최저 기온 {mt}도 예상됩니다. '
                    + ('이불을 두껍게 준비하세요.' if mt < 10 else '가벼운 이불로 충분합니다.'))

    # 일반 날씨 쿼리 — 포괄 응답
    min_t = f', 최저 {w["min_temp"]}도' if w['min_temp'] is not None else ''
    return (f'현재 {DISTRICT} {t}도, {cond}입니다. '
            f'체감 {w["feels"]}도, 습도 {hum}%, 강수 확률 {rain}%{min_t}.')


def air_response(raw_text, a):
    if a is None:
        return None
    g10, g25 = a['pm10_grade'], a['pm25_grade']
    pm10, pm25 = a['pm10'], a['pm25']
    worst = g10 if g10 in ('나쁨', '매우나쁨') else g25

    if re.search(r'환기', raw_text):
        ok = worst in ('좋음', '보통')
        return (f'현재 미세먼지(PM10) {pm10}㎍/m³ {g10}, 초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다. '
                + ('환기하기 좋습니다.' if ok else '대기질이 나빠 환기를 자제하세요.'))
    if re.search(r'마스크', raw_text):
        need = worst in ('나쁨', '매우나쁨')
        return (f'현재 미세먼지 {g10}, 초미세먼지 {g25}입니다. '
                + ('마스크 착용을 권장합니다.' if need else '마스크 없이 괜찮습니다.'))

    return (f'현재 {DISTRICT} 미세먼지(PM10) {pm10}㎍/m³ {g10}, '
            f'초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다.')


def news_response(n):
    if not n:
        return None
    items = '\n'.join(f'  {i+1}. {t}' for i, t in enumerate(n))
    return f'주요 뉴스 {len(n)}건입니다.\n{items}'


# ─────────────────────────────────────────────────
# 디바이스 상태 추적
# ─────────────────────────────────────────────────
QUERY_FNS = {'unknown', 'home_info', 'weather_query', 'traffic_query',
             'news_query', 'market_query', 'energy_query', 'system_meta',
             'medical_query', 'security_mode'}


class DeviceState:
    def __init__(self):
        self._s = {}

    def update(self, fn, room, direction, value=None):
        if direction == 'on':              self._s[(fn, room)] = 'on'
        elif direction == 'off':           self._s[(fn, room)] = 'off'
        elif direction == 'set' and value: self._s[(fn, room)] = f'{value[0]}={value[1]}'
        elif direction == 'open':          self._s[(fn, room)] = 'open'
        elif direction == 'close':         self._s[(fn, room)] = 'closed'

    def summary(self):
        if not self._s:
            return '(없음)'
        return ', '.join(f'{fn}/{room}: {st}' for (fn, room), st in self._s.items())


# ─────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args()

    from deployment_pipeline_v2 import DeploymentPipelineV2

    print('모델 로딩...')
    p  = DeploymentPipelineV2()
    ds = DeviceState()
    print('준비 완료. (날씨·대기질: Open-Meteo / 뉴스: 경향신문 RSS — 키 없음)')
    print(GRAY + '종료: q  |  디바이스 상태: d  |  DST 초기화: Enter×2' + RESET)
    print()

    # 60초 캐시
    cache = {'w': {'t': 0, 'data': None}, 'a': {'t': 0, 'data': None}}

    def _w():
        if time.time() - cache['w']['t'] > 60:
            cache['w']['data'] = get_weather()
            cache['w']['t'] = time.time()
        return cache['w']['data']

    def _a():
        if time.time() - cache['a']['t'] > 60:
            cache['a']['data'] = get_air()
            cache['a']['t'] = time.time()
        return cache['a']['data']

    prev_empty = False
    while True:
        try:
            line = input(CYAN + '> ' + RESET).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if line.lower() in ('q', 'quit'):
            break
        if line == 'd':
            print(GRAY + '  디바이스: ' + ds.summary() + RESET + '\n')
            continue
        if not line:
            if prev_empty:
                p.reset_dst()
                print(GRAY + '  [DST 초기화]' + RESET)
                prev_empty = False
            else:
                prev_empty = True
            continue
        prev_empty = False

        r    = p.process(line, use_dst=True)
        fn   = r['fn']
        dirn = r['param_direction']
        room = r['room']
        val  = r['value']
        resp = r['response']

        if fn not in QUERY_FNS:
            ds.update(fn, room, dirn, val)

        # API 응답 조합
        api_resp = None
        if fn == 'weather_query':
            w, a = _w(), _a()
            if re.search(r'미세먼지|공기|대기|환기|마스크', line):
                api_resp = air_response(line, a)
                if api_resp is None:
                    api_resp = fill_placeholders(resp, w, a)
            else:
                api_resp = weather_response(line, w)
                if api_resp is None:
                    # 판단형 미매칭 → OO placeholder 치환
                    api_resp = fill_placeholders(resp, w, a)
        elif fn == 'news_query':
            api_resp = news_response(get_news())
        else:
            # 날씨 외 응답에도 OO가 있으면 치환 시도
            filled = fill_placeholders(resp, _w() if 'OO' in resp or '00도' in resp else None,
                                       _a() if '미세먼지' in resp else None)
            if filled != resp:
                api_resp = filled

        final_resp = api_resp if api_resp else resp

        # 출력
        parts = [CYAN + fn + RESET]
        if dirn != 'none': parts.append(YELLOW + dirn + RESET)
        if room != 'none': parts.append(GREEN + room + RESET)
        if val:            parts.append(f'{val[0]}={val[1]}')
        if api_resp:       parts.append(BOLD + '[API]' + RESET)
        print('  ' + ' | '.join(parts))
        if args.verbose:
            print(GRAY + f'    exec={r["exec_type"]}  dst={r["dst_applied"]}  rooms={r["rooms"]}' + RESET)
        print(BOLD + f'  → {final_resp}' + RESET)
        print()


if __name__ == '__main__':
    main()
