#!/usr/bin/env python3
"""NLU 대화형 테스트 — 발화 입력 → 분류 결과 + 응답 (실제 API 포함).

날씨/대기질: Open-Meteo (무료, 키 없음)
뉴스: 경향신문 RSS (무료, 키 없음)

사용:
    python3 test_interactive.py          # 대화형
    python3 test_interactive.py -v       # verbose (exec/rooms/dst)
    python3 test_interactive.py -c       # 복합 발화 split
    echo "거실 불 켜줘" | python3 test_interactive.py   # 배치
"""
import os, sys, re, json, time, urllib.request, argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from deployment_pipeline_v2 import DeploymentPipelineV2

RESET  = '\033[0m'; BOLD = '\033[1m'
GREEN  = '\033[92m'; YELLOW = '\033[93m'
CYAN   = '\033[96m'; GRAY = '\033[90m'
RED    = '\033[91m'

LAT, LON = 37.4837, 127.0324
DISTRICT = '서초구'
DAYS_KR  = ['월', '화', '수', '목', '금', '토', '일']

# ─────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────
WMO_KR = {
    0: '맑음', 1: '대체로 맑음', 2: '구름 많음', 3: '흐림',
    45: '안개', 48: '안개',
    51: '이슬비', 53: '이슬비', 55: '이슬비',
    61: '비', 63: '비', 65: '강한 비',
    71: '눈', 73: '눈', 75: '강한 눈',
    80: '소나기', 81: '소나기', 82: '강한 소나기',
    95: '천둥번개', 96: '천둥번개', 99: '천둥번개',
}
RAIN_CODES   = {61, 63, 65, 80, 81, 82, 95, 96, 99, 51, 53, 55}
SNOW_CODES   = {71, 73, 75, 85, 86}
THUNDER_CODES = {95, 96, 99}


def _get_json(url, timeout=10):
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
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


def _uv_grade(uv):
    if uv is None: return '보통'
    if uv < 3:  return '낮음'
    if uv < 6:  return '보통'
    if uv < 8:  return '높음'
    if uv < 11: return '매우높음'
    return '위험'


def _pm_grade(pm10, pm25):
    g10 = '좋음' if pm10 <= 30 else '보통' if pm10 <= 80 else '나쁨' if pm10 <= 150 else '매우나쁨'
    g25 = '좋음' if pm25 <= 15 else '보통' if pm25 <= 35 else '나쁨' if pm25 <= 75 else '매우나쁨'
    return g10, g25


def get_weather():
    d = _get_json(
        f'https://api.open-meteo.com/v1/forecast'
        f'?latitude={LAT}&longitude={LON}'
        f'&current=temperature_2m,apparent_temperature,relative_humidity_2m,'
        f'wind_speed_10m,precipitation,weather_code'
        f'&daily=temperature_2m_min,temperature_2m_max,'
        f'precipitation_probability_max,uv_index_max,sunrise,sunset'
        f'&timezone=Asia%2FSeoul'
    )
    if not d or 'current' not in d:
        return None
    c     = d['current']
    daily = d.get('daily', {})
    code  = c.get('weather_code', 0)
    prec  = c.get('precipitation', 0) or 0
    rain_prob = daily.get('precipitation_probability_max', [None])[0]
    if rain_prob is None:
        rain_prob = 85 if prec > 0 else (35 if code in range(1, 4) else 5)

    def _hhmm(s):
        m = re.search(r'T(\d+):(\d+)', s or '')
        if not m: return None
        h, mn = int(m.group(1)), int(m.group(2))
        return f'{"오전" if h < 12 else "오후"} {h if h <= 12 else h-12}시 {mn:02d}분'

    uv_raw  = (daily.get('uv_index_max') or [None])[0]
    min_t   = (daily.get('temperature_2m_min') or [None])[0]
    max_t   = (daily.get('temperature_2m_max') or [None])[0]
    return {
        'temp': round(c.get('temperature_2m', 0), 1),
        'feels': round(c.get('apparent_temperature', 0), 1),
        'humidity': c.get('relative_humidity_2m', 0),
        'wind': round(c.get('wind_speed_10m', 0), 1),
        'condition': WMO_KR.get(code, '흐림'),
        'code': code,
        'rain_prob': int(rain_prob),
        'prec': prec,
        'min_temp': round(min_t, 1) if min_t is not None else None,
        'max_temp': round(max_t, 1) if max_t is not None else None,
        'uv': uv_raw,
        'uv_grade': _uv_grade(uv_raw),
        'sunrise': _hhmm((daily.get('sunrise') or [''])[0]),
        'sunset':  _hhmm((daily.get('sunset')  or [''])[0]),
    }


def get_air():
    d = _get_json(
        f'https://air-quality-api.open-meteo.com/v1/air-quality'
        f'?latitude={LAT}&longitude={LON}&current=pm10,pm2_5'
    )
    if not d or 'current' not in d:
        return None
    c = d['current']
    pm10, pm25 = c.get('pm10') or 0, c.get('pm2_5') or 0
    g10, g25 = _pm_grade(pm10, pm25)
    return {'pm10': round(pm10, 1), 'pm25': round(pm25, 1), 'pm10_grade': g10, 'pm25_grade': g25}


def get_news(n=3):
    xml    = _get_raw('https://www.khan.co.kr/rss/rssdata/total_news.xml')
    titles = re.findall(r'<title><!\[CDATA\[(.+?)\]\]></title>', xml)
    if not titles:
        titles = re.findall(r'<title>([^<]{5,})</title>', xml)
    titles = [t.strip() for t in titles if len(t.strip()) > 5][:n]
    return titles or None


def fill_placeholders(resp, w, a):
    if not resp:
        return resp
    now = datetime.now()
    resp = re.sub(r'N월\s*M일\s*OO요일',
                  f'{now.month}월 {now.day}일 {DAYS_KR[now.weekday()]}요일', resp)
    resp = re.sub(r'OO구\s*OO동', DISTRICT, resp)
    if w:
        t, feels, hum, rain, cond, code = (
            w['temp'], w['feels'], w['humidity'], w['rain_prob'], w['condition'], w['code'])
        resp = re.sub(r'최저 기온은\s*00도', f'최저 기온은 {w["min_temp"]}도', resp)
        resp = re.sub(r'최저\s*00도', f'최저 {w["min_temp"]}도', resp)
        resp = re.sub(r'최고 기온은\s*00도', f'최고 기온은 {w["max_temp"]}도', resp)
        resp = re.sub(r'기온은\s*(?:OO|00)도', f'기온은 {t}도', resp)
        resp = re.sub(r'체감온도는\s*OO도', f'체감온도는 {feels}도', resp)
        resp = re.sub(r'강수 확률은\s*(?:0|OO)%', f'강수 확률은 {rain}%', resp)
        resp = re.sub(r'습도는\s*OO%', f'습도는 {hum}%', resp)
        resp = re.sub(r'실내 습도는\s*OO%', f'외부 습도는 {hum}%', resp)
        if '강설 예보는 없습니다' in resp:
            resp = resp.replace('강설 예보는 없습니다. 맑은 날씨가 예상됩니다.',
                                f'오늘 눈이 내릴 예정입니다.' if code in SNOW_CODES
                                else f'강설 예보는 없습니다. 현재 {cond}입니다.')
        if '천둥·번개 예보는 없습니다' in resp and code in THUNDER_CODES:
            resp = resp.replace('천둥·번개 예보는 없습니다.', '천둥번개 예보가 있습니다. 외출을 자제하세요.')
        if '야외 활동하기 적합합니다' in resp:
            resp = resp.replace('날씨는 맑고 야외 활동하기 적합합니다.',
                                f'현재 {cond}로 야외 활동이 어려울 수 있습니다. 우산을 준비하세요.'
                                if code in RAIN_CODES or rain >= 50
                                else f'현재 {cond}, {t}도로 야외 활동하기 좋습니다.')
        if '자외선 지수' in resp:
            resp = re.sub(r'자외선 지수는 높음 수준', f'자외선 지수는 {w["uv_grade"]} 수준', resp)
            if w['uv_grade'] in ('낮음', '보통'):
                resp = resp.replace('외출 시 자외선 차단제를 바르세요.', '자외선 차단제는 선택 사항입니다.')
        if w['sunrise']:
            resp = re.sub(r'일출 시각은 오전 \d+시 \d+분', f'일출 시각은 {w["sunrise"]}', resp)
        if w['sunset']:
            resp = re.sub(r'일몰 시각은 오후 \d+시 \d+분', f'일몰 시각은 {w["sunset"]}', resp)
        if '어제보다 00도' in resp and w['max_temp']:
            resp = resp.replace('어제보다 00도 높습니다', f'최고 {w["max_temp"]}도 예상됩니다')
        if '황사 농도는 보통 수준' in resp and a:
            resp = re.sub(r'황사 농도는 보통 수준', f'미세먼지는 {a["pm10_grade"]} 수준', resp)
    if a:
        g10, g25, pm10, pm25 = a['pm10_grade'], a['pm25_grade'], a['pm10'], a['pm25']
        for grade in ('좋음', '보통', '나쁨', '매우나쁨'):
            resp = resp.replace(f'미세먼지는 {grade} 수준으로',
                                f'미세먼지(PM10 {pm10}㎍/m³)는 {g10} 수준으로')
            resp = resp.replace(f'미세먼지 {grade} 수준으로',
                                f'미세먼지(PM10 {pm10}㎍/m³) {g10} 수준으로')
        resp = resp.replace('초미세먼지가 나쁨 수준으로',
                            f'초미세먼지(PM2.5 {pm25}㎍/m³)가 {g25} 수준으로')
        resp = resp.replace('미세먼지는 보통 수준입니다.',
                            f'미세먼지(PM10) {pm10}㎍/m³ {g10}, 초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다.')
        worst = g10 if g10 in ('나쁨', '매우나쁨') else g25
        if '창문 개방은 권장하지 않습니다' in resp and worst not in ('나쁨', '매우나쁨'):
            resp = resp.replace('창문 개방은 권장하지 않습니다.', '환기하기 좋은 날씨입니다.')
        if '창문을 열어 환기하기 적절합니다' in resp and worst in ('나쁨', '매우나쁨'):
            resp = resp.replace('창문을 열어 환기하기 적절합니다.', '대기질이 나빠 환기를 자제하세요.')
        if '마스크 착용을 권장합니다' in resp and worst not in ('나쁨', '매우나쁨'):
            resp = resp.replace('마스크 착용을 권장합니다.',
                                f'현재 대기질 {g10}으로 마스크 없이 괜찮습니다.')
    return resp


def weather_response(raw_text, w):
    if w is None:
        return None
    t, cond, hum, rain, code = w['temp'], w['condition'], w['humidity'], w['rain_prob'], w['code']
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
        tip = ('반팔이 적당합니다.' if t >= 25 else '가벼운 긴팔이 좋습니다.' if t >= 18
               else '얇은 겉옷을 챙기세요.' if t >= 10 else '따뜻하게 입고 나가세요.')
        return f'현재 {t}도, {cond}입니다. {tip}'
    if re.search(r'눈\s*(올까|와|내려)', raw_text):
        return (f'{"오늘 눈이 내리고 있습니다." if code in SNOW_CODES else "강설 예보는 없습니다."} 현재 {t}도입니다.')
    if re.search(r'세차', raw_text):
        ok = code not in RAIN_CODES and rain < 20
        return ('세차하기 좋은 날씨입니다.' if ok else f'강수 확률 {rain}%로 세차를 미루는 것이 좋습니다.')
    if re.search(r'이불|난방.*필요|히터', raw_text):
        mt = w['min_temp']
        if mt is not None:
            return (f'오늘 최저 기온 {mt}도 예상됩니다. '
                    + ('이불을 두껍게 준비하세요.' if mt < 10 else '가벼운 이불로 충분합니다.'))
    min_t = f', 최저 {w["min_temp"]}도' if w['min_temp'] is not None else ''
    return f'현재 {DISTRICT} {t}도, {cond}입니다. 체감 {w["feels"]}도, 습도 {hum}%, 강수 확률 {rain}%{min_t}.'


def air_response(raw_text, a):
    if a is None:
        return None
    g10, g25, pm10, pm25 = a['pm10_grade'], a['pm25_grade'], a['pm10'], a['pm25']
    worst = g10 if g10 in ('나쁨', '매우나쁨') else g25
    if re.search(r'환기', raw_text):
        return (f'현재 미세먼지(PM10) {pm10}㎍/m³ {g10}, 초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다. '
                + ('환기하기 좋습니다.' if worst in ('좋음', '보통') else '대기질이 나빠 환기를 자제하세요.'))
    if re.search(r'마스크', raw_text):
        return (f'현재 미세먼지 {g10}, 초미세먼지 {g25}입니다. '
                + ('마스크 착용을 권장합니다.' if worst in ('나쁨', '매우나쁨') else '마스크 없이 괜찮습니다.'))
    return f'현재 {DISTRICT} 미세먼지(PM10) {pm10}㎍/m³ {g10}, 초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다.'


def news_response(n):
    if not n:
        return None
    return '주요 뉴스 {}건입니다.\n{}'.format(len(n), '\n'.join(f'  {i+1}. {t}' for i, t in enumerate(n)))


# ─────────────────────────────────────────────────
# 디바이스 상태
# ─────────────────────────────────────────────────
QUERY_FNS = {'unknown', 'home_info', 'weather_query', 'traffic_query',
             'news_query', 'market_query', 'energy_query', 'system_meta',
             'medical_query', 'security_mode'}

HVAC_CONFLICT  = {'heat_control': 'ac_control', 'ac_control': 'heat_control'}
HVAC_DEVICE_KR = {'ac_control': '에어컨을', 'heat_control': '난방을'}

# 명시적 냉난방 키워드 (있으면 explicit 명령, 없으면 간접 표현)
EXPLICIT_HVAC_KW = {
    'heat_control': {'난방', '히터', '보일러', '온풍기', '온풍'},
    'ac_control':   {'에어컨', '냉방', '냉풍'},
}
ROOM_KR = {
    'living': '거실', 'kitchen': '주방', 'bedroom_main': '안방',
    'bedroom_sub': '작은방', 'all': '전체', 'none': '',
}
FN_KR = {
    'ac_control': '에어컨', 'heat_control': '난방', 'light_control': '조명',
    'vent_control': '환기', 'gas_control': '가스', 'door_control': '도어',
    'window_control': '창문', 'curtain_control': '커튼', 'schedule_manage': '일정',
}
ST_KR = {'on': '켜짐', 'off': '꺼짐', 'open': '열림', 'closed': '닫힘'}


def _is_explicit_hvac(text, fn):
    return any(kw in text for kw in EXPLICIT_HVAC_KW.get(fn, set()))


class DeviceState:
    def __init__(self):
        self._s = {}

    def update(self, fn, room, direction, value=None):
        if direction == 'on':              self._s[(fn, room)] = 'on'
        elif direction == 'off':           self._s[(fn, room)] = 'off'
        elif direction == 'set' and value: self._s[(fn, room)] = f'{value[0]}={value[1]}'
        elif direction == 'open':          self._s[(fn, room)] = 'open'
        elif direction == 'close':         self._s[(fn, room)] = 'closed'

    def resolve_hvac_conflict(self, fn, room, direction):
        """냉·난방 충돌 감지·해소. 반환: '에어컨을'/'난방을' or None."""
        if direction != 'on' or fn not in HVAC_CONFLICT:
            return None
        rival = HVAC_CONFLICT[fn]
        if room in ('none', 'all'):
            rooms_on = [r for (f, r), st in self._s.items() if f == rival and st == 'on']
        else:
            rooms_on = [room] if self._s.get((rival, room)) == 'on' else []
        if not rooms_on:
            return None
        for r in rooms_on:
            self._s[(rival, r)] = 'off'
        return HVAC_DEVICE_KR[rival]

    def pretty(self):
        if not self._s:
            return '없음'
        parts = []
        for (fn, room), st in self._s.items():
            fn_kr = FN_KR.get(fn, fn)
            room_kr = ROOM_KR.get(room, '')
            st_kr = ST_KR.get(st, st)
            label = f'{fn_kr}({room_kr})' if room_kr else fn_kr
            parts.append(f'{label} {st_kr}')
        return ' · '.join(parts)


# ─────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────
def fmt_fn(fn):
    return (RED if fn == 'unknown' else CYAN) + fn + RESET


def print_result(r, final_resp, verbose=False, is_api=False):
    fn, dirn, room, val = r['fn'], r['param_direction'], r['room'], r['value']
    parts = [fmt_fn(fn)]
    if dirn != 'none': parts.append(YELLOW + dirn + RESET)
    if room != 'none': parts.append(GREEN + room + RESET)
    if val:            parts.append(f'{val[0]}={val[1]}')
    if r.get('dst_applied'): parts.append(GRAY + '[DST]' + RESET)
    if is_api:         parts.append(BOLD + '[API]' + RESET)
    print('  ' + ' | '.join(parts))
    if verbose:
        print(GRAY + f'    exec={r["exec_type"]}  rooms={r.get("rooms",[])}' + RESET)
        if r.get('preprocessed') != r.get('raw'):
            print(GRAY + f'    preproc: {r["preprocessed"]}' + RESET)
    print(BOLD + f'  → {final_resp}' + RESET)


# ─────────────────────────────────────────────────
# API 캐시
# ─────────────────────────────────────────────────
class ApiCache:
    def __init__(self, ttl=60):
        self._w = {'t': 0, 'data': None}
        self._a = {'t': 0, 'data': None}
        self._ttl = ttl

    def weather(self):
        if time.time() - self._w['t'] > self._ttl:
            self._w['data'] = get_weather()
            self._w['t'] = time.time()
        return self._w['data']

    def air(self):
        if time.time() - self._a['t'] > self._ttl:
            self._a['data'] = get_air()
            self._a['t'] = time.time()
        return self._a['data']


# ─────────────────────────────────────────────────
# 처리 (NLU + API)
# ─────────────────────────────────────────────────
def process_with_api(pipeline, api, line, use_dst=True, compound=False):
    """단일 발화 또는 복합 발화를 처리해 (result, final_resp, is_api) 리스트 반환."""
    if compound:
        res = pipeline.process_compound(line, use_dst=use_dst)
        results = res['actions']
        is_compound = res['is_compound']
    else:
        results = [pipeline.process(line, use_dst=use_dst)]
        is_compound = False

    out = []
    for r in results:
        fn, dirn = r['fn'], r['param_direction']
        resp     = r['response']
        raw_text = r.get('preprocessed', line)

        api_resp = None
        if fn == 'weather_query':
            w, a = api.weather(), api.air()
            if re.search(r'미세먼지|공기|대기|환기|마스크', raw_text):
                api_resp = air_response(raw_text, a) or fill_placeholders(resp, w, a)
            else:
                api_resp = weather_response(raw_text, w) or fill_placeholders(resp, w, a)
        elif fn == 'news_query':
            api_resp = news_response(get_news())
        else:
            has_oo = 'OO' in resp or '00도' in resp
            filled = fill_placeholders(
                resp,
                api.weather() if has_oo else None,
                api.air() if '미세먼지' in resp else None,
            )
            if filled != resp:
                api_resp = filled

        final = api_resp if api_resp else resp
        out.append((r, final, api_resp is not None, is_compound))
    return out


# ─────────────────────────────────────────────────
# 대화형 실행
# ─────────────────────────────────────────────────
def run_interactive(pipeline, api, ds, verbose=False, compound=False):
    print(BOLD + 'NLU 테스트 (종료: q  |  디바이스 상태: d  |  DST 초기화: Enter×2)' + RESET)
    print(GRAY + '날씨·대기질: Open-Meteo  /  뉴스: 경향신문 RSS  — 키 없음' + RESET)
    print()
    prev_empty = False
    while True:
        try:
            line = input(CYAN + '> ' + RESET).strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if line.lower() in ('q', 'quit', 'exit'):
            break
        if line == 'd':
            print(GRAY + '  [집 상태] ' + ds.pretty() + RESET + '\n')
            continue
        if not line:
            if prev_empty:
                pipeline.reset_dst()
                print(GRAY + '  [DST 초기화]' + RESET)
                prev_empty = False
            else:
                prev_empty = True
            continue
        prev_empty = False

        items = process_with_api(pipeline, api, line, use_dst=True, compound=compound)
        if len(items) > 1:
            print(GRAY + f'  복합 발화 ({len(items)}개)' + RESET)
        state_changed = False
        for r, final, is_api, _ in items:
            if r['fn'] not in QUERY_FNS:
                conflict = ds.resolve_hvac_conflict(r['fn'], r['room'], r['param_direction'])
                if conflict:
                    if _is_explicit_hvac(line, r['fn']):
                        ds.update(r['fn'], r['room'], r['param_direction'], r['value'])
                        cleaned = re.sub(r'^네,?\s*', '', final)
                        final = f'{conflict} 끄고 {cleaned}'
                    else:
                        room_kr = ROOM_KR.get(r['room'], '')
                        prefix = f'{room_kr} ' if room_kr else ''
                        final = f'{prefix}{conflict} 꺼드릴게요.'
                else:
                    ds.update(r['fn'], r['room'], r['param_direction'], r['value'])
                state_changed = True
            print_result(r, final, verbose=verbose, is_api=is_api)
        state_str = ds.pretty()
        if state_changed and state_str != '없음':
            print(GRAY + f'  [집 상태] {state_str}' + RESET)
        print()


# ─────────────────────────────────────────────────
# 배치 실행 (파이프/리다이렉트)
# ─────────────────────────────────────────────────
def run_batch(pipeline, api, lines, verbose=False, compound=False):
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        print(BOLD + f'"{line}"' + RESET)
        items = process_with_api(pipeline, api, line, use_dst=False, compound=compound)
        for r, final, is_api, _ in items:
            print_result(r, final, verbose=verbose, is_api=is_api)
        print()


# ─────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='NLU 대화형/배치 테스트')
    ap.add_argument('-v', '--verbose', action='store_true', help='exec/rooms/dst 출력')
    ap.add_argument('-c', '--compound', action='store_true', help='복합 발화 split')
    args = ap.parse_args()

    print('모델 로딩...')
    pipeline = DeploymentPipelineV2()
    api      = ApiCache(ttl=60)
    ds       = DeviceState()
    print('준비 완료.\n')

    if sys.stdin.isatty():
        run_interactive(pipeline, api, ds, verbose=args.verbose, compound=args.compound)
    else:
        run_batch(pipeline, api, sys.stdin.readlines(), verbose=args.verbose, compound=args.compound)


if __name__ == '__main__':
    main()
