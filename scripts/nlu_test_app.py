#!/usr/bin/env python3
"""NLU 테스트 웹앱 — Streamlit
실행: streamlit run scripts/nlu_test_app.py --server.port 8501

test_api.py의 실API(날씨/미세먼지/뉴스) 기능 포함.
"""
import sys, os, re, time, json, urllib.request
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
st.set_page_config(page_title="NLU 테스트", layout="wide", page_icon="🏠")

# ── API 함수 (test_api.py에서 가져옴) ────────────────────────
LAT, LON = 37.4837, 127.0324
DISTRICT = '서초구'

WMO_KR = {
    0:'맑음', 1:'대체로 맑음', 2:'구름 많음', 3:'흐림',
    45:'안개', 48:'안개', 51:'이슬비', 53:'이슬비', 55:'이슬비',
    61:'비', 63:'비', 65:'강한 비', 71:'눈', 73:'눈', 75:'강한 눈',
    80:'소나기', 81:'소나기', 82:'강한 소나기', 95:'천둥번개', 96:'천둥번개', 99:'천둥번개',
}
RAIN_CODES = {61,63,65,80,81,82,95,96,99,51,53,55}
SNOW_CODES = {71,73,75,85,86}

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

def _pm_grade(v):
    if v <= 30: return '좋음'
    if v <= 80: return '보통'
    if v <= 150: return '나쁨'
    return '매우나쁨'

def _pm25_grade(v):
    if v <= 15: return '좋음'
    if v <= 35: return '보통'
    if v <= 75: return '나쁨'
    return '매우나쁨'

@st.cache_data(ttl=60)
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
    code = c.get('weather_code', 0)
    rain_prob = (daily.get('precipitation_probability_max') or [None])[0]
    if rain_prob is None:
        prec = c.get('precipitation', 0) or 0
        rain_prob = 85 if prec > 0 else 5
    min_t = (daily.get('temperature_2m_min') or [None])[0]
    max_t = (daily.get('temperature_2m_max') or [None])[0]
    return {
        'temp': round(c.get('temperature_2m', 0), 1),
        'feels': round(c.get('apparent_temperature', 0), 1),
        'humidity': c.get('relative_humidity_2m', 0),
        'wind': round(c.get('wind_speed_10m', 0), 1),
        'condition': WMO_KR.get(code, '흐림'),
        'code': code,
        'rain_prob': int(rain_prob),
        'min_temp': round(min_t, 1) if min_t is not None else None,
        'max_temp': round(max_t, 1) if max_t is not None else None,
    }

@st.cache_data(ttl=60)
def get_air():
    url = (f'https://air-quality-api.open-meteo.com/v1/air-quality'
           f'?latitude={LAT}&longitude={LON}&current=pm10,pm2_5')
    d = _get_json(url)
    if not d or 'current' not in d:
        return None
    c = d['current']
    pm10 = round(c.get('pm10') or 0, 1)
    pm25 = round(c.get('pm2_5') or 0, 1)
    return {'pm10': pm10, 'pm25': pm25,
            'pm10_grade': _pm_grade(pm10), 'pm25_grade': _pm25_grade(pm25)}

@st.cache_data(ttl=300)
def get_news(n=3):
    xml = _get_raw('https://www.khan.co.kr/rss/rssdata/total_news.xml')
    if not xml:
        return None
    titles = re.findall(r'<title><!\[CDATA\[(.+?)\]\]></title>', xml)
    if not titles:
        titles = re.findall(r'<title>([^<]{5,})</title>', xml)
    return [t.strip() for t in titles if len(t.strip()) > 5][:n] or None

def fill_placeholders(resp, w, a):
    """test_api.py의 fill_placeholders — OO/00 → 실데이터"""
    if not resp:
        return resp
    if w:
        t, feels, hum, rain = w['temp'], w['feels'], w['humidity'], w['rain_prob']
        cond, code = w['condition'], w['code']
        resp = re.sub(r'OO구\s*OO동', DISTRICT, resp)
        resp = re.sub(r'기온은\s*(?:OO|00)도', f'기온은 {t}도', resp)
        resp = re.sub(r'체감온도는\s*(?:OO|00)도', f'체감온도는 {feels}도', resp)
        resp = re.sub(r'강수 확률은\s*(?:OO|0)%', f'강수 확률은 {rain}%', resp)
        resp = re.sub(r'습도는\s*OO%', f'습도는 {hum}%', resp)
        if w['min_temp'] is not None:
            resp = re.sub(r'최저\s*(?:기온은\s*)?00도', f'최저 {w["min_temp"]}도', resp)
        if w['max_temp'] is not None:
            resp = re.sub(r'최고\s*(?:기온은\s*)?00도', f'최고 {w["max_temp"]}도', resp)
        if code in RAIN_CODES and rain >= 40:
            resp = resp.replace('강설 예보는 없습니다. 맑은 날씨가 예상됩니다.',
                                f'현재 {cond}입니다. 우산을 준비하세요.')
        if '야외 활동하기 적합합니다' in resp:
            if code in RAIN_CODES or rain >= 50:
                resp = resp.replace('날씨는 맑고 야외 활동하기 적합합니다.',
                                    f'현재 {cond}으로 야외 활동이 어려울 수 있습니다.')
            else:
                resp = resp.replace('날씨는 맑고 야외 활동하기 적합합니다.',
                                    f'현재 {cond}, {t}도로 야외 활동하기 좋습니다.')
    if a:
        g10, g25 = a['pm10_grade'], a['pm25_grade']
        pm10, pm25 = a['pm10'], a['pm25']
        for grade in ('좋음', '보통', '나쁨', '매우나쁨'):
            resp = resp.replace(f'미세먼지는 {grade} 수준으로',
                                f'미세먼지(PM10 {pm10}㎍/m³)는 {g10} 수준으로')
        resp = resp.replace('미세먼지는 보통 수준입니다.',
                            f'미세먼지(PM10) {pm10}㎍/m³ {g10}, 초미세먼지(PM2.5) {pm25}㎍/m³ {g25}입니다.')
        worst = g10 if g10 in ('나쁨','매우나쁨') else g25
        if '창문 개방은 권장하지 않습니다' in resp and worst not in ('나쁨','매우나쁨'):
            resp = resp.replace('창문 개방은 권장하지 않습니다.', '환기하기 좋은 날씨입니다.')
        if '마스크 착용을 권장합니다' in resp and worst not in ('나쁨','매우나쁨'):
            resp = resp.replace('마스크 착용을 권장합니다.', f'현재 대기질 {g10}으로 마스크 없이 괜찮습니다.')
    return resp

def weather_response(raw_text, w):
    if not w:
        return None
    t, cond, hum, rain, code = w['temp'], w['condition'], w['humidity'], w['rain_prob'], w['code']
    if re.search(r'빨래|세탁', raw_text):
        ok = code not in RAIN_CODES and rain < 20 and hum < 70
        return f'현재 {t}도, {cond}, 습도 {hum}%. ' + ('빨래 널기 좋습니다.' if ok else '빨래 건조가 어려울 수 있습니다.')
    if re.search(r'우산|비\s*(올까|와|맞)', raw_text):
        return f'강수 확률 {rain}%. ' + ('우산을 챙기세요.' if rain >= 40 else '우산은 필요 없을 것 같습니다.')
    if re.search(r'반팔|옷|뭐\s*입', raw_text):
        tip = ('반팔이 적당합니다.' if t >= 25 else '가벼운 긴팔이 좋습니다.' if t >= 18
               else '얇은 겉옷을 챙기세요.' if t >= 10 else '따뜻하게 입고 나가세요.')
        return f'현재 {t}도, {cond}. {tip}'
    if re.search(r'미세먼지|공기|대기|환기|마스크', raw_text):
        return None
    min_t = f', 최저 {w["min_temp"]}도' if w['min_temp'] is not None else ''
    max_t = f', 최고 {w["max_temp"]}도' if w['max_temp'] is not None else ''
    return f'현재 {DISTRICT} {t}도, {cond}. 체감 {w["feels"]}도, 습도 {hum}%, 강수 확률 {rain}%{min_t}{max_t}.'

def air_response(raw_text, a):
    if not a:
        return None
    g10, g25, pm10, pm25 = a['pm10_grade'], a['pm25_grade'], a['pm10'], a['pm25']
    worst = g10 if g10 in ('나쁨','매우나쁨') else g25
    base = f'미세먼지(PM10 {pm10}㎍/m³) {g10}, 초미세먼지(PM2.5 {pm25}㎍/m³) {g25}.'
    if re.search(r'환기', raw_text):
        return base + (' 환기하기 좋습니다.' if worst not in ('나쁨','매우나쁨') else ' 대기질이 나빠 환기를 자제하세요.')
    if re.search(r'마스크', raw_text):
        return base + (' 마스크 착용을 권장합니다.' if worst in ('나쁨','매우나쁨') else ' 마스크 없이 괜찮습니다.')
    return base

def make_api_response(fn, text, raw_resp):
    """fn에 따라 API 응답 생성 (날씨/미세먼지/뉴스)"""
    if fn == 'weather_query':
        w, a = get_weather(), get_air()
        if re.search(r'미세먼지|공기|대기|환기|마스크', text):
            return air_response(text, a) or fill_placeholders(raw_resp, w, a)
        api = weather_response(text, w)
        return api if api else fill_placeholders(raw_resp, w, a)
    elif fn == 'news_query':
        n = get_news()
        if n:
            return '주요 뉴스입니다.\n' + '\n'.join(f'{i+1}. {t}' for i, t in enumerate(n))
    # 날씨 외 응답에 OO 있으면 치환 시도
    if 'OO' in raw_resp or '00도' in raw_resp:
        w = get_weather() if ('OO' in raw_resp or '00도' in raw_resp) else None
        a = get_air() if '미세먼지' in raw_resp else None
        filled = fill_placeholders(raw_resp, w, a)
        if filled != raw_resp:
            return filled
    return None

# ── 파이프라인 로드 ───────────────────────────────────────────
@st.cache_resource
def _load_model():
    """모델/토크나이저만 캐시 (공유해도 무방)"""
    from deployment_pipeline_v2 import DeploymentPipelineV2
    return DeploymentPipelineV2()

def get_single_pipeline():
    if 'single_pipeline' not in st.session_state:
        from deployment_pipeline_v2 import DeploymentPipelineV2
        st.session_state.single_pipeline = DeploymentPipelineV2()
    return st.session_state.single_pipeline

def get_multi_pipeline():
    if 'multi_pipeline' not in st.session_state:
        from deployment_pipeline_v2 import DeploymentPipelineV2
        st.session_state.multi_pipeline = DeploymentPipelineV2()
    return st.session_state.multi_pipeline

# ── UI ────────────────────────────────────────────────────────
st.title("🏠 르엘 NLU 테스트")
st.caption(f"날씨/미세먼지: Open-Meteo (실시간) | 뉴스: 경향신문 RSS | 위치: {DISTRICT}")

col_left, col_right = st.columns([1, 1])

# ── 왼쪽: 단일 발화 ──────────────────────────────────────────
with col_left:
    st.subheader("📝 단일 발화 테스트")

    utterance = st.text_input("발화 입력", placeholder="예: 거실 불 켜줘 / 오늘 날씨 어때", key="single_input")
    use_dst = st.checkbox("DST 사용", value=False, key="dst_single")

    if st.button("분석", type="primary", key="btn_single"):
        if utterance.strip():
            p = get_single_pipeline()
            p.reset_dst()
            r = p.process(utterance.strip(), use_dst=use_dst)
            api_resp = make_api_response(r['fn'], r['preprocessed'], r['response'])
            final_resp = api_resp if api_resp else r['response']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("fn", r["fn"])
            col2.metric("exec", r["exec_type"])
            col3.metric("dir", r["param_direction"])
            col4.metric("room", r["room"])
            if api_resp:
                st.success(f"💬 **[API]** {final_resp}")
            else:
                st.info(f"💬 {final_resp}")

            with st.expander("상세"):
                st.json({"preprocessed": r["preprocessed"], "fn": r["fn"],
                         "exec_type": r["exec_type"], "param_direction": r["param_direction"],
                         "room": r["room"], "rooms": r["rooms"], "value": r["value"],
                         "dst_applied": r["dst_applied"], "api_used": bool(api_resp)})

    st.markdown("---")
    st.subheader("⚡ 복합 명령")
    compound = st.text_input("복합 발화", placeholder="예: 거실 불 끄고 에어컨 켜줘", key="compound_input")
    if st.button("분석", key="btn_compound") and compound.strip():
        p = get_single_pipeline()
        p.reset_dst()
        result = p.process_compound(compound.strip())
        st.write(f"복합 명령: {'✅' if result['is_compound'] else '❌'}")
        for i, action in enumerate(result['actions'], 1):
            api_resp = make_api_response(action['fn'], action['preprocessed'], action['response'])
            with st.expander(f"명령 {i}: {action['raw']}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("fn", action["fn"]); c2.metric("dir", action["param_direction"]); c3.metric("room", action["room"])
                resp = api_resp if api_resp else action['response']
                st.write(f"💬 {'[API] ' if api_resp else ''}{resp}")

    st.markdown("---")
    st.subheader("📡 실시간 API 상태")
    if st.button("날씨/미세먼지 조회", key="btn_weather"):
        w, a = get_weather(), get_air()
        if w:
            st.write(f"🌤 **{w['condition']}** {w['temp']}°C (체감 {w['feels']}°C) | 습도 {w['humidity']}% | 강수 {w['rain_prob']}%")
            if w['max_temp']: st.write(f"최고 {w['max_temp']}°C / 최저 {w['min_temp']}°C")
        else:
            st.error("날씨 API 실패")
        if a:
            st.write(f"💨 PM10 {a['pm10']}㎍/m³ **{a['pm10_grade']}** | PM2.5 {a['pm25']}㎍/m³ **{a['pm25_grade']}**")
        n = get_news()
        if n:
            st.write("📰 " + " / ".join(n))

# ── 오른쪽: 멀티턴 ───────────────────────────────────────────
with col_right:
    st.subheader("🔄 멀티턴 대화 (DST 활성)")

    if "multiturn_history" not in st.session_state:
        st.session_state.multiturn_history = []

    for turn in st.session_state.multiturn_history:
        with st.chat_message("user"):
            st.write(turn["input"])
        with st.chat_message("assistant"):
            if turn.get("api_used"):
                st.success(f"**[API]** {turn['response']}")
            else:
                st.write(f"**{turn['response']}**")
            st.caption(
                f"fn=`{turn['fn']}` | exec=`{turn['exec_type']}` | "
                f"dir=`{turn['param_direction']}` | room=`{turn['room']}`"
                + (" | 🔁 DST" if turn.get("dst_applied") else "")
            )

    multiturn_input = st.chat_input("발화 입력")
    if multiturn_input:
        p = get_multi_pipeline()
        r = p.process(multiturn_input.strip(), use_dst=True)
        api_resp = make_api_response(r['fn'], r['preprocessed'], r['response'])
        final_resp = api_resp if api_resp else r['response']
        st.session_state.multiturn_history.append({
            "input": multiturn_input,
            "response": final_resp,
            "fn": r["fn"], "exec_type": r["exec_type"],
            "param_direction": r["param_direction"], "room": r["room"],
            "dst_applied": r["dst_applied"], "api_used": bool(api_resp),
        })
        st.rerun()

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🗑️ 초기화"):
            st.session_state.multiturn_history = []
            get_multi_pipeline().reset_dst()
            st.rerun()
    with col_b2:
        st.caption(f"{len(st.session_state.multiturn_history)}턴")

    st.markdown("---")
    st.subheader("📋 시나리오")
    scenarios = {
        "귀가": ["다 왔어", "거실 불 켜줘", "에어컨도 켜줘", "온도 23도로"],
        "취침 준비": ["불 좀 낮춰줘", "에어컨 꺼줘", "커튼 닫아줘"],
        "날씨 대응": ["오늘 날씨 어때?", "미세먼지는?", "환기 켜줘"],
        "가족 요청": ["남편이 춥다고 해", "난방 좀 높여줘", "됐어"],
    }
    selected = st.selectbox("시나리오", list(scenarios.keys()))
    if st.button("▶️ 실행"):
        st.session_state.multiturn_history = []
        p = get_multi_pipeline()
        p.reset_dst()
        for turn_text in scenarios[selected]:
            r = p.process(turn_text, use_dst=True)
            api_resp = make_api_response(r['fn'], r['preprocessed'], r['response'])
            st.session_state.multiturn_history.append({
                "input": turn_text,
                "response": api_resp if api_resp else r['response'],
                "fn": r["fn"], "exec_type": r["exec_type"],
                "param_direction": r["param_direction"], "room": r["room"],
                "dst_applied": r["dst_applied"], "api_used": bool(api_resp),
            })
        st.rerun()

# ── 배치 ─────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🧪 배치 테스트"):
    batch_text = st.text_area("발화 목록 (한 줄에 하나)", height=120, key="batch_input")
    if st.button("실행", key="btn_batch") and batch_text.strip():
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        p = get_single_pipeline(); p.reset_dst()
        rows = []
        for line in lines:
            r = p.process(line, use_dst=False)
            api_resp = make_api_response(r['fn'], r['preprocessed'], r['response'])
            rows.append({"발화": r["raw"], "fn": r["fn"], "dir": r["param_direction"],
                         "room": r["room"], "응답": api_resp if api_resp else r["response"],
                         "API": "✅" if api_resp else ""})
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
