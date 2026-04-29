#!/usr/bin/env python3
"""NLU 테스트 웹앱 — Streamlit
실행: streamlit run scripts/nlu_test_app.py --server.port 8501

test_api.py를 그대로 import해서 사용 (API 로직 단일 소스).
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import streamlit as st
st.set_page_config(page_title="NLU 테스트", layout="wide", page_icon="🏠")

# ── test_api.py의 API 함수 그대로 import ─────────────────────
import test_api as api
from test_api import (
    LAT, LON, DISTRICT,
    fill_placeholders, weather_response, air_response, news_response,
)

# 네트워크 호출은 Streamlit 캐싱
@st.cache_data(ttl=60)
def get_weather():
    return api.get_weather()

@st.cache_data(ttl=60)
def get_air():
    return api.get_air()

@st.cache_data(ttl=300)
def get_news(n=3):
    return api.get_news(n)


def make_api_response(fn, text, raw_resp):
    """fn에 따라 API 응답 생성 (날씨/미세먼지/뉴스)."""
    if fn == 'weather_query':
        w, a = get_weather(), get_air()
        if re.search(r'미세먼지|공기|대기|환기|마스크', text):
            ar = air_response(text, a)
            if ar:
                return ar
        wr = weather_response(text, w)
        if wr:
            return wr
        return fill_placeholders(raw_resp, w, a)
    elif fn == 'news_query':
        n = get_news()
        if n:
            return news_response(n) if callable(news_response) else (
                '주요 뉴스입니다.\n' + '\n'.join(f'{i+1}. {t}' for i, t in enumerate(n)))
    # 비날씨 응답에 OO 있으면 치환 시도
    if 'OO' in raw_resp or '00도' in raw_resp:
        w = get_weather()
        a = get_air() if '미세먼지' in raw_resp else None
        filled = fill_placeholders(raw_resp, w, a)
        if filled != raw_resp:
            return filled
    return None


# ── 파이프라인 (단일/멀티턴 분리) ────────────────────────────
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

# 왼쪽: 단일 발화 ────────────────────────────────────────────
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
            if w.get('max_temp'): st.write(f"최고 {w['max_temp']}°C / 최저 {w['min_temp']}°C")
        else:
            st.error("날씨 API 실패")
        if a:
            st.write(f"💨 PM10 {a['pm10']}㎍/m³ **{a['pm10_grade']}** | PM2.5 {a['pm25']}㎍/m³ **{a['pm25_grade']}**")
        n = get_news()
        if n:
            st.write("📰 " + " / ".join(n))

# 오른쪽: 멀티턴 ─────────────────────────────────────────────
with col_right:
    st.subheader("🔄 멀티턴 대화 (DST 활성)")

    if "multiturn_history" not in st.session_state:
        st.session_state.multiturn_history = []

    # 현재 집 상태 박스 (항상 표시)
    home_state_now = get_multi_pipeline().home_state.summary_kr()
    if home_state_now:
        st.info(f"🏠 **집 상태:** {home_state_now}")
    else:
        st.caption("🏠 집 상태: (아직 제어된 기기 없음)")

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
            "home_state": r.get("home_state"),
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

# 배치 ────────────────────────────────────────────────────────
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
