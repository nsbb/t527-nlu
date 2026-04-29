#!/usr/bin/env python3
"""NLU 테스트 웹앱 — Streamlit
실행: streamlit run scripts/nlu_test_app.py --server.port 8501
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(page_title="NLU 테스트", layout="wide", page_icon="🏠")

# ── 모델 로드 (캐시) ──────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    from deployment_pipeline_v2 import DeploymentPipelineV2
    return DeploymentPipelineV2()

pipeline = load_pipeline()

# ── UI ────────────────────────────────────────────────────────
st.title("🏠 르엘 NLU 테스트")
st.caption("발화를 입력하면 fn/exec/dir/room과 AI기대응답을 바로 확인할 수 있습니다.")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📝 단일 발화 테스트")

    utterance = st.text_input(
        "발화 입력",
        placeholder="예: 거실 불 켜줘",
        key="single_input"
    )
    use_dst = st.checkbox("DST 사용 (이전 컨텍스트 반영)", value=False, key="dst_single")

    if st.button("분석", type="primary", key="btn_single") or utterance:
        if utterance.strip():
            pipeline.reset_dst()
            r = pipeline.process(utterance.strip(), use_dst=use_dst)

            # 결과 표시
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("fn", r["fn"])
            col2.metric("exec_type", r["exec_type"])
            col3.metric("direction", r["param_direction"])
            col4.metric("room", r["room"])

            st.markdown(f"**💬 AI 응답:** {r['response']}")

            with st.expander("상세 정보"):
                st.json({
                    "raw": r["raw"],
                    "preprocessed": r["preprocessed"],
                    "fn": r["fn"],
                    "exec_type": r["exec_type"],
                    "param_direction": r["param_direction"],
                    "room": r["room"],
                    "rooms": r["rooms"],
                    "value": r["value"],
                    "dst_applied": r["dst_applied"],
                })

    st.markdown("---")
    st.subheader("⚡ 복합 명령 테스트")
    compound = st.text_input(
        "복합 발화 입력",
        placeholder="예: 거실 불 끄고 에어컨 켜줘",
        key="compound_input"
    )
    if st.button("복합 분석", key="btn_compound") and compound.strip():
        pipeline.reset_dst()
        result = pipeline.process_compound(compound.strip())
        st.markdown(f"**복합 명령 여부:** {'✅ 예' if result['is_compound'] else '❌ 아니오'}")
        for i, action in enumerate(result['actions'], 1):
            with st.expander(f"명령 {i}: {action['raw']}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("fn", action["fn"])
                c2.metric("direction", action["param_direction"])
                c3.metric("room", action["room"])
                st.markdown(f"**💬 응답:** {action['response']}")


with col_right:
    st.subheader("🔄 멀티턴 대화 테스트")

    if "multiturn_history" not in st.session_state:
        st.session_state.multiturn_history = []
        st.session_state.multiturn_pipeline = load_pipeline()

    # 대화 히스토리 표시
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.multiturn_history:
            with st.chat_message("user"):
                st.write(turn["input"])
            with st.chat_message("assistant"):
                st.markdown(f"**{turn['response']}**")
                st.caption(
                    f"fn=`{turn['fn']}` | exec=`{turn['exec_type']}` | "
                    f"dir=`{turn['param_direction']}` | room=`{turn['room']}`"
                    + (" | 🔁 DST" if turn.get("dst_applied") else "")
                )

    # 입력
    multiturn_input = st.chat_input("발화 입력 (멀티턴 DST 활성)", key="multiturn_input")

    if multiturn_input:
        r = st.session_state.multiturn_pipeline.process(multiturn_input.strip(), use_dst=True)
        st.session_state.multiturn_history.append({
            "input": multiturn_input,
            "response": r["response"],
            "fn": r["fn"],
            "exec_type": r["exec_type"],
            "param_direction": r["param_direction"],
            "room": r["room"],
            "dst_applied": r["dst_applied"],
        })
        st.rerun()

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ 대화 초기화", key="clear_history"):
            st.session_state.multiturn_history = []
            st.session_state.multiturn_pipeline.reset_dst()
            st.rerun()
    with col_btn2:
        st.caption(f"총 {len(st.session_state.multiturn_history)}턴")

    # 시나리오 빠른 실행
    st.markdown("---")
    st.subheader("📋 시나리오 빠른 실행")
    scenarios = {
        "귀가 시나리오": [
            "다 왔어", "거실 불 켜줘", "에어컨도 켜줘", "온도 23도로 맞춰줘"
        ],
        "취침 준비": [
            "불 좀 낮춰줘", "에어컨 꺼줘", "커튼 닫아줘", "잘게요"
        ],
        "날씨 대응": [
            "오늘 날씨 어때?", "비 온다고? 창문 좀 닫아줘", "집이 좀 습하네", "환기 켜줘"
        ],
        "가족 요청": [
            "아이가 춥다고 해", "난방 좀 높여줘", "안방이 좀 따뜻해졌어?", "됐어"
        ],
    }
    selected = st.selectbox("시나리오 선택", list(scenarios.keys()))
    if st.button("▶️ 시나리오 실행", key="run_scenario"):
        st.session_state.multiturn_history = []
        st.session_state.multiturn_pipeline.reset_dst()
        p = st.session_state.multiturn_pipeline
        for turn_text in scenarios[selected]:
            r = p.process(turn_text, use_dst=True)
            st.session_state.multiturn_history.append({
                "input": turn_text,
                "response": r["response"],
                "fn": r["fn"],
                "exec_type": r["exec_type"],
                "param_direction": r["param_direction"],
                "room": r["room"],
                "dst_applied": r["dst_applied"],
            })
        st.rerun()

# ── 하단: 배치 테스트 ─────────────────────────────────────────
st.markdown("---")
with st.expander("🧪 배치 테스트 (여러 줄 한 번에)"):
    batch_text = st.text_area(
        "발화 목록 (한 줄에 하나씩)",
        placeholder="거실 불 켜줘\n에어컨 꺼줘\n오늘 날씨 어때\n",
        height=150,
        key="batch_input"
    )
    if st.button("배치 실행", key="btn_batch") and batch_text.strip():
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        pipeline.reset_dst()
        results = []
        for line in lines:
            r = pipeline.process(line, use_dst=False)
            results.append({
                "발화": r["raw"],
                "fn": r["fn"],
                "exec": r["exec_type"],
                "dir": r["param_direction"],
                "room": r["room"],
                "응답": r["response"],
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(results), use_container_width=True)
